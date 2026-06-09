from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import wandb
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from accelerate import PartialState

OptimizerCallable = Callable[[Iterable], Optimizer]

from genie.modules import RQ_HandLatentActionModel
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

class Action_Tokenizer(LightningModule):
    """A action tokenizer for video reconstruction and hand action reconstruction"""

    def __init__(
            self,
            image_channels: int = 3,
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_num_latents: int = 48,
            lam_num_latents_video: int = 8,
            lam_num_latents_uni: int = 8,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            vq_beta: float = 0.25,
            hand_beta: float = 1.0,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            task_name: str = 'motion_dynamics',
            stage: str = 'stage-1',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            stage_one_ckpt: str = None,
            resume_from_checkpoint: str = None,
    ) -> None:
        super(Action_Tokenizer, self).__init__()
       
        lam = RQ_HandLatentActionModel

        self.lam = lam(
                    in_dim=image_channels,
                    model_dim=lam_model_dim,
                    latent_dim=lam_latent_dim,
                    num_latents=lam_num_latents,
                    patch_size=lam_patch_size,
                    enc_blocks=lam_enc_blocks,
                    dec_blocks=lam_dec_blocks,
                    num_heads=lam_num_heads,
                    dropout=lam_dropout,
                )

        if stage_one_ckpt and path.exists(stage_one_ckpt):
            lam_ckpt = torch.load(stage_one_ckpt)['state_dict']
            stage1_ckpt = {}
            for key in lam_ckpt.keys():
                if 'vq' in key or 'action_latent' in key:
                    stage1_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
            self.lam.load_state_dict(stage1_ckpt, strict=False)

        if resume_from_checkpoint and path.exists(resume_from_checkpoint):
            try:
                print(f"Loading checkpoint from: {resume_from_checkpoint}")
                checkpoint = torch.load(resume_from_checkpoint, map_location='cpu')

                # checkpoint
                print(f"Checkpoint keys: {list(checkpoint.keys())}")

                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print(f"State dict keys count: {len(state_dict.keys())}")

                    # state_dict
                    missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

                    if missing_keys:
                        print(f"Missing keys: {missing_keys[:10]}...")
                    if unexpected_keys:
                        print(f"Unexpected keys: {unexpected_keys[:10]}...")

                    print(f"Successfully loaded checkpoint with {len(state_dict)} parameters")


                    if 'epoch' in checkpoint:
                        print(f"Resumed from epoch: {checkpoint['epoch']}")
                    if 'global_step' in checkpoint:
                        print(f"Resumed from global step: {checkpoint['global_step']}")


                    self.verify_model_loaded()

                else:
                    print("Warning: No 'state_dict' found in checkpoint")

            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Continuing without checkpoint...")
        else:
            print("No checkpoint specified or checkpoint not found")


        self.lam_num_latents = lam_num_latents
        self.vq_beta = vq_beta
        self.hand_beta = hand_beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair
        self.save_hyperparameters()

        self.task_name = task_name
        task_name = f"{task_name}_{lam_num_latents}_{vq_beta}_{hand_beta}"
        self.distributed_state = PartialState()
        if self.distributed_state.is_main_process:
            wandb.init(name=task_name, reinit=True)
            # wandb.init(entity="zhaojunkai1515-Ruhr University Bochum", project="lam_video_hand", name=task_name)



    def verify_model_loaded(self):

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


        if hasattr(self.lam, 'vq_video'):
            print(f"VQ video codebook size: {self.lam.vq_video.num_latents}")
        if hasattr(self.lam, 'vq_hand'):
            print(f"VQ hand codebook size: {self.lam.vq_hand.codebook_size}")

    def summarize_model_devices(self, model):
        param_devs = {p.device.type for p in model.parameters()}
        buff_devs  = {b.device.type for b in model.buffers()}
        print("param devices:", param_devs)
        print("buffer devices:", buff_devs)

    def find_cpu_modules(self, model):
        cpu_modules = []
        for name, module in model.named_modules():
            has_cpu_param = any(p.device.type == 'cpu' for p in module.parameters(recurse=False))
            has_cpu_buff  = any(b.device.type == 'cpu' for b in module.buffers(recurse=False))
            if has_cpu_param or has_cpu_buff:
                cpu_modules.append(name or 'root')
        return cpu_modules

    def shared_step(self, batch: Dict) -> Tuple:
        # batch: keys['videos', 'task_instruction', 'action', 'dataset_names']

        outputs = self.lam(batch)
        if self.global_step % 500 == 0:
            print('indices_hand:',outputs["indices_hand"])
            print('indices_video:',outputs["indices_video"])

        gt_future_frames = outputs["video_target"]
        gt_future_actions = outputs["action_target"]

        video_loss = ((gt_future_frames - outputs["video_recon"]) ** 2).mean()
        hand_loss = ((gt_future_actions - outputs["action_recon"]) ** 2).mean()

        mse_loss = video_loss  + hand_loss
        q_loss =  ((outputs["emb_video"].detach() - outputs["z_video"]) ** 2).mean()
        commit_loss =((outputs["emb_video"] - outputs["z_video"].detach()) ** 2).mean()
        commit_loss_hand =(outputs["losses_hand_commit"]).mean()

        loss = mse_loss + q_loss + self.vq_beta * commit_loss + commit_loss_hand * 0.1

        # Compute latent code usage in separated codebook
        unique, counts = torch.unique(outputs["indices_video"], return_counts=True)
        index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).cuda()
        index_counts[unique] = counts
        code_usage_video = (index_counts != 0).float().mean()
        unique_hand, counts_hand = torch.unique(outputs["indices_hand"], return_counts=True)
        index_counts_hand = torch.zeros(512, dtype=torch.long).cuda()
        index_counts_hand[unique_hand] = counts_hand
        code_usage_hand = (index_counts_hand != 0).float().mean()
        # shared_codebook=True
        cb = self.lam.vq_hand.layers[0]._codebook
        usage_ratio = (cb.cluster_size > 0).float().mean()
        alive_ratio = (cb.cluster_size >= cb.threshold_ema_dead_code).float().mean()  # “”

        loss_logs = (
            ("mse_loss", mse_loss),
            ("q_loss", q_loss),
            ("commit_loss", commit_loss),
            ("commit_loss_hand", commit_loss_hand),
            ("code_usage_video", code_usage_video),
            ("code_usage_hand", code_usage_hand),
            ("code_usage_ratio_hand", usage_ratio),
            ("code_usage_alive_ratio_hand", alive_ratio),
            ("hand_loss", hand_loss),
            ("video_loss", video_loss),
        )

        return outputs, loss, loss_logs

    def check_unused_parameters(self):

        unused_params = []
        zero_grad_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    unused_params.append(name)
                elif torch.allclose(param.grad, torch.zeros_like(param.grad), atol=1e-8):
                    zero_grad_params.append(name)

        if unused_params or zero_grad_params:
            print("\nWarning: found unused parameters:")
            if unused_params:
                print(f"  Parameters without gradients ({len(unused_params)}):")
                for param_name in unused_params:
                    print(f"    - {param_name}")
            if zero_grad_params:
                print(f"  Parameters with zero gradients ({len(zero_grad_params)}):")
                for param_name in zero_grad_params:
                    print(f"    - {param_name}")
        else:
            print("\nAll trainable parameters have valid gradients.")

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        try:
            if self.global_step == 0:
                print("LightningModule device:", self.device)
                self.summarize_model_devices(self.lam)
                print("cpu modules:", self.find_cpu_modules(self.lam))

            # Compute the training loss
            outputs, loss, aux_losses = self.shared_step(batch)

            # Log the training loss
            self.log_dict(
                {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True
            )

            if self.distributed_state.is_main_process:
                wandb.log({**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}})

            return loss

        except ValueError as e:
            if "NaN/Inf" in str(e):
                print(f"🛑 Training stopped due to NaN/Inf loss")

                return None  # Nonebatch
            else:
                raise e
        except Exception as e:
            print(f"❌ Unexpected error in training_step: {e}")
            raise e

    def training_step_old(self, batch: Dict, batch_idx: int) -> Tensor:
        if self.global_step == 0:
            print("LightningModule device:", self.device)
            self.summarize_model_devices(self.lam)
            print("cpu modules:", self.find_cpu_modules(self.lam))

        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # self.check_unused_parameters()
        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        if self.distributed_state.is_main_process:
            wandb.log({**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}})

        return loss


    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def on_train_epoch_end(self):

        self.lam.vq_video.random_restart()
        self.lam.vq_video.reset_usage()

    def on_test_epoch_end(self):
        if self.make_data_pair:
            completed = len(listdir("output_pairs"))
            todo_name = listdir("../data/retro")[completed]
            makedirs(f"output_pairs/{todo_name}")
            top_indices = torch.topk(self.lam.vq.usage, 16, largest=True, sorted=True).indices
            top_latents = self.lam.vq.codebook(top_indices)
            torch.save(top_latents, f"output_pairs/{todo_name}/top_16.pt")
            with open(f"output_pairs/{todo_name}/top_16.txt", "w") as f:
                f.write(" ".join([str(i) for i in top_indices.tolist()]))

        self.plot_usage_distribution(self.lam.vq.usage, "unsorted_usage")
        self.plot_usage_distribution(self.lam.vq.usage.sort().values, "sorted_usage")

    def plot_usage_distribution(self, usage, filename):
        data = usage.cpu().numpy()
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        data = data.reshape(2 ** n, -1)
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim
