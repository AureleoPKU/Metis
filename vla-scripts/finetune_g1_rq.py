import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import tqdm
import pickle
from ema_pytorch import EMA
from accelerate import PartialState, Accelerator
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from metis.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from metis.vla.action_tokenizer import ActionTokenizer
from metis.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from metis.vla.datasets import RLDSBatchTransformCoTLatentAction, RLDSBatchTransform_withoutCoTLatentAction, RLDSDataset
from metis.util.data_utils import PaddedCollatorForActionPrediction


from metis.extern.hf.configuration_prismatic import DexVLAConfig
from metis.extern.hf.modeling_prismatic import DexVLAForActionPrediction, ActionDecoder
from metis.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FinetuneConfig:
    # Directory Paths
    data_root_dir: Path = Path("/share/project/fyk/dataset/real")     # Path to Open-X dataset directory

    vla_path: str = "/share/project/fyk/UniDex/vla-scripts/hf_ckpt/RQ_HandLatentActionModel_LAM_lw_action_decoder_chunk32_new_step-065000-epoch-09-loss-0.7056"            # Path to your local Metis path
    motion_dynamics_path: str = "/share/project/fyk/UniDex/ckpt/rq_lam_chunk32/epoch=2-step=300000.ckpt"
    # vla_path: str = "local METIS path"
    # motion_dynamics_path: str = "Motion Dynamics Model path"

    dataset_name: str = "real_g1"                                    # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 4                                             # Fine-tuning batch size
    max_steps: int = 100_000                                          # Max number of fine-tuning steps
    save_steps: int = 10_000                                          # Interval for checkpoint saving
    learning_rate: float = 3.5e-4                                   # Fine-tuning learning rate
    # num_steps_before_decay: int = 20_000                            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 2                                # Gradient accumulation steps
    image_aug: bool = True                                         # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_00                               # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = False                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    # LAM setting
    codebook_size: int = 16
    motion_codebook_size: int = 512
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 8
    lam_num_heads: int = 12
    
    use_diffusion_head: bool = False
    window_size: int = 32
    freeze_vla: bool = False
    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "fientune-dexgarment"                          # Name of W&B project to log to (use default!)
    wandb_entity: str = "aureleo"                              # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning MEITS Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    exp_id += f'=w-LowLevelDecoder-ws-{cfg.window_size}'

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("dexvla", DexVLAConfig)
    AutoImageProcessor.register(DexVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(DexVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(DexVLAConfig, DexVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
           
    vla.action_decoder = ActionDecoder(window_size=cfg.window_size, input_dim=48, vis_dim=4096, hidden_dim=512)
    vla.action_decoder.to(torch.float32)
    vla.window_size = cfg.window_size
    vla.config.window_size = cfg.window_size
    processor.window_size = cfg.window_size
    
    vla.input_dim = 48
    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    def collect_lora_targets(model, include_prefixes=("language_model", "projector", "vision_backbone")):
        target_names = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name.startswith(include_prefixes):
                    target_names.append(name)  # 只保留最后一段名字，比如 qkv, fc1
        return list(set(target_names))

    if cfg.use_lora:
        target_modules = collect_lora_targets(vla)
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            # target_modules=[
            #     "language_model.*",
            #     "vision_backbone.*",
            #     "projector.*",
            # ],
            init_lora_weights="gaussian",
            modules_to_save=["action_decoder"],
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()
    
    trainable_total_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    print('Total Trainable Params: ', trainable_total_params)
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    
    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = int(cfg.max_steps * 0.8), gamma=0.1)

    # optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    # scheduler = MultiStepLR(
    #     optimizer,
    #     milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
    #     gamma=0.1,  # Multiplicative factor of learning rate decay
    # )

    from motion_tokenizer.genie.modules.motion_tokenizer import RQ_HandLatentActionModel

    motion_dynamics_model = RQ_HandLatentActionModel(
        in_dim=3,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size,
        enc_blocks=cfg.lam_enc_blocks,
        dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=0.,
    )
    motion_dynamics_ckpt = torch.load(cfg.motion_dynamics_path, map_location='cpu')['state_dict']
    new_ckpt = {}
    for key in motion_dynamics_ckpt.keys():
        new_ckpt[key.replace("lam.", "")] = motion_dynamics_ckpt[key]

    motion_dynamics_model.load_state_dict(new_ckpt, strict=True)
    motion_dynamics_model = motion_dynamics_model.to(device_id).eval()

    batch_transform = RLDSBatchTransform_withoutCoTLatentAction(
        motion_dynamics_model,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        image_transform_lam=transforms.ToTensor(),
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        window_size=cfg.window_size
    )
    
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=cfg.window_size + 1,        # for constructing history latent actions
        training_phase='post-training',
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right", num_padding=0,
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    
    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
    # if accelerator.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            batch["input_ids"] = batch["input_ids"].to(device_id)
            batch["attention_mask"] = batch["attention_mask"].to(device_id)
            batch["labels"] = batch["labels"].to(device_id)
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id)
            batch['actions'] = batch['actions'].to(device_id)
            
            # Forward pass
            vla_output, action_loss, loss_one_step, action_tokens, loss_batch = vla(batch)
            loss = action_loss if cfg.freeze_vla else action_loss + vla_output.loss
            # loss = action_loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps
            torch.nn.utils.clip_grad_norm_(vla.parameters(), max_norm=1.)

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = vla_output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > 32002

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()


            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)

            # Push Metrics to W&B (every 5 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 5 == 0:
                
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "latent_action_accuracy": smoothened_action_accuracy,
                        "action_loss": action_loss.item(),
                        "action_loss_1step": loss_one_step.item(),
                        "lr": optimizer.state_dict()['param_groups'][0]['lr'],
                    },
                    step=gradient_step_idx,
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    if not cfg.freeze_vla:
                        processor.save_pretrained(run_dir)
                        vla.module.save_pretrained(save_dir)

                    # Save low-level policy
                    # torch.save(vla.module.action_decoder.state_dict(), str(run_dir) + f'/action_decoder-{gradient_step_idx}.pt')

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    base_vla.action_decoder = ActionDecoder(window_size=cfg.window_size, input_dim=48, vis_dim=4096, hidden_dim=512)
                    base_vla.action_decoder.to(torch.float32)
                    base_vla.window_size = cfg.window_size
                    base_vla.input_dim = 48
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    merged_vla.config.window_size = cfg.window_size
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)
                            torch.save(vla.module.action_decoder.state_dict(), checkpoint_dir / 'action_decoder.pt')

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)
    finetune()
