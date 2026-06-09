from typing import Optional, Sequence
import os
import time
import json
import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
from PIL import Image

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from metis.extern.hf.configuration_prismatic import DexVLAConfig
from metis.extern.hf.modeling_prismatic import DexVLAForActionPrediction
from metis.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 48
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

def get_processor(pretrained_checkpoint: str):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(pretrained_checkpoint, trust_remote_code=True)
    return processor

from metis.models.policy.transformer_utils import MAPBlock


class ActionDecoderHead(torch.nn.Module):
    def __init__(self, window_size = 5):
        super().__init__()
        self.attn_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = 512, n_heads = 8)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = 512, n_heads = 8)

        self.proprio_proj = nn.Sequential(
                                nn.Linear(7, 512), 
                                nn.GELU(),
                                nn.Linear(512, 512)
                            )

        self.proj = nn.Sequential(
                                nn.Linear(1024, 7 * window_size),
                                # nn.Tanh(),
                    )

    def forward(self, latent_action_tokens, visual_embed, proprio=None):

        latent_action_tokens = latent_action_tokens[:, -4:]

        proprio = self.proprio_proj(proprio)
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(torch.cat([self.attn_pool(latent_action_tokens, init_embed=visual_embed), proprio], dim=-1))
        
        return action


class ActionDecoder(nn.Module):
    def __init__(self,window_size=5):
        super().__init__()
        self.net = ActionDecoderHead(window_size=window_size)
        self.window_size = window_size
        self.temporal_size = window_size
        self.temporal_size = 8
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

        # Action chunking with temporal aggregation
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]


    def reset(self):
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

    
    def forward(self, latent_actions, visual_embed, proprio=None):
        # Forward action decoder
        # NOTE: We take the last 8 actions in an action chunk for non-blocking controller to tackle possible mismatch led by model latency
        pred_action = self.net(latent_actions.to(torch.float), visual_embed.to(torch.float), proprio).reshape(-1, self.window_size, 7)[:, self.window_size - self.temporal_size:]
        pred_action = np.array(pred_action.tolist())
        
        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

        # Add to action buffer
        self.action_buffer[0] = pred_action  
        self.action_buffer_mask[0] = np.array([True] * self.temporal_mask.shape[0], dtype=np.bool_)

        # Ensemble temporally to predict action
        action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights, axis=0) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)


        return action_prediction
# Initialize DexVLA model
def get_dexvla(pretrained_checkpoint: str):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    AutoConfig.register("dexvla", DexVLAConfig)
    AutoImageProcessor.register(DexVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(DexVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(DexVLAConfig, DexVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        load_in_4bit=False,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla

class DexVLAInference:
    def __init__(
        self,
        saved_model_path: str = "checkpoint/univla-7b",
        unnorm_key: Optional[str] = None,
        horizon: int = 1,
        pred_action_horizon: int = 8,
        exec_horizon: int = 1,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load model
        self.vla = get_dexvla(saved_model_path).cuda()
        self.processor = get_processor(saved_model_path)

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.unnorm_key = unnorm_key

        self.task = None
        self.task_description = None
        self.num_image_history = 0
        self.temporal_size = pred_action_horizon
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 48))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

        # Action chunking with temporal aggregation
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]
        self.prev_hist_action = ['']
        self.motion_dynamics = None

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 48))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)
        self.prev_hist_action = ['']

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, proprio = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: torch.Tensor of shape (3, H, W)
            task_description: str; task description
            proprio: torch.Tensor; proprioceptive state of the robot
        Output:
            action: numpy.array of shape (1, 48); processed action to be sent to the robot arms
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        image = (image.squeeze().permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)

        image: Image.Image = Image.fromarray(image)

        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"


        # predict action (7-dof; un-normalize for bridgev2)
        inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            pred_action, visual_embed, generated_ids = self.vla.predict_latent_action(**inputs, unnorm_key=self.unnorm_key, proprio=proprio, do_sample=True, temperature=0.75, top_p = 0.9)
            # Decode latent action tokens for debugging (e.g., "<ACT_...>" fragments).
            try:
                gen_ids = generated_ids[0].detach().cpu().tolist()
                decoded = self.processor.decode(gen_ids, skip_special_tokens=False).strip()
                print("generated_ids:", gen_ids)
                print("decoded_generated_ids:", decoded)
            except Exception as e:
                print("decode generated_ids failed:", repr(e))
            print("motion_dynamics:", self.vla.motion_dynamics.shape)
            self.motion_dynamics = self.vla.motion_dynamics
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

        # Add to action buffer
        self.action_buffer[0] = pred_action.cpu().numpy()
        self.action_buffer_mask[0] = np.array([True] * self.temporal_mask.shape[0], dtype=np.bool_)

        # Ensemble temporally to predict action
        action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights, axis=0) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)

        return pred_action
    
