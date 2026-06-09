"""
deploy.py

Starts VLA server which the client can query to get robot actions.
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import numpy as np
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import torchvision
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import time

from real_world_deployment import DexVLAInference
from metis.extern.hf.modeling_prismatic import DexVLAForActionPrediction, ActionDecoder

# === Server Interface ===
class DexVLAServer:
    def __init__(self, cfg) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given observation + instruction.
        """
        self.cfg = cfg

        self.policy = DexVLAInference(saved_model_path=cfg.pretrained_checkpoint, pred_action_horizon=32)
        self.policy.vla.action_decoder = ActionDecoder(window_size=32, input_dim=48, vis_dim=4096, hidden_dim=512).to("cuda:0")
       
        decoder_path = os.path.join(cfg.pretrained_checkpoint, "action_decoder.pt")
        checkpoint = torch.load(decoder_path)
        updated_checkpoint = {}
        for key, value in checkpoint.items():
            # 优先使用 modules_to_save.default 的权重
            if key.startswith('modules_to_save.default.'):
                print(f"{key}: {type(value)} | Shape: {getattr(value, 'shape', 'No shape')} | dtype: {getattr(value, 'dtype', 'No dtype')}")
                new_key = key.replace('modules_to_save.default.', '')
                updated_checkpoint[new_key] = value
        self.policy.vla.action_decoder.to(torch.float)
        self.policy.vla.action_decoder.load_state_dict(updated_checkpoint)
        self.unnorm_key = cfg.unnorm_key


    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])
            start_time = time.time()
            observation = payload
            instruction = observation["instruction"]
            print("instruction:", instruction)
            img = observation["image"]
            proprio = torch.tensor(observation["state"], dtype=torch.float32).to("cuda:0")

            unnorm_key = self.unnorm_key
            action_norm_stats = self.policy.vla.get_action_stats(unnorm_key)
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            mask = torch.tensor(mask, dtype=torch.bool)
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
            action_high = torch.from_numpy(np.array(action_norm_stats["max"]))
            action_low = torch.from_numpy(np.array(action_norm_stats["min"]))
            proprio_norm_stats = self.policy.vla.get_proprio_stats(unnorm_key)
            proprio_high, proprio_low = np.array(proprio_norm_stats["max"]), np.array(proprio_norm_stats["min"])
            proprio_high = torch.from_numpy(np.array(proprio_norm_stats["max"]))
            proprio_low = torch.from_numpy(np.array(proprio_norm_stats["min"]))
            print("proprio:", proprio)
            # 确保所有张量在同一个设备和数据类型
            action_high = action_high.to(device=proprio.device, dtype=proprio.dtype)
            action_low = action_low.to(device=proprio.device, dtype=proprio.dtype)
            proprio_high = proprio_high.to(device=proprio.device, dtype=proprio.dtype)
            proprio_low = proprio_low.to(device=proprio.device, dtype=proprio.dtype)
            mask = mask.to(device=proprio.device)
            
            proprio = torch.where(
                mask,
                torch.clamp(2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1, -1, 1),
                proprio,
            )
            img = torch.from_numpy(np.array(img))      # (H, W, C), uint8
            # # img = img[..., [2, 1, 0]]                            # BGR -> RGB
            # img = img.permute(2, 0, 1).contiguous()              # (C, H, W)
            # # img = torchvision.transforms.functional.resize(img, (224, 224), antialias=True)
            # img = torchvision.transforms.Resize((224, 224))(torch.flip(img,(0,)))
            img = img.float().div(255.0)                         # [0,1]
            # 若模型需要批维+GPU：
            resized_curr_image = img.to("cuda:0",dtype=torch.float)   # (3, 224, 224)
            all_actions = self.policy.step(resized_curr_image, instruction, proprio)
            all_actions = all_actions[0]
            print("Inference time:", time.time() - start_time)
            
            for i in range(all_actions.shape[0]):
                all_actions[i] = torch.where(
                    mask,
                    0.5 * (all_actions[i] + 1) * (action_high - action_low) + action_low,
                    all_actions[i],
                )
            all_actions = all_actions.cpu().numpy()
            if double_encode:
                return JSONResponse(json_numpy.dumps(all_actions))
            else:
                return JSONResponse(all_actions)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8777                                               # Host Port

    pretrained_checkpoint: Union[str, Path] = "pretrained_checkpoint"
    unnorm_key: Union[str, Path] = "real_g1"       #put_the_cola_into_the_basket         # Action un-normalization key
    seed: int = 7                                    # Random Seed (for reproducibility)


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = DexVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
