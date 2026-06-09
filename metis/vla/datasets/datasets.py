"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from einops import repeat

from metis.models.backbones.llm.prompting import PromptBuilder
from metis.models.backbones.vision import ImageTransform
from metis.util.data_utils import tree_map
from metis.vla.action_tokenizer import ActionTokenizer
from metis.vla.fast_action_tokenizer import FastActionTokenizer
from metis.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, IGNORE_INDEX
from metis.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from metis.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights,get_ego_dataset_kwargs_and_weights
from metis.vla.datasets.rlds.utils.data_utils import NormalizationType

from motion_tokenizer.genie.modules.motion_tokenizer import RQ_HandLatentActionModel
 
# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        # print(labels)
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)
        
@dataclass
class RLDSBatchTransformLatentAction:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    image_transform_lam: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    window_size: int = 5 

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        # return action (T, d) proprio (d)
        # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        # print(len(rlds_batch["observation"]["image_primary"]))
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][-1])
        pixel_values = self.image_transform(img)
        with torch.no_grad():
            initial_pixel_values = self.image_transform_lam(img)
            target_pixel_values = self.image_transform_lam(img_k)
            video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0).to(self.action_tokenizer.device)
            if isinstance(self.action_tokenizer, UncontrolledHandVideoLatentActionModel):
                lang_embed, attention_mask = self.action_tokenizer.encode_text(lang)
                lang_embed = self.action_tokenizer.lang_proj(lang_embed).to(self.action_tokenizer.device)
                B, T = video.shape[:2]
                H, W = video.shape[3:5]
                total_len = self.action_tokenizer.num_codes + (H // self.action_tokenizer.patch_size) ** 2 + self.action_tokenizer.num_joints
                attention_mask = torch.cat([torch.ones((B, total_len), device=self.action_tokenizer.device),
                                                attention_mask],
                                                dim=-1)
                window_size = rlds_batch["action"].shape[0]-1
                action = torch.tensor(rlds_batch["action"][[0,window_size], 14:]).unsqueeze(0).to(self.action_tokenizer.device)
                latent_action_idx = self.action_tokenizer.vq_encode(video, action, repeat(lang_embed, 'b l d -> b T l d', T=T), attention_mask.repeat(T, 1))['indices'].squeeze()
                action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]   # [ACT_1, ACT_2, ... ACT_K]
            elif isinstance(self.action_tokenizer, UncontrolledHandVideo_LatentActionModel):
                lang_embed, attention_mask = self.action_tokenizer.encode_text(lang)
                lang_embed = self.action_tokenizer.lang_proj(lang_embed).to(self.action_tokenizer.device)
                B, T = video.shape[:2]
                H, W = video.shape[3:5]
                total_len = self.action_tokenizer.num_codes * 2 + (H // self.action_tokenizer.patch_size) ** 2 + self.action_tokenizer.num_joints
                attention_mask = torch.cat([torch.ones((B, total_len), device=self.action_tokenizer.device),
                                                attention_mask],
                                                dim=-1)
                window_size = rlds_batch["action"].shape[0]-1
                action = torch.tensor(rlds_batch["action"][[0,window_size]]).unsqueeze(0).to(self.action_tokenizer.device)
                output = self.action_tokenizer.vq_encode(video, action, repeat(lang_embed, 'b l d -> b T l d', T=T), attention_mask.repeat(T, 1))
                video_latent_action_idx = output['indices_video'].squeeze()
                hand_action_idx = output['indices_hand'].squeeze()
                action_vocab = [f'<VIDEO_{i.item()}>' for i in video_latent_action_idx] + [f'<DEX_{i.item()}>' for i in hand_action_idx]
                latent_action_idx = torch.cat([video_latent_action_idx, hand_action_idx], dim=0)
            # elif isinstance(self.action_tokenizer, LatentCrossModalFusionModel):
            #     lang_embed, attention_mask = self.action_tokenizer.encode_text(lang)
            #     lang_embed = self.action_tokenizer.lang_proj(lang_embed).to(self.action_tokenizer.device)
            #     B, T = video.shape[:2]
            #     H, W = video.shape[3:5]
            #     total_len = self.action_tokenizer.num_codes + (H // self.action_tokenizer.patch_size) ** 2 + self.action_tokenizer.num_joints
            #     attention_mask = torch.cat([torch.ones((B, total_len), device=self.action_tokenizer.device),
            #                                     attention_mask],
            #                                     dim=-1)
            #     video = torch.stack([self.image_transform_lam(Image.fromarray(rlds_batch["observation"]["image_primary"][i])) for i in [0, 2, 4, 7]], dim=0).unsqueeze(0).to(self.action_tokenizer.device)
            #     action = torch.tensor(rlds_batch["action"][[0, 2, 4, 7]]).unsqueeze(0).to(self.action_tokenizer.device)
            #     num_codes_fusion = 8
            #     num_codes = 8
            #     num_finger = 10
            #     attention_mask_1 = torch.cat([torch.ones((B, num_codes_fusion + num_codes + num_finger), device=self.action_tokenizer.device),
            #                         attention_mask_1],
            #                         dim = -1)
            #     with torch.no_grad():
            #         _, predictions = self.vggt(video)
            #     vggt_features = predictions["vggt_features"]
            #     latent_action_idx = self.action_tokenizer.fusion_encode(video, repeat(lang_embed, 'b l d -> b T l d', T=T), attention_mask.repeat(T, 1), attention_mask_1.repeat(T, 1), vggt_features, action)['indices'].squeeze()
            #     action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]   # [ACT_1, ACT_2, ... ACT_K]
            elif isinstance(self.action_tokenizer, ControllableDINOLatentActionModel):
                latent_action_idx = self.action_tokenizer.vq_encode(video)['indices'].squeeze()
                action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]   # [ACT_1, ACT_2, ... ACT_K]
            else:
                raise TypeError("Unsupported action_tokenizer type.")

        action_tokens = ''
        for i, action in enumerate(action_vocab):
            action_tokens += action

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_tokens},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)


        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action_vocab) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, proprio=rlds_batch["observation"]["proprio"][0], action=rlds_batch["action"][: self.window_size])


@dataclass
class RLDSBatchTransformCoTLatentAction:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    image_transform_lam: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    window_size: int = 5 

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        # return action (T, d) proprio (d)
        # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        latest_reasoning = rlds_batch["observation"]["latest_reasoning"][0].decode()
        use_reasoning = False
        if(rlds_batch["observation"]["reasoning"][0].decode() != ''):
            reasoning = rlds_batch["observation"]["reasoning"][0].decode()
            use_reasoning = True

        # print(len(rlds_batch["observation"]["image_primary"]))
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][-1])
        pixel_values = self.image_transform(img)
        with torch.no_grad():
            initial_pixel_values = self.image_transform_lam(img)
            target_pixel_values = self.image_transform_lam(img_k)
            video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0).to(self.action_tokenizer.device)
    
            if isinstance(self.action_tokenizer, RQ_HandLatentActionModel):
                B, T = video.shape[:2]
                H, W = video.shape[3:5]
                action = torch.tensor(action[:self.window_size]).unsqueeze(0).to(self.action_tokenizer.device)
                outputs = self.action_tokenizer.vq_encode(video, action)
                video_latent_action_idx = outputs['indices_video'].squeeze()
                hand_action_idx = outputs['indices_hand'].squeeze()
                action_vocab = [f'<DYN_{i.item()}>' for i in video_latent_action_idx] + [f'<DEX_{i.item()}>' for i in hand_action_idx]
            else:
                raise TypeError("Unsupported action_tokenizer type.")

        action_tokens = '<BOA>'
        for i, action in enumerate(action_vocab):
            action_tokens += action
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        if not use_reasoning:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}? Here is my previous reasoning: {latest_reasoning}."},
                {"from": "gpt", "value": action_tokens},
            ]
        else:
            reasoning = '<BOR>' + reasoning
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}? Here is my previous reasoning: {latest_reasoning}."},
                {"from": "gpt", "value": reasoning},
            ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
    
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        if not use_reasoning:
            labels[: -(len(action_vocab) + 2)] = IGNORE_INDEX
            # labels[: -(len(action_vocab) + 3)] = IGNORE_INDEX
        else:
            reason_ids = self.base_tokenizer(reasoning, add_special_tokens=False).input_ids
            labels[:-(len(reason_ids) + 1)] = IGNORE_INDEX
            # labels[:-(len(reason_ids) + 2)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        diff_loss_mask = not use_reasoning
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, proprio=rlds_batch["observation"]["proprio"][0:1], action=rlds_batch["action"][: self.window_size], diff_loss_mask=diff_loss_mask)

@dataclass
class RLDSBatchTransform_AdaptiveReasoning_CoTLatentAction:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    image_transform_lam: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    window_size: int = 5 

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        # return action (T, d) proprio (d)
        # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        latest_reasoning = rlds_batch["observation"]["latest_reasoning"][0].decode()
        use_reasoning = False
        if(rlds_batch["observation"]["reasoning"][0].decode() != ''):
            reasoning = rlds_batch["observation"]["reasoning"][0].decode()
            use_reasoning = True

        # print(len(rlds_batch["observation"]["image_primary"]))
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][-1])
        pixel_values = self.image_transform(img)
        with torch.no_grad():
            initial_pixel_values = self.image_transform_lam(img)
            target_pixel_values = self.image_transform_lam(img_k)
            video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0).to(self.action_tokenizer.device)
            
            if isinstance(self.action_tokenizer, RQ_HandLatentActionModel):
                B, T = video.shape[:2]
                H, W = video.shape[3:5]
                action = torch.tensor(action[:self.window_size]).unsqueeze(0).to(self.action_tokenizer.device)
                outputs = self.action_tokenizer.vq_encode(video, action)
                video_latent_action_idx = outputs['indices_video'].squeeze()
                hand_action_idx = outputs['indices_hand'].squeeze()
                action_vocab = [f'<DYN_{i.item()}>' for i in video_latent_action_idx] + [f'<DEX_{i.item()}>' for i in hand_action_idx]
            else:
                raise TypeError("Unsupported action_tokenizer type.")

        action_tokens = '<BOA>'
        for i, action in enumerate(action_vocab):
            action_tokens += action
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        if not use_reasoning:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": action_tokens},
            ]
        else:
            reasoning = '<BOR>' + reasoning
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": reasoning},
            ]
            # print("conversation", conversation)
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
    
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        if not use_reasoning:
            labels[: -(len(action_vocab) + 2)] = IGNORE_INDEX
            # labels[: -(len(action_vocab) + 3)] = IGNORE_INDEX
        else:
            reason_ids = self.base_tokenizer(reasoning, add_special_tokens=False).input_ids
            labels[:-(len(reason_ids) + 1)] = IGNORE_INDEX
            # labels[:-(len(reason_ids) + 2)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        diff_loss_mask = not use_reasoning
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, proprio=rlds_batch["observation"]["proprio"][0:1], action=rlds_batch["action"][: self.window_size], diff_loss_mask=diff_loss_mask)


@dataclass
class RLDSBatchTransform_Alwaysresoning_CoTLatentAction:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    image_transform_lam: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    window_size: int = 5 

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        # return action (T, d) proprio (d)
        # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        latest_reasoning = rlds_batch["observation"]["latest_reasoning"][0].decode()
        use_reasoning = False
        if(rlds_batch["observation"]["reasoning"][0].decode() != ''):
            reasoning = rlds_batch["observation"]["reasoning"][0].decode()

        # print(len(rlds_batch["observation"]["image_primary"]))
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][-1])
        pixel_values = self.image_transform(img)
        with torch.no_grad():
            initial_pixel_values = self.image_transform_lam(img)
            target_pixel_values = self.image_transform_lam(img_k)
            video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0).to(self.action_tokenizer.device)
            
            if isinstance(self.action_tokenizer, RQ_HandLatentActionModel):
                B, T = video.shape[:2]
                H, W = video.shape[3:5]
                action = torch.tensor(action[:self.window_size]).unsqueeze(0).to(self.action_tokenizer.device)
                outputs = self.action_tokenizer.vq_encode(video, action)
                video_latent_action_idx = outputs['indices_video'].squeeze()
                hand_action_idx = outputs['indices_hand'].squeeze()
                action_vocab = [f'<DYN_{i.item()}>' for i in video_latent_action_idx] + [f'<DEX_{i.item()}>' for i in hand_action_idx]
            else:
                raise TypeError("Unsupported action_tokenizer type.")

        action_tokens = '<BOA>'
        for i, action in enumerate(action_vocab):
            action_tokens += action
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        # print("reasoning", rlds_batch["observation"]["episode_idx"], rlds_batch["observation"]["frame_idx"], reasoning)
        reasoning = '<BOR>' + reasoning
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": reasoning + action_tokens},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
    
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!

        reason_ids = self.base_tokenizer(reasoning+action_tokens, add_special_tokens=False).input_ids
        labels[:-(len(reason_ids) + 1)] = IGNORE_INDEX
        # labels[:-(len(reason_ids) + 2)] = IGNORE_INDEX

        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        diff_loss_mask = not use_reasoning
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, proprio=rlds_batch["observation"]["proprio"][0:1], action=rlds_batch["action"][: self.window_size], diff_loss_mask=diff_loss_mask)

@dataclass
class RLDSBatchTransform_withoutCoTLatentAction:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    image_transform_lam: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    window_size: int = 5 

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        # return action (T, d) proprio (d)
        # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        latest_reasoning = rlds_batch["observation"]["latest_reasoning"][0].decode()
        use_reasoning = False
        if(rlds_batch["observation"]["reasoning"][0].decode() != ''):
            reasoning = rlds_batch["observation"]["reasoning"][0].decode()
            use_reasoning = True

        # print(len(rlds_batch["observation"]["image_primary"]))
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][-1])
        pixel_values = self.image_transform(img)
        with torch.no_grad():
            initial_pixel_values = self.image_transform_lam(img)
            target_pixel_values = self.image_transform_lam(img_k)
            
            if isinstance(self.action_tokenizer, RQ_HandLatentActionModel):
                video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0).to(self.action_tokenizer.device)
                B, T = video.shape[:2]
                H, W = video.shape[3:5]
                action = torch.tensor(action[:self.window_size]).unsqueeze(0).to(self.action_tokenizer.device)
                outputs = self.action_tokenizer.vq_encode(video, action)
                video_latent_action_idx = outputs['indices_video'].squeeze()
                hand_action_idx = outputs['indices_hand'].squeeze()
                action_vocab = [f'<DYN_{i.item()}>' for i in video_latent_action_idx] + [f'<DEX_{i.item()}>' for i in hand_action_idx]
            elif isinstance(self.action_tokenizer, FastActionTokenizer):
                # physical-intelligence/fast (see starVLA fast_ActionHeader + QwenFast.map_fast_token_to_vlm_action)
                action_chunk = np.asarray(rlds_batch["action"][: self.window_size], dtype=np.float32)
                fast_ids = self.action_tokenizer.encode_chunk(action_chunk)
                action_vocab = [f"<robot_action_{t}>" for t in fast_ids]
                # print("action_vocab", len(action_vocab))
            elif isinstance(self.action_tokenizer, ActionTokenizer):
                future_actions = rlds_batch["action"][1:self.window_size]
                current_action = rlds_batch["action"][0]
                future_actions_string = ''.join(self.action_tokenizer(future_actions))
                # Get action chunk string
                current_action_string = self.action_tokenizer(current_action)
                action_vocab = current_action_string + future_actions_string
                action_chunk_len = len(action_vocab)
            else:
                raise TypeError("Unsupported action_tokenizer type.")

        action_tokens = '<BOA>'
        for i, action in enumerate(action_vocab):
            action_tokens += action
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_tokens},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        if isinstance(self.action_tokenizer, ActionTokenizer):
            labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        else:
            labels[: -(len(action_vocab) + 2)] = IGNORE_INDEX
        
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        diff_loss_mask = not use_reasoning
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, proprio=rlds_batch["observation"]["proprio"][0:1], action=rlds_batch["action"][: self.window_size], diff_loss_mask=diff_loss_mask)


@dataclass
class RLDSBatchTransformVideo:
    image_transform: ImageTransform

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], np.array(rlds_batch["action"])
        
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])#.copy()
        initial_pixel_values = self.image_transform(img)
        
        # the frame interval is already tackled in RLDS dataloader
        target_frame_index = -1
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][target_frame_index])#.copy()
        # print(sum(np.array(img_k) - np.array(img)))
        target_pixel_values= self.image_transform(img_k)

        return dict(initial_pixel_values=initial_pixel_values, target_pixel_values=target_pixel_values, 
                    task_instruction=lang, action=action, dataset_name=dataset_name)



@dataclass
class RLDSBatchTransformVideoall_vision:
    image_transform: ImageTransform

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], np.array(rlds_batch["action"])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        all_images = rlds_batch["observation"]["image_primary"]
        num_frames = len(all_images)
        
        if num_frames >= 2:
            selected_frames = [all_images[0], all_images[-1]]
        else:
            selected_frames = list(all_images)
        
        selected_actions = action
        
        all_pixel_values = []
        for frame in selected_frames:
            img = Image.fromarray(frame)
            pixel_values = self.image_transform(img)
            all_pixel_values.append(pixel_values)

        return dict(all_pixel_values=all_pixel_values, 
                    task_instruction=lang, action=selected_actions, dataset_name=dataset_name)

class RLDSDatasetMotionDynamics(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        window_size: int = 16,
        train: bool = True,
        image_aug: bool = False,
        training_phase: str = 'lam',
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        per_dataset_kwargs, weights = get_ego_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=window_size-1,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=8,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            training_phase=training_phase,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
    

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        window_size: int = 10,
        train: bool = True,
        image_aug: bool = False,
        training_phase: str = 'lam',
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        # per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
        #     self.data_root_dir,
        #     mixture_spec,
        #     load_camera_views=("primary",),
        #     load_depth=False,
        #     load_proprio=True,
        #     load_language=True,
        #     action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        # )
        per_dataset_kwargs, weights = get_ego_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=window_size-1,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=8,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            training_phase=training_phase,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
    

class LeDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        window_size: int = 10,
        train: bool = True,
        image_aug: bool = False,
        training_phase: str = 'lam',
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                            # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=8,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            training_phase=training_phase,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

class EpisodicRLDSDatasetMotionDynamics(RLDSDatasetMotionDynamics):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
