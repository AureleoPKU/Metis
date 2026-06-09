"""
Fast action tokenizer — wraps Hugging Face `physical-intelligence/fast` (AutoProcessor).

Aligned with starVLA `model/modules/action_model/fast_ActionHeader.py`:
  https://huggingface.co/physical-intelligence/fast

Use `encode_chunk` on an (T, D) action window to obtain FAST token ids, then pair with
`map_to_prompt_fragment` / `<robot_action_{id}>` strings for the VLM prompt (same as
`QwenFast.map_fast_token_to_vlm_action`).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from transformers import AutoProcessor


class FastActionTokenizer:
    """Thin wrapper around `AutoProcessor.from_pretrained(..., trust_remote_code=True)`."""

    def __init__(
        self,
        pretrained_name_or_path: str = "physical-intelligence/fast",
        time_horizon: Optional[int] = None,
        action_dim: Optional[int] = None,
    ) -> None:
        self.fast_processor = AutoProcessor.from_pretrained(pretrained_name_or_path, trust_remote_code=True)
        if time_horizon is not None:
            self.fast_processor.time_horizon = time_horizon
        if action_dim is not None:
            self.fast_processor.action_dim = action_dim

    def set_horizon_and_dim(self, time_horizon: int, action_dim: int) -> None:
        """Match starVLA `QwenFast`: set `time_horizon` and `action_dim` on the underlying processor."""
        self.fast_processor.time_horizon = time_horizon
        self.fast_processor.action_dim = action_dim

    def encode_chunk(self, action_chunk: np.ndarray) -> List[int]:
        """
        Encode one trajectory chunk (T, D) to FAST discrete token ids.

        Mirrors `Fast_Action_Tokenizer.encoder_action2fastoken` with a batch of size 1:
        `np.stack([action_chunk], axis=0)` -> `fast_processor(batch)`.
        """
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"action_chunk must be (T, D), got shape {action_chunk.shape}")
        batch_actions = np.stack([action_chunk], axis=0)
        out = self.fast_processor(batch_actions)
        return self._normalize_processor_output(out)

    def _normalize_processor_output(self, out: object) -> List[int]:
        """Accept list/tuple, BatchEncoding-like dict, or raw id list."""
        if isinstance(out, (list, tuple)):
            first = out[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                return [int(x) for x in first]
            if isinstance(first, int):
                return [int(x) for x in out]
            return [int(x) for x in first]
        if isinstance(out, dict):
            for key in ("input_ids", "token_ids", "fast_token_ids", "ids"):
                if key not in out:
                    continue
                v = out[key]
                arr = np.asarray(v)
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr[0]
                return [int(x) for x in arr.flatten().tolist()]
        raise TypeError(f"Unexpected FAST processor output: {type(out)!r}")

    @staticmethod
    def map_fast_token_to_vlm_action_string(fast_token_ids: List[int]) -> str:
        """Same string pattern as starVLA `QwenFast.map_fast_token_to_vlm_action`."""
        return "".join(f"<robot_action_{token}>" for token in fast_token_ids)

    def decode(self, fast_token_ids: List[int]) -> np.ndarray:
        """Decode FAST token ids to continuous actions (pi-fast API)."""
        return np.asarray(self.fast_processor.decode(fast_token_ids))
