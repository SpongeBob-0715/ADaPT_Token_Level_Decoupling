# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     XXXX
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...

class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics




class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """
            1. Insert uid field (if exists)
            2. Avoid <think> and <answer> being skipped as special tokens
        """
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)

        has_uid = "uid" in data.non_tensor_batch  # ✅ Check if uid field exists

        # ✅ Define special tokens to keep
        # allowed_special_tokens = {"<think>", "<answer>","</answer>","</think>"}
        allowed_special_tokens = {"<think>", "<answer>"}
        # ✅ Get all special tokens from tokenizer
        all_special_tokens = set(self.tokenizer.all_special_tokens)
        # ✅ Special tokens to remove = all special - allowed ones
        remove_special_tokens = all_special_tokens - allowed_special_tokens


        # ✅ New: Record if each response contains <think>
        contains_think_tags = []

        prompt_ids = data.batch["prompts"]

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]

            # Decode response
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            
            # Decode corresponding prompt
            prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=False)

            # Check if it contains <think>
            contains_think = "<think>" in response_str
            contains_think_tags.append(contains_think)

            # Remove disallowed special tokens
            for tok in remove_special_tokens:
                response_str = response_str.replace(tok, "").strip()
                prompt_str = prompt_str.replace(tok, "").strip()


            input_item = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }

            if has_uid:
                input_item["index"] = data.non_tensor_batch["uid"][i]

            reward_inputs.append(input_item)



        # ✅ Call reward function
        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)

        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            
            # ✅ Modified: Only set R_think reward when response contains <think> and uid exists
            if has_uid and contains_think_tags[i]:
                reward_tensor[i, 0] = score["R_think"]
            
            for key, value in score.items():
                reward_metrics[key].append(value)
        
        return reward_tensor, reward_metrics