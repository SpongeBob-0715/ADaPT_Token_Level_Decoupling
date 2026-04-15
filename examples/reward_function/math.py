# math.py
import re
from typing import Any
from collections import defaultdict

# Import helper functions from util module
# Ensure util.py is in the same directory or python path
try:
    from util import format_reward, accuracy_reward
except ImportError:
    # Fallback for local testing if needed, or adjust import based on project structure
    from .util import format_reward, accuracy_reward

def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    has_index = "index" in reward_inputs[0]

    # Step 1: Calculate base scores (ungrouped)
    base_scores = []
    for reward_input in reward_inputs:
        # Handle format variations (e.g., qwen2.5vl-32b) by cleaning whitespace around tags
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        response_length = reward_input["response_length"]

        format_score = format_reward(response, response_length)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        if format_score == 0.0:
            accuracy_score_2 = 0.0
        else:
            accuracy_score_2 = accuracy_score * format_score
            
        base_scores.append(
            {
                "overall": accuracy_score_2 + 0.8 * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    # Step 2: If 'index' field exists, group and calculate fast/slow thinking accuracy
    if has_index:
        grouped = defaultdict(list)
        for i, reward_input in enumerate(reward_inputs):
            grouped[reward_input["index"]].append((i, reward_input, base_scores[i]))

        # Initialize results
        scores = [None] * len(reward_inputs)

        # Step 3: Calculate accuracy stats within groups
        for idx, items in grouped.items():
            fast_accs, slow_accs = [], []

            for _, reward_input, base_score in items:
                response = reward_input["response"]
                if "<think>" in response:
                    slow_accs.append(base_score["accuracy"])
                else:
                    fast_accs.append(base_score["accuracy"])

            fast_acc = sum(fast_accs) / len(fast_accs) if fast_accs else 0.0
            slow_acc = sum(slow_accs) / len(slow_accs) if slow_accs else 0.0

            # Calculate R_think reward
            R_think = 0.5 * (fast_acc - 0.5) + 0.5 * (fast_acc - slow_acc)
            
            # Step 4: Write results back in original order
            for i, reward_input, base_score in items:
                response = reward_input["response"]
                is_think = "<think>" in response
                scores[i] = {
                    **base_score,
                    "R_think": (0 - R_think) if is_think else 0.0
                }

    else:
        # No index provided, return base scores directly
        scores = base_scores

    return scores