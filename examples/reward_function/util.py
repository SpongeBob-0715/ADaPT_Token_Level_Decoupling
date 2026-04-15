import re
from mathruler.grader import extract_boxed_content, grade_answer

# Define allowed characters
allowed_base = set(
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    " \n\t"
    ".,!?;:'\"()[]{}<>/\\-_+=*&%$#@~^|`√°²Δ×£π"
)

def contains_cjk_or_garbled(text: str) -> bool:
    """
    Checks if the text contains CJK (Chinese, Japanese, Korean) characters
    or specific garbled characters.
    """
    for ch in text:
        code_point = ord(ch)
        # CJK character ranges
        if (
            0x4E00 <= code_point <= 0x9FFF or     # Chinese characters
            0x3040 <= code_point <= 0x309F or     # Japanese Hiragana
            0x30A0 <= code_point <= 0x30FF or     # Japanese Katakana
            0xAC00 <= code_point <= 0xD7AF or     # Korean Hangul Syllables
            0x1100 <= code_point <= 0x11FF         # Korean Hangul Jamo
        ):
            return True
        
        # Check for common garbled substitution character
        if ch == '':  
            return True
    return False

def has_illegal_char_rule(text: str) -> bool:
    """
    Checks for streaks of characters not in the allowed_base set.
    Returns True if a streak of 3 or more illegal characters is found.
    """
    illegal_streak = 0
    for ch in text:
        if ch not in allowed_base:
            illegal_streak += 1
            if illegal_streak >= 3:
                return True
        else:
            illegal_streak = 0
    return False

def format_reward(response: str, response_length: int) -> float:
    """
    Checks if the response format is correct and calculates a format score.
    Returns:
        float: The format score (1.0 for perfect format, lower for penalties).
    """
    
    # 0. Check for illegal characters or garbled text
    # If detected, return 0.0 immediately
    if has_illegal_char_rule(response) or contains_cjk_or_garbled(response):
        return 0.0

    # 1. Check tag occurrence: each tag must appear <= 1 time
    if sum(1 for _ in re.finditer(r"<think>", response)) > 1 or sum(1 for _ in re.finditer(r"<answer>", response)) > 1: 
        return 0.1

    # 2. Define two legal formats
    
    # (A) Starts with <think>, followed by any content
    pattern_with_think = re.compile(
        r"^<think>.*$", 
        re.DOTALL
    )

    # (B) Starts with <answer> only, followed by any content
    pattern_answer_only = re.compile(
        r"^<answer>.*$", 
        re.DOTALL
    )

    if re.fullmatch(pattern_with_think, response):
        return 1.0
        
    elif re.fullmatch(pattern_answer_only, response):
        # Apply length penalty for answer-only responses
        length = response_length
        base_length = 400
        max_deduction = 0.5
        deduction_per_100 = 0.1
        
        if length > base_length:
            excess_length = length - base_length
            # Calculate how many 100-char blocks act as excess
            hundreds_over = excess_length // 100
            # Ensure deduction does not exceed max_deduction
            deduction = min(hundreds_over * deduction_per_100, max_deduction)
            score = 1.0 - deduction
        else:
            score = 1.0
            
        return round(score, 2)  # Keep two decimal places
    
    # Format does not match either pattern
    return 0.1

def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Extracts the answer from the response and compares it with the ground truth.
    """
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0