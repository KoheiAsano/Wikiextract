import re
from mojimoji import han_to_zen

def normalize_number(text):
    # 連続した数字を0で置換
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text

def upper_text(text):
    return text.upper()

def normalize(text):
    normalized_text = han_to_zen(text)
    normalized_text = normalize_number(normalized_text)
    normalized_text = han_to_zen(text)
    normalized_text = upper_text(normalized_text)
    return normalized_text
