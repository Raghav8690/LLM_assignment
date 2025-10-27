import re

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def estimate_pages_from_text(text: str) -> int:
    # Rough estimate: 300 words per page
    words = len(text.split())
    return max(1, words // 300)