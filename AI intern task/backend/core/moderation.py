BANNED_WORDS = {"badword1", "badword2"}

def is_clean(text: str) -> bool:
    return all(word not in text.lower() for word in BANNED_WORDS)
