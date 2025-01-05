import re

def clean_text(text):
    """Remove unwanted characters and convert text to uppercase."""
    return re.sub(r'[^A-Za-z0-9]', '', text).upper()
