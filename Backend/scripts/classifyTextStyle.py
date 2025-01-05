import re

def classify_text_style(text):
    """Classify text style based on a regex pattern."""
    if re.match(r'^[A-Z0-9]{2,3}-[A-Z0-9]{1,2}-[0-9]{1,4}$', text):
        return "bold black"
    else:
        return "other"
