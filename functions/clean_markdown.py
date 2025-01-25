import re

def clean_markdown(text):
    """Remove markdown formatting characters from text"""

    patterns = [
        (r'\*\*(.*?)\*\*', r'\1'),  # Remove bold
        (r'#{1,6}\s*', ''),         # Remove headers
        (r'\[(.*?)\]\(.*?\)', r'\1'), # Remove links
        (r'[*_]{1,2}(.*?)[*_]{1,2}', r'\1'), # Remove emphasis
        (r'`{1,3}.*?`{1,3}', ''),   # Remove code blocks
        (r'\n{3,}', '\n\n')         # Clean multiple newlines
    ]
    
    cleaned = text
    for pattern, replacement in patterns:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned.strip()