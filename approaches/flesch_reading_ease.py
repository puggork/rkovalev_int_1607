import textstat
from textstat import flesch_reading_ease


def get_flesch_reading_ease(txt):
    """
    txt: str
        A sentence to evaluate
    """
    flesch_score = flesch_reading_ease(txt)
    if flesch_score < 0:
        final_score = 3.0
    else:
        final_score = 3.0 - (flesch_score / 121.22 * 6)
    return final_score
