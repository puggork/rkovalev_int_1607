import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("s-nlp/xlmr_formality_classifier")
model = AutoModelForSequenceClassification.from_pretrained("s-nlp/xlmr_formality_classifier")

id2formality = {0: "formal", 1: "informal"}

def get_formality_xlmr(txt):
    """
    txt: str
        A sentence to evaluate
    """
    encoding = tokenizer(
        txt,
        add_special_tokens=True,
        return_token_type_ids=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    output = model(**encoding)

    formality_scores = [
        {id2formality[idx]: score for idx, score in enumerate(text_scores.tolist())}
        for text_scores in output.logits.softmax(dim=1)
    ]
    
    return 1 if formality_scores[0]["formal"] > 0.5 else 0