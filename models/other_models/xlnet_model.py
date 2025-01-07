from transformers import XLNetTokenizer, XLNetForSequenceClassification

class XLNetModel:
    """
    A simple wrapper for XLNet-based sequence classification.
    """
    def __init__(self, model_name="xlnet-base-cased"):
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.model = XLNetForSequenceClassification.from_pretrained(model_name)

    def predict(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.logits.argmax(dim=-1)
