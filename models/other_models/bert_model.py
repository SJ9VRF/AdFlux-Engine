from transformers import BertTokenizer, BertForSequenceClassification

class BERTModel:
    """
    A simple wrapper for BERT-based sequence classification.
    """
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

    def predict(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.logits.argmax(dim=-1)
