import torch
from transformers import CLIPProcessor, CLIPModel

class MultimodalModel:
    """
    A multimodal model integrating CLIP for text and image understanding.
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

    def predict(self, text, image):
        """
        Perform inference on text and image inputs.
        """
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_text = outputs.logits_per_text  # Similarity score for text->image
        logits_per_image = outputs.logits_per_image  # Similarity score for image->text
        return logits_per_text, logits_per_image

    def find_best_match(self, text_candidates, image_candidates):
        """
        Find the best text-image match from candidate sets.
        """
        best_match = None
        highest_score = float("-inf")
        for text in text_candidates:
            for image in image_candidates:
                logits_text, logits_image = self.predict(text, image)
                score = logits_text.item() + logits_image.item()
                if score > highest_score:
                    highest_score = score
                    best_match = (text, image)
        return best_match, highest_score

