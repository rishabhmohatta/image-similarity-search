import torch
import torch.nn as nn
# from transformers import ViTModel, ViTConfig
from transformers import ViTModel, ViTFeatureExtractor

class ViT():
    def __init__(self):
        
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.VIT_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.VIT_model.eval()

    def extract_feature(self,dataset):
        embeddings = []
        with torch.no_grad():
            for image, _ in dataset:
                inputs = self.feature_extractor(images=image, return_tensors="pt")
                outputs = self.VIT_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()  # Get the [CLS] token
                embeddings.append(embedding)
        return np.vstack(embeddings)