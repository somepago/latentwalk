import torch
import torch.nn as nn

class ModelWithIntermediateLayers(nn.Module):
    def __init__(self):
        super().__init__()
        dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.model = dinov2_vits14_reg
        self.model.eval()

    def forward(self, images):
        with torch.inference_mode():
            features = self.model.get_intermediate_layers(images, 1, return_class_token=True)
            # First items of tuple contains features from last layer and linear layer
            # Return the features from the last layer
            features = features[0][0]  
        return features


if __name__ == "__main__":
    model = ModelWithIntermediateLayers()
    images = torch.randn(1, 3, 224, 224)  # Example input
    features = model(images)
    print(features.shape)