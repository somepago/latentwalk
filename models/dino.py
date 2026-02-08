import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ModelWithIntermediateLayers(nn.Module):
    def __init__(self):
        super().__init__()
        dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.model = dinov2_vits14_reg
        self.model.eval()

    def forward(self, images):
        """
        Args:
            images: [B, C, H, W] in [-1, 1] range. H, W must be divisible by 14.
        """
        # [-1, 1] -> [0, 1] -> ImageNet normalized
        images = (images + 1) / 2
        images = TF.normalize(images, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        with torch.inference_mode():
            features = self.model.get_intermediate_layers(images, 1, return_class_token=True)
            features = features[0][0]
        return features


if __name__ == "__main__":
    model = ModelWithIntermediateLayers()
    images = torch.randn(10, 3, 224, 224)  # Example input
    features = model(images)
    print(features.shape) # torch.Size([10, 256, 384])

