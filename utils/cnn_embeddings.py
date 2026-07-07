
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embedding_model():
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.classifier = nn.Identity()
    model.to(DEVICE)
    model.eval()
    transform = weights.transforms()
    return model, transform

def extract_efficientnet_embedding(pil_image, model, transform):
    image = pil_image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            embedding = model(tensor)

    embedding = embedding.squeeze().detach().cpu().numpy()

    return {f"effb0_{i}": float(v) for i, v in enumerate(embedding)}
