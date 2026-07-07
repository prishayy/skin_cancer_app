
import cv2
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_segmentation_model(model_path, encoder_name="resnet18"):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )

    state_dict = torch.load(model_path, map_location=DEVICE)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v

    model.load_state_dict(cleaned, strict=True)
    model.to(DEVICE)
    model.eval()

    return model

def segment_image(pil_image, model, image_size=256, threshold=0.5):
    original_rgb = np.array(pil_image.convert("RGB"))
    h, w = original_rgb.shape[:2]

    resized = cv2.resize(original_rgb, (image_size, image_size))
    img = resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))

    tensor = torch.tensor(img).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()

    mask_small = (prob > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    binary_mask = (mask > 127).astype(np.uint8)

    segmented = original_rgb.copy()
    segmented[binary_mask == 0] = 0

    mask_rgb = np.stack([mask, mask, mask], axis=-1)

    segmented_image = Image.fromarray(segmented)
    mask_image = Image.fromarray(mask_rgb)

    return segmented_image, mask_image
