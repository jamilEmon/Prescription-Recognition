import torch
from torchvision import transforms
from PIL import Image
import json
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "./model"

# Load model
from torchvision import models
num_classes = 78  # Update this according to your dataset

# Initialize ResNet18 and replace fc layer
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
ckpt_path = os.path.join(MODEL_DIR, "best_resnet18.pt")
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE)
model.eval()

# Load label maps
with open(os.path.join(MODEL_DIR, "label_map.json"), "r", encoding="utf-8") as f:
    label_map = json.load(f)
with open(os.path.join(MODEL_DIR, "med2gen_map.json"), "r", encoding="utf-8") as f:
    med2gen_map = json.load(f)

# Preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_idx = model(img_t).argmax(1).item()
    medicine = label_map[str(pred_idx)]
    generic = med2gen_map.get(medicine, "<NA>")
    return {"medicine": medicine, "generic": generic}
