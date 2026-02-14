import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import joblib
import os
import requests

# Artifacts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_pickle("recipes.pkl")
encoder = joblib.load("label_encoder.pkl")

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# Model file handling
MODEL_PATH = "food_model.pth"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1aaAQXW0sQRWrrhEtY-EZkPzQIAMjlwId"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(encoder.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

def predict_dish(file_obj_or_path):
    # Handle file input
    if hasattr(file_obj_or_path, "read"):  # file-like object
        file_obj_or_path.seek(0)
        img = Image.open(file_obj_or_path).convert("RGB")
    else:  # path string
        img = Image.open(file_obj_or_path).convert("RGB")

    # Transform + predict
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    dish_name = encoder.inverse_transform([predicted.item()])[0]

    # Collect recipes
    matches = df[df["name"] == dish_name]
    recipes = []
    for _, row in matches.iterrows():
        recipes.append({
            "dish": dish_name,
            "ingredients": row.get("ingredients", ""),
            "instructions": row.get("instructions", "")
        })
    return recipes