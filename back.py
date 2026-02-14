import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import joblib

# Load artifacts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_pickle("recipes.pkl")
encoder = joblib.load("label_encoder.pkl")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(encoder.classes_))
model.load_state_dict(torch.load("food_model.pth", map_location=device))
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

    # Collect all recipes for this dish
    matches = df[df["name"] == dish_name]
    recipes = []
    for _, row in matches.iterrows():
        recipes.append({
            "dish": dish_name,
            "ingredients": row.get("ingredients", ""),
            "instructions": row.get("instructions", "")
        })
    return recipes