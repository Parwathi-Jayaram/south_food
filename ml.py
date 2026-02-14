import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load CSV
# -----------------------------
df = pd.read_csv(r"C:/Users/parwa/OneDrive/Desktop/archive/cuisines.csv")

# -----------------------------
# 2. Map CSV rows to local images
# -----------------------------
image_folder = r"C:/Users/parwa/OneDrive/Desktop/archive/image_for _cuisines/data"
files = os.listdir(image_folder)

def find_image(dish_name, files):
    dish_name_clean = dish_name.replace(" ", "_").lower()
    for f in files:
        if dish_name_clean.split("_")[0] in f.lower():
            return os.path.join(image_folder, f)
    return None

df["image_path"] = df["name"].apply(lambda x: find_image(x, files))
df = df.dropna(subset=["image_path"])

# -----------------------------
# 3. Encode labels
# -----------------------------
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["name"])

# -----------------------------
# 4. Dataset class
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

class FoodDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["label"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# -----------------------------
# 5. Train/Test split
# -----------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = FoodDataset(train_df, transform=transform)
test_dataset = FoodDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# -----------------------------
# 6. Model setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(encoder.classes_))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 7. Training loop
# -----------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# -----------------------------
# 8. Evaluation
# -----------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# -----------------------------
# 9. Save model + encoder + dataframe
# -----------------------------
torch.save(model.state_dict(), "food_model.pth")
df.to_pickle("recipes.pkl")
import joblib
joblib.dump(encoder, "label_encoder.pkl")