###################################################################################### Train Model

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

image_dir = '370/fit_images'

if not os.path.exists(image_dir):
    print(f"Directory {image_dir} does not exist.")
else:
    print(f"Contents of {image_dir}: {os.listdir(image_dir)}")

label_mapping = {'bad_fits': 0, 'good_fits': 1}

image_paths = []
labels = []

for folder_name, label in label_mapping.items():
    folder_path = os.path.join(image_dir, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_name} does not exist. Skipping...")
        continue

    for image_name in os.listdir(folder_path):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_name)
            image_paths.append(image_path)
            labels.append(label)
            print(f"File: {image_path}, Label: {label}")
        else:
            print(f"Skipping non-image file: {image_name}")

class GenderDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

train_dataset = GenderDataset(train_paths, train_labels, transform=transform)
val_dataset = GenderDataset(val_paths, val_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

model.eval()
val_correct = 0
val_total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

val_acc = 100 * val_correct / val_total
print(f"Validation Accuracy: {val_acc:.2f}%")

torch.save(model.state_dict(), 'fit_classifier.pt')
print("Model saved to 'fit_classifier.pt'")

###################################################################################### Test Confidence
# import torch
# from torchvision import transforms, models
# from PIL import Image

# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = models.resnet18(weights=None)
# model.fc = torch.nn.Linear(model.fc.in_features, 2) 
# model.load_state_dict(torch.load('fit_classifier.pt', weights_only=True))
# model.to(device)
# model.eval() 

# import torch.nn.functional as F

# def predict_gender(image_path):
#     image = Image.open(image_path).convert("RGB")  
    
#     image = transform(image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(image)
#         probabilities = F.softmax(outputs, dim=1)
#         confidence, predicted = torch.max(probabilities, 1)

#     gender = "Bad" if predicted.item() == 0 else "Good"
#     confidence_percentage = confidence.item() * 100

#     return gender, confidence_percentage

# image_path = "370/test10.jpg"
# predicted_gender, confidence = predict_gender(image_path)
# print(f"Predicted Fit: {predicted_gender} with Confidence: {confidence:.2f}%")

