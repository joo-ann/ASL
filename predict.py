import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageFilter
import random
import os
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 64  
img_size = 200
data_dir = 'asl_alphabet_train'

class WebcamAugmentation:
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, Image.Image):
                img = np.array(img)
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            if random.random() < 0.2:
                blur_radius = random.uniform(0.3, 1.0)
                img = Image.fromarray(img)
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                img = np.array(img)
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-10, 10)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
        return img

train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    WebcamAugmentation(p=0.4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

class ASLModel(nn.Module):
    def __init__(self, num_classes):
        super(ASLModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256)
        )
        self.relu3 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x += self.res1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x += self.res2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x += self.res3(x)
        x = self.relu3(x)
        x = self.pool(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    num_classes = len(full_dataset.classes)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ASLModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    def train_model(model, criterion, optimizer, num_epochs=15):
        best_acc = 0.0
        best_model_wts = None
        for epoch in range(num_epochs):
            model.train()
            total_loss, correct = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

            epoch_loss = total_loss / len(train_loader.dataset)
            epoch_acc = correct / len(train_loader.dataset)
            print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2%}")

            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item() * images.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            val_loss /= len(val_loader.dataset)
            val_acc = val_correct / len(val_loader.dataset)
            print(f"→ Validation: Loss = {val_loss:.4f}, Accuracy = {val_acc:.2%}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = model.state_dict()

        if best_model_wts:
            torch.save(best_model_wts, "asl_model_best.pth")
            print("Best model saved as 'asl_model_best.pth'")
        return model

    model = train_model(model, criterion, optimizer)

    test_dataset = datasets.ImageFolder('asl_alphabet_test', transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    class_names = test_dataset.classes

    model.load_state_dict(torch.load("asl_model_best.pth"))
    model.eval()
    all_preds, all_labels = [], []
    test_loss, test_correct = 0.0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item() * images.size(0)
            preds = outputs.argmax(1)
            test_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = test_correct / len(test_dataset)
    print(f"Test Accuracy: {acc:.2%}")
    print(f"Test Loss: {test_loss / len(test_dataset):.4f}")

    plt.figure(figsize=(20, 16))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    torch.save(model.state_dict(), "asl_model_final.pth")
    print("Final model saved as 'asl_model_final.pth'")
