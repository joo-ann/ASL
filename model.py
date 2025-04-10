import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import cv2
from PIL import Image, ImageFilter
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 32
img_size = 200
data_dir = 'asl_alphabet_train'

class WebcamAugmentation:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, Image.Image):
                img = np.array(img)
            noise = np.random.normal(0, random.uniform(2, 10), img.shape).astype(np.uint8)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                blur_radius = random.uniform(0.3, 1.5)
                img = Image.fromarray(img)
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                img = np.array(img)
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-15, 15)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
        return img

train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
    WebcamAugmentation(p=0.7),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.02),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    WebcamAugmentation(p=0.3),
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
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
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
        identity = x
        x = self.res1(x)
        x += identity
        x = self.relu1(x)
        x = self.conv2(x)
        identity = x
        x = self.res2(x)
        x += identity
        x = self.relu2(x)
        x = self.conv3(x)
        identity = x
        x = self.res3(x)
        x += identity
        x = self.relu3(x)
        x = self.pool(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    os.makedirs('augmentation_samples', exist_ok=True)
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    num_classes = len(full_dataset.classes)
    print("Saving augmentation samples...")
    aug_indices = torch.randperm(len(full_dataset))[:10]
    for i, idx in enumerate(aug_indices):
        img, label = full_dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        cv2.imwrite(f'augmentation_samples/sample_{i}_class_{full_dataset.classes[label]}.jpg', 
                    cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = ASLModel(num_classes).to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=3, min_lr=1e-6)
    
    def train_model(model, criterion, optimizer, scheduler, num_epochs=20, patience=5):
        best_val_loss = float('inf')
        no_improve_epochs = 0
        best_model_state = None

        for epoch in range(num_epochs):
            model.train()
            running_loss, correct = 0.0, 0
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    batch_acc = (preds == labels).float().mean().item()
                    print(f"  Batch {batch_idx+1}/{len(train_loader)} - "
                          f"Loss: {loss.item():.4f}, Acc: {batch_acc:.2%}")
                    
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct / len(train_loader.dataset)
            model.eval()
            val_loss, val_correct = 0.0, 0
            val_preds, val_labels_list = [], []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels_list.extend(labels.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_acc = val_correct / len(val_loader.dataset)
            scheduler.step(val_loss)
            print(f"→ Epoch Summary: "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2%}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                best_model_state = model.state_dict().copy()
                print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
                torch.save(model.state_dict(), "asl_model_best.pth")
            else:
                no_improve_epochs += 1
                print(f"  No improvement for {no_improve_epochs} epochs.")
            if no_improve_epochs >= patience:
                print(f"Early stopping after {epoch+1} epochs!")
                model.load_state_dict(best_model_state)
                break
        return model
    
    model = train_model(model, criterion, optimizer, scheduler)
    test_dataset = datasets.ImageFolder('asl_alphabet_test', transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    class_names = test_dataset.classes
    model.eval()
    all_preds, all_labels = [], []
    test_loss, test_correct = 0.0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = test_correct / len(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Loss: {test_loss / len(test_dataset):.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.title(f"Test Accuracy: {test_accuracy:.2%}")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    fig, axs = plt.subplots(3, 5, figsize=(15, 9))
    axs = axs.flatten()
    indices = np.random.choice(range(len(test_dataset)), 15, replace=False)
    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]
        axs[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))
        axs[i].axis('off')
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            confidence, pred_idx = torch.max(probabilities, 0)
        pred = class_names[pred_idx.item()]
        true = class_names[label]
        color = 'green' if pred == true else 'red'
        axs[i].set_title(f"Pred: {pred} ({confidence:.2f})\nTrue: {true}", 
                         color=color, fontsize=10)
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    plt.close()
    torch.save(model.state_dict(), "asl_model_final.pth")
    
    print("\nTraining complete!")
    print("Best model saved as 'asl_model_best.pth'")
    print("Final model saved as 'asl_model_final.pth'")
