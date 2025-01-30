import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from dataset import CustomImageDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 6

class ExpertModel(nn.Module):
    def __init__(self, num_classes):
        super(ExpertModel, self).__init__()
        # Use MobileNetV2 as the base model
        base_model = models.mobilenet_v2(pretrained=True)
        # Replace the classifier with a new one
        base_model.classifier[1] = nn.Linear(base_model.last_channel, num_classes)
        self.model = base_model

    def forward(self, x):
        return self.model(x)

def train_model(model, dataloader, criterion, num_epochs=20, learning_rate=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
        
        scheduler.step()
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions.double() / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return model

def evaluate(model, dataloader, report_filename="forward_classification_report.txt"):
    model.eval()
    correct_predictions = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    accuracy = correct_predictions.double() / total
    report = classification_report(all_labels, all_preds, digits=4)
    print(f'Validation Accuracy: {accuracy:.4f}')
    print(report)
    
    with open(report_filename, "w") as f:
        f.write(f'Validation Accuracy: {accuracy:.4f}\n\n')
        f.write(report)
    
    return accuracy

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def main():
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),

    ])
    
    # Load datasets
    train_dataset = CustomImageDataset(
        images_folder='../OPMD_new/output/train/images',
        labels_folder='../OPMD_new/output/train/labels',
        transform=transform
    )
    
    val_dataset = CustomImageDataset(
        images_folder='../OPMD_new/output/val/images',
        labels_folder='../OPMD_new/output/val/labels',
        transform=transform
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Model, criterion, optimizer
    model = ExpertModel(num_classes=NUM_CLASSES).to(DEVICE)
    # Compute class weights to handle imbalance
    train_labels = [sample['class_id'] for sample in train_dataset.samples]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = class_counts.sum() / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Train the model
    print("Training Forward Expert...")
    model = train_model(model, train_loader, criterion)
    
    # Evaluate the model
    print("\nEvaluating on validation data...")
    evaluate(model, val_loader, report_filename="forward_classification_report.txt")
    
    # Save the trained model
    save_model(model, 'forward_expert.pth')

if __name__ == '__main__':
    main()
