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
        base_model = models.mobilenet_v2(pretrained=False)
        # Replace the classifier with a new one
        base_model.classifier[1] = nn.Linear(base_model.last_channel, num_classes)
        self.model = base_model

    def forward(self, x):
        return self.model(x)

class AggregationModule(nn.Module):
    def __init__(self, num_classes, num_experts=3):
        super(AggregationModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_experts * num_classes, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, expert_outputs):
        # Concatenate the outputs from the experts
        concat_outputs = torch.cat(expert_outputs, dim=1)
        # Forward pass through the aggregation module
        aggregated_output = self.fc(concat_outputs)
        return aggregated_output

def load_expert_model(model_path, num_classes):
    # Load the expert model and load its weights
    model = ExpertModel(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()  # Set to evaluation mode
    return model

def train_aggregation_module(expert_models, aggregation_module, dataloader, num_epochs=20, learning_rate=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(aggregation_module.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    aggregation_module.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with torch.no_grad():
                # Get predictions from each expert
                expert_outputs = [model(images) for model in expert_models]
                
            # Concatenate expert outputs and pass through aggregation module
            outputs = aggregation_module(expert_outputs)
            
            # Compute loss and backpropagate
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
        
        scheduler.step()
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions.double() / len(dataloader.dataset)
        print(f'Aggregation Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return aggregation_module

def evaluate_mixture_of_experts(expert_models, aggregation_module, dataloader, report_filename="mixture_classification_report.txt"):
    expert_models = [model.eval() for model in expert_models]
    aggregation_module.eval()
    correct_predictions = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            expert_outputs = [model(images) for model in expert_models]
            outputs = aggregation_module(expert_outputs)
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
    # Data transformations and normalization
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
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Load trained expert models
    forward_model = load_expert_model('forward_expert.pth', NUM_CLASSES)
    uniform_model = load_expert_model('uniform_expert.pth', NUM_CLASSES)
    backward_model = load_expert_model('backward_expert.pth', NUM_CLASSES)
    
    # List of expert models
    expert_models = [forward_model, uniform_model, backward_model]
    
    # Create and train the aggregation module
    aggregation_module = AggregationModule(num_classes=NUM_CLASSES, num_experts=3).to(DEVICE)
    
    print("\nTraining aggregation module with mixture of experts...")
    aggregation_module = train_aggregation_module(expert_models, aggregation_module, train_loader)
    
    # Evaluate the mixture of experts
    print("\nEvaluating mixture of experts on validation data...")
    evaluate_mixture_of_experts(expert_models, aggregation_module, val_loader, report_filename="mixture_classification_report.txt")
    
    # Save the aggregation module
    save_model(aggregation_module, 'mixture_aggregation_module.pth')

if __name__ == '__main__':
    main()
