import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from models import load_pretrained_googlenet
from dataloader import get_dataloaders

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
num_diseases = 584

# Load the pre-trained model
model = load_pretrained_googlenet(num_classes=num_diseases).to(device)
model.load_state_dict(torch.load('path_to_weights.pth'))  # Update with actual path to the model weights
model.eval()

# Transform for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Get test dataloader
_, test_dataloader = get_dataloaders('/img')

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return accuracy, recall, precision, f1

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix by row (i.e., by the number of samples)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    # Create a large figure
    plt.figure(figsize=(30, 30))  # Increased figure size for better readability
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=classes, yticklabels=classes, cbar=True)

    plt.title(title or 'Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=90)  # Rotate x labels for better visibility
    plt.yticks(rotation=0)   # Rotate y labels for better visibility
    plt.tight_layout()

    # Save the confusion matrix as a high-resolution image
    plt.savefig(f'confusion_matrix_{"normalized" if normalize else "frequency"}.png', dpi=300)  # Save with high DPI
    plt.show()

# Perform inference and evaluate metrics
def infer_and_evaluate(dataloader, model):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            pred = output.argmax(dim=1, keepdim=True).squeeze()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Calculate metrics
    accuracy, recall, precision, f1 = calculate_metrics(all_labels, all_preds)

    # Calculate top-k accuracy
    all_preds = np.array([output.cpu().numpy() for X, y in dataloader for output in model(X.to(device))])
    all_labels = np.array([y.cpu().numpy() for _, y in dataloader])

    top1_accuracy = np.mean(np.any(all_preds.argsort(axis=1)[:, -1:] == all_labels[:, None], axis=1))
    top5_accuracy = np.mean(np.any(all_preds.argsort(axis=1)[:, -5:] == all_labels[:, None], axis=1))
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")

    # Plot confusion matrices
    plot_confusion_matrix(all_labels, all_preds, classes=range(num_diseases), normalize=False, title='Confusion Matrix (Frequency)')
    plot_confusion_matrix(all_labels, all_preds, classes=range(num_diseases), normalize=True, title='Confusion Matrix (Normalized)')

# Run inference and evaluation
infer_and_evaluate(test_dataloader, model)
