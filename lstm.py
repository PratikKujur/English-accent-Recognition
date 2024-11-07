import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import mlflow
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import Counter

# Dataset setup
class AccentDataset(Dataset):
    def __init__(self, data_dir, label_encoder):
        self.data_dir = data_dir
        self.label_encoder = label_encoder
        self.files = []
        self.labels = []
        for label_folder in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_folder)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith(".npy"):
                        self.files.append(os.path.join(label_path, file))
                        self.labels.append(label_folder)

        self.labels = label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features = np.load(self.files[idx])
        label = self.labels[idx]
        return torch.tensor(features, dtype=torch.float32), label

# Model setup
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Training, evaluation, and utilities
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    all_labels, all_preds = [], []
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device).long()  # Convert labels to Long tensor
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    return avg_loss, report

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device).long()  # Convert labels to Long tensor
            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    return avg_loss, report


def compute_class_weights(labels):
    label_counts = Counter(labels)
    total_count = sum(label_counts.values())
    class_weights = [total_count / label_counts[i] for i in range(len(label_counts))]
    return torch.tensor(class_weights, dtype=torch.float32)

# Main training script
def main():
    data_dir = 'Tinyembeddings'
    num_classes = 12  # Adjust this based on your dataset
    batch_size = 16
    num_epochs = 10
    patience = 5  # For early stopping
    hidden_dim = 128
    num_layers = 2
    input_dim = 64  # Each feature vector length
    label_encoder = LabelEncoder()
    label_encoder.fit(os.listdir(data_dir))

    # Dataset and DataLoader setup
    dataset = AccentDataset(data_dir, label_encoder)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Compute class weights and set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_class_weights(dataset.labels).to(device)

    # Model, Loss, Optimizer, Early Stopping
    model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stop_counter = 0
    best_val_loss = float("inf")
    
    # Metrics for loss comparison
    train_losses, val_losses = [], []

    # Logging with MLflow
    mlflow.start_run()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_report = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_report = evaluate(model, val_loader, criterion, device)
        
        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_report["accuracy"],
            "val_accuracy": val_report["accuracy"]
        }, step=epoch)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_lstm_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

        # Track loss for comparison
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print("Train Classification Report:", train_report)
        print("Validation Classification Report:", val_report)

    mlflow.end_run()

    # Confusion matrix on validation set
    all_labels, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    # Plot train vs validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
