import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import mlflow
import matplotlib.pyplot as plt
import os
from sklearn.utils.class_weight import compute_class_weight

# Dataset class
class AccentDataset(Dataset):
    def __init__(self, data_dir, label_encoder):
        self.files = []
        self.labels = []
        for accent_folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, accent_folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.npy'):
                        self.files.append(os.path.join(folder_path, file))
                        self.labels.append(accent_folder)
        
        # Encode labels
        self.label_encoder = label_encoder
        self.labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        features = np.load(file_path)  # Shape (230, 64)
        features = torch.tensor(features, dtype=torch.float32)  # Convert to tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Get label
        return features, label

# RNN Model for Classification
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        _, h_n = self.rnn(x)  # Use hidden state from the last layer
        out = self.fc(h_n[-1])  # Apply fully connected layer to the last hidden state
        return out

# Function to compute class weights
def compute_class_weights(labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

# Training and evaluation functions
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []
    for features, labels in tqdm(train_loader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / len(train_loader)
    report = classification_report(all_labels, all_preds, output_dict=True)
    return avg_loss, report

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Evaluating"):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    report = classification_report(all_labels, all_preds, output_dict=True)
    return avg_loss, report

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
    class_weights = compute_class_weights(dataset.labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights=class_weights.to(device)

    # Model, Loss, Optimizer, Early Stopping
    model = RNNClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes)
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
            torch.save(model.state_dict(), "best_rnn_model.pth")
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
