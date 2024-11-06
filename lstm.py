import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from tqdm import tqdm
import mlflow

# Dataset class to handle loading .npy files
class AccentDataset(Dataset):
    def __init__(self, data_dir, label_encoder):
        self.data_dir = data_dir
        self.files = []
        self.labels = []
        
        # Collect all files and their respective labels
        for accent_folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, accent_folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.npy'):
                        self.files.append(os.path.join(folder_path, file))
                        self.labels.append(accent_folder)

        # Encode the labels
        self.label_encoder = label_encoder
        self.labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        features = np.load(file_path)  # Load the .npy feature embedding
        features = torch.tensor(features).float()  # Convert to tensor
        label = torch.tensor(self.labels[idx]).long()  # Get label
        return features, label

# LSTM Model for Accent Classification
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # LSTM expects input of shape (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)  # out has shape (batch_size, seq_len, hidden_dim)
        out = out[:, -1, :]    # Taking output of the last time step
        out = self.fc(out)      # Final layer for classification
        return out

# Function to compute class weights
def compute_class_weights(labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

# Load dataset and split into train/test
def load_data_and_split(data_dir, label_encoder, batch_size=16, test_split=0.2):
    dataset = AccentDataset(data_dir, label_encoder)
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, dataset.labels

# Training function with early stopping
def train_model(model, train_loader, criterion, optimizer, device, patience=5):
    model.train()
    train_losses = []
    early_stop_count = 0
    min_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0
        all_labels = []
        all_preds = []
        
        for features, labels in tqdm(train_loader):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Early stopping check
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), "lstm_model.pth")
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= patience:
            print("Early stopping triggered")
            break

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation function with confusion matrix and metrics
def evaluate_model(model, test_loader, criterion, device, label_encoder):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")

    # Classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# Main script
if __name__ == "__main__":
    data_dir = 'Tinyembeddings'  # Path to the main folder with accents
    num_classes = 12  # Number of accent classes
    batch_size = 16
    num_epochs = 20
    patience = 5  # Early stopping patience

    # Label encoding for accents
    from sklearn.preprocessing import LabelEncoder
    accents = sorted(os.listdir(data_dir))
    label_encoder = LabelEncoder()
    label_encoder.fit(accents)

    # Load train/test data
    train_loader, test_loader, labels = load_data_and_split(data_dir, label_encoder, batch_size=batch_size)

    # Compute class weights for handling imbalance
    class_weights = compute_class_weights(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)

    # Define model, loss function, optimizer
    input_dim = 64
    hidden_dim = 128
    num_layers = 2
    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Track training with MLflow
    mlflow.start_run()
    mlflow.log_param("model", "LSTM")
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("num_layers", num_layers)

    # Train the model with early stopping
    train_model(model, train_loader, criterion, optimizer, device, patience)

    # Load the best model for evaluation
    model.load_state_dict(torch.load("lstm_model.pth"))

    # Evaluate the model on test set
    evaluate_model(model, test_loader, criterion, device, label_encoder)

    # End MLflow tracking
    mlflow.end_run()
