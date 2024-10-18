import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import mlflow
import mlflow.pytorch

# Custom Dataset to load .npy files
class AccentDataset(Dataset):
    def __init__(self, data_dir, label_encoder):
        self.data_dir = data_dir
        self.files = []
        self.labels = []

        for accent_folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, accent_folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.npy'):
                        self.files.append(os.path.join(folder_path, file))
                        self.labels.append(accent_folder)

        self.label_encoder = label_encoder
        self.labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        features = np.load(file_path)
        features = torch.tensor(features).float()
        label = torch.tensor(self.labels[idx]).long()
        return features, label

# Updated Simple ANN Model
class SimpleANN(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes):
        super(SimpleANN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * (seq_len // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to compute class weights
def compute_class_weights(labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Load datasets and split into train/test
def load_data_and_split(data_dir, label_encoder, batch_size=16, test_split=0.2):
    dataset = AccentDataset(data_dir, label_encoder)
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset.labels

# Calculate number of trainable parameters
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training loop with logging and MLflow
def train_model(model, train_loader, criterion, optimizer, device, early_stopping, logger):
    model.train()
    total_loss = 0
    for features, labels in tqdm(train_loader):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Training Loss: {avg_loss:.4f}")
    mlflow.log_metric('train_loss', avg_loss)
    return avg_loss

# Evaluation loop with MLflow logging
def evaluate_model(model, test_loader, device, logger):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"Validation Accuracy: {accuracy * 100:.2f}%")
    mlflow.log_metric('val_accuracy', accuracy)
    return accuracy, all_labels, all_preds

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes, logger):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("confusion_matrix.png")
    logger.info("Confusion Matrix saved.")
    mlflow.log_artifact("confusion_matrix.png")
    plt.show()

# Main script
if __name__ == "__main__":
    data_dir = 'Tinyembeddings'  # Path to the main folder with accents
    num_classes = 12  # Number of accent classes

    # Set up logging
    logging.basicConfig(filename='training.log', level=logging.INFO)
    logger = logging.getLogger()

    # MLflow setup
    mlflow.start_run()

    # Label encoding for accents
    from sklearn.preprocessing import LabelEncoder
    accents = sorted(os.listdir(data_dir))
    label_encoder = LabelEncoder()
    label_encoder.fit(accents)

    # Log model parameters
    mlflow.log_param("num_classes", num_classes)

    # Load train/test data
    train_loader, test_loader, labels = load_data_and_split(data_dir, label_encoder)

    # Compute class weights
    class_weights = compute_class_weights(labels)
    class_weights = class_weights.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Define model, loss function, optimizer
    model = SimpleANN(input_dim=64, seq_len=230, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Log number of parameters
    num_params = count_trainable_parameters(model)
    logger.info(f"Number of trainable parameters: {num_params}")
    mlflow.log_param('num_params', num_params)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # Training the model
    num_epochs = 10
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = train_model(model, train_loader, criterion, optimizer, device, early_stopping, logger)
        print(f"Training Loss: {train_loss:.4f}")

        # Validate after each epoch
        accuracy, all_labels, all_preds = evaluate_model(model, test_loader, device, logger)
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")

        # Early stopping check
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            print("Early stopping")
            break

    # Save the final model
    torch.save(model.state_dict(), "final_model.pth")
    mlflow.log_artifact("final_model.pth")
    logger.info("Model saved as final_model.pth")

    # Confusion Matrix after training
    # Confusion Matrix after training
    plot_confusion_matrix(y_true=all_labels, y_pred=all_preds, classes=label_encoder.classes_, logger=logger)


    # End MLflow run
    mlflow.end_run()

    print("Training completed")
