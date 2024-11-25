import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import Adam
from torch.nn.functional import cross_entropy


# Dataset Class
class AccentDataset(Dataset):
    def __init__(self, data_dir, label_encoder):
        self.files = []
        self.labels = []
        for accent_folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, accent_folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(".npy"):
                        self.files.append(os.path.join(folder_path, file))
                        self.labels.append(accent_folder)

        # Encode labels
        self.label_encoder = label_encoder
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        features = np.load(file_path)  # Shape: [230, 64]
        features = torch.tensor(features, dtype=torch.float32)  # Convert to tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_values": features, "labels": label}


# Tiny Conformer Model
class TinyConformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=4, num_heads=4):
        super(TinyConformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=0.1, activation="relu"
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        x = self.transformer_encoder(x)  # [seq_len, batch_size, hidden_dim]
        x = x.mean(dim=0)  # Global average pooling: [batch_size, hidden_dim]
        logits = self.classifier(x)  # [batch_size, num_classes]
        return logits


# Compute Metrics
def compute_metrics(preds, labels):
    predictions = preds.argmax(axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Train Function
def train(model, train_loader, val_loader, optimizer, class_weights, num_epochs=10, device="cpu"):
    model.to(device)
    class_weights = class_weights.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_values)
            loss = cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_values)
                loss = cross_entropy(logits, labels, weight=class_weights)
                val_loss += loss.item()

                preds.append(logits.cpu().numpy())
                true_labels.append(labels.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
        metrics = compute_metrics(preds, true_labels)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"Metrics: {metrics}")


# Prepare Dataset
data_dir = "Tinyembeddings"
label_encoder = LabelEncoder()
dataset = AccentDataset(data_dir, label_encoder)

# Split Dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Data Loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: {
    "input_values": torch.stack([i["input_values"] for i in x]),
    "labels": torch.tensor([i["labels"] for i in x]),
})
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: {
    "input_values": torch.stack([i["input_values"] for i in x]),
    "labels": torch.tensor([i["labels"] for i in x]),
})

# Handle Class Imbalance
class_weights = compute_class_weight("balanced", classes=np.unique(dataset.labels), y=dataset.labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Instantiate Model
input_dim = 64  # Embedding feature size
hidden_dim = 128  # Hidden size for transformer layers
num_classes = len(label_encoder.classes_)
model = TinyConformer(input_dim, hidden_dim, num_classes)

# Optimizer
optimizer = Adam(model.parameters(), lr=5e-4)

# Train Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model, train_loader, val_loader, optimizer, class_weights, num_epochs=10, device=device)
