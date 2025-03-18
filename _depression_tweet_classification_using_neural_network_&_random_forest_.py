import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import optuna

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))



import gdown
import pandas as pd

# Google Drive file ID (Extracted from the link)
file_id = "1lrw6c8ZPQEawXw7do1VbdwBY0wk01C8D"

# Generate direct download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Download the file
output = "dataset.csv"
gdown.download(url, output, quiet=False)

# Read the CSV file
df = pd.read_csv(output)

# Display first few rows
print(df.head())

# Drop unnecessary columns
df.drop(columns=["Unnamed: 0"], inplace=True)

df.dropna(inplace=True)


# Load Sentence Transformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings in batch mode (faster processing)
print("Computing embeddings... ⏳")
X = embedding_model.encode(df["processed_text"].tolist(), batch_size=256, convert_to_numpy=True)

y = df["sentiment"].values # Labels: (134349,)


# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
batch_size = 512
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Calculating best parameters and hypertunning model.
# Define Neural Network with Activation Function Choice
class DepressionClassifier(nn.Module):
    def __init__(self, dropout_rate, activation_fn):
        super(DepressionClassifier, self).__init__()
        self.fc1 = nn.Linear(384, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.activation = activation_fn()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(32, 2)  # Output layer

    def forward(self, x):
        x = self.dropout1(self.activation(self.bn1(self.fc1(x))))
        x = self.dropout2(self.activation(self.bn2(self.fc2(x))))
        x = self.dropout3(self.activation(self.bn3(self.fc3(x))))
        return self.fc4(x)  # Raw logits

# Optuna Objective Function
def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameter search space
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD", "RAdam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    scheduler_name = trial.suggest_categorical("scheduler", ["ReduceLROnPlateau", "CosineAnnealingLR"])
    activation_name = trial.suggest_categorical("activation", ["ELU", "LeakyReLU"])

    # Activation Function Selection
    activation_fn = nn.ELU if activation_name == "ELU" else nn.LeakyReLU

    # Initialize model
    model = DepressionClassifier(dropout_rate, activation_fn).to(device)

    # Select optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Select scheduler
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.2)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0  # Track best validation accuracy
    patience, patience_counter = 3, 0  # Early stopping criteria

    # Training loop
    for epoch in range(10):  # Increase training epochs
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total  # Validation Accuracy
        scheduler.step(loss if scheduler_name == "ReduceLROnPlateau" else acc)  # Update Scheduler

        # Early Stopping
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0  # Reset counter
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch} with best val acc: {best_acc:.4f}")
                break

    return best_acc  # Return Best Validation Accuracy

# Run Optuna Optimization
study = optuna.create_study(direction="maximize")  # Maximize Accuracy
study.optimize(objective, n_trials=50)  # Run 50 trials for better tuning

# Best Hyperparameters
print(f"Best Hyperparameters: {study.best_params}")
# in progress
# Finally fitting the model with best parameter
class DepressionClassifier(nn.Module):
    def __init__(self):
        super(DepressionClassifier, self).__init__()
        self.fc1 = nn.Linear(384, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 32)  # New hidden layer
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.1)

        self.fc4 = nn.Linear(32, 2)  # Output layer

    def forward(self, x):
        x = self.dropout1(self.elu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.elu(self.bn2(self.fc2(x))))
        x = self.dropout3(self.elu(self.bn3(self.fc3(x))))  # New layer
        return self.fc4(x)  # Keep raw logits



# Initialize Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_nn = DepressionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.AdamW(model_nn.parameters(), lr=0.001, weight_decay=1e-4)
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

#optimizer = optim.SGD(model_nn.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.2, verbose=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

#optimizer = optim.AdamW(model_nn.parameters(), lr=0.0005, weight_decay=1e-3)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)


import time  # Ensure time is imported
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

epochs = 100
best_val_loss = float("inf")
early_stopping_patience = 10
no_improvement_epochs = 0

print("\n🔹 Training Neural Network...")

for epoch in tqdm(range(epochs), desc="Training Progress", leave=True):
    start_time = time.perf_counter()  # Use perf_counter for better precision
    model_nn.train()
    total_loss = 0.0

    # Training loop
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model_nn(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # Sum loss instead of averaging in loop

    total_loss /= len(train_loader)  # Compute average once outside the loop

    # Validation loop
    model_nn.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model_nn(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    val_loss /= len(test_loader)  # Compute average once outside the loop
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    accuracy = accuracy_score(all_labels, all_preds)
    epoch_time = time.perf_counter() - start_time  # More accurate timing

    print(f"🔹 Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Val Loss: {val_loss:.4f} - Accuracy: {accuracy:.4f} - Time: {epoch_time:.2f}s")

    # Reduce LR if validation loss plateaus
    scheduler.step(val_loss)

    # Save Best Model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model_nn.state_dict(), "best_model.pth")
        print("✅ Best model saved!")
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1

    # Early Stopping
    if no_improvement_epochs >= early_stopping_patience:
        print("⏹️ Early stopping triggered. Training stopped.")
        break

# Final Performance Metrics
print("\n🔹 Final Model Performance:")
print(classification_report(all_labels, all_preds))
# in progress


