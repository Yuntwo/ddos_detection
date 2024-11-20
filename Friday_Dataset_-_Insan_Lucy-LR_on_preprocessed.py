# Import necessary libraries
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import pickle

# Redirect print output to a file
import sys
sys.stdout = open("training_log.txt", "w")

# Load the dataset
df = pd.read_csv("preProcessed_FeatureClean_AttackTypes.csv")

# Drop non-numeric columns if needed (assuming 'Label' is the target)
features = df.drop(columns=['Label'])
target = df['Label']

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)

# Split into train and validation sets
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
y_train, y_val = train_test_split(target, test_size=0.2, random_state=42)

# Create DataLoaders for parallel processing
train_dataset = TensorDataset(X_train)
val_dataset = TensorDataset(X_val)
num_workers = 64  # Number of workers for DataLoader

# Define the auto-encoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define parameter grid for hyperparameter tuning
param_grid = {
    'hidden_dim1': [64],
    'hidden_dim2': [32],
    'learning_rate': [1e-3],
    'batch_size': [32]
}

# Define loss function
loss_fn = nn.MSELoss()

# Initialize results
best_score = float('inf')
best_params = None
best_model = None

# Perform grid search
for params in ParameterGrid(param_grid):
    # Initialize the model with given parameters
    model = Autoencoder(input_dim=X.shape[1], hidden_dim1=params['hidden_dim1'], hidden_dim2=params['hidden_dim2'])
    optimizer = Adam(model.parameters(), lr=params['learning_rate'])
    batch_size = params['batch_size']
    
    # Create DataLoader instances with multiple workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch[0]  # Extract tensor from dataset tuple
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0]
                val_outputs = model(batch)
                val_loss += loss_fn(val_outputs, batch).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the best model based on validation loss
    if val_loss < best_score:
        best_score = val_loss
        best_params = params
        best_model = model.state_dict()

# Print best parameters and best score
print("Best Parameters:", best_params)
print("Best Validation Loss:", best_score)

# Load the best model
model.load_state_dict(best_model)

# Save the trained model
with open("autoencoder_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate the model on validation data
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    val_loss = loss_fn(val_outputs, X_val).item()

# Convert to numpy arrays for metric calculation
y_val_pred = val_outputs.numpy()
y_val_true = y_val.numpy()

# Calculate evaluation metrics
f1 = f1_score(y_val_true, np.round(y_val_pred), average="weighted")
accuracy = accuracy_score(y_val_true, np.round(y_val_pred))
precision = precision_score(y_val_true, np.round(y_val_pred), average="weighted")
recall = recall_score(y_val_true, np.round(y_val_pred), average="weighted")

print(f"Final Validation Loss: {val_loss:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Calculate and plot ROC-AUC
fpr, tpr, _ = roc_curve(y_val_true, y_val_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")

# Close the log file
sys.stdout.close()
sys.stdout = sys.__stdout__
