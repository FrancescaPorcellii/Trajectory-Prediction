import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, pred_horizon=1):
        """
        sequences: lista di numpy array shape (seq_len, feature_dim)
        pred_horizon: quanti passi in avanti voglio predire (es 1 = prossimo frame)
        """
        self.sequences = sequences
        self.pred_horizon = pred_horizon

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]  # (seq_len, feature_dim)
        seq_len = seq.shape[0]

        # Input: tutti tranne gli ultimi pred_horizon step
        input_seq = seq[:seq_len - self.pred_horizon, :]

        # Target: le posizioni future x,y,z degli ultimi pred_horizon step
        # Qui assumiamo le posizioni sono prime 3 feature (x,y,z)
        target_seq = seq[seq_len - self.pred_horizon:, 0:3]

        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)


class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # prendiamo l'output solo dell'ultimo step temporale
        out = out[:, -1, :]
        out = self.fc(out)
        return out  # predizione posizione (x,y,z)


# Hyperparametri
input_size = 9   # come da feature vettore (x,y,z + size + velocity)
hidden_size = 64
num_layers = 1
output_size = 3  # prediciamo posizione (x,y,z)
learning_rate = 0.001
batch_size = 32
epochs = 10

# Dataset e dataloader
dataset = TrajectoryDataset(all_sequences, pred_horizon=1)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modello, loss, optimizer
model = TrajectoryLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, 3)
        loss = criterion(outputs, targets.squeeze(1))  # targets (batch, 3)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

