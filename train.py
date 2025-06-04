import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

class TrajectoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, ann_tokens_window, is_aug = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), ann_tokens_window, is_aug

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, pred_len=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, x):
        batch_size = x.size(0)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        _, (h, c) = self.lstm(x, (h, c))

        last_output = x[:, -1, :].unsqueeze(1)
        preds = []

        for _ in range(self.pred_len):
            out, (h, c) = self.lstm(last_output, (h, c))
            out = self.dropout(out)
            pred = self.fc(out.squeeze(1))
            preds.append(pred)
            last_output = pred.unsqueeze(1)

        return torch.stack(preds, dim=1)
def trajectory_length(traj):
    diffs = np.diff(traj, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.sum(dists)

# ----------------------------
# Setup
# ----------------------------
def train_model(samples, num_epochs=50, batch_size= 4, lr= 0.001):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dataset = TrajectoryDataset(samples)
  train_size = int(0.8 * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  model = TrajectoryLSTM().to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
  criterion = nn.SmoothL1Loss()

  # ----------------------------
  # Training
  # ----------------------------
  for epoch in range(num_epochs):
      model.train()
      total_loss = 0.0

      for batch in dataloader:
          x, y, _, _ = batch
          x, y = x.to(device), y.to(device)

          optimizer.zero_grad()
          pred = model(x)
          loss = criterion(pred, y)
          loss.backward()
          optimizer.step()

          total_loss += loss.item()

      avg_train_loss = total_loss / len(dataloader)

      model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for val_batch in val_dataloader:
              x_val, y_val, _, _ = val_batch
              x_val, y_val = x_val.to(device), y_val.to(device)
              pred_val = model(x_val)
              val_loss += criterion(pred_val, y_val).item()

      avg_val_loss = val_loss / len(val_dataloader)
      print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

  # ----------------------------
  # Salvataggio predizioni finali (solo non augmentati)
  # ----------------------------
  torch.save(model, 'model_full.pth')


  #model = torch.load('/content/Trajectory-Prediction/model_full.pth')

  debug_predictions = []
  model.eval()
  with torch.no_grad():
      for input_seq, target_seq, ann_tokens, is_aug in samples:
          if not is_aug:
              x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
              y = torch.tensor(target_seq, dtype=torch.float32).unsqueeze(0).to(device)
              pred = model(x)
              debug_predictions.append({
                  'input': input_seq.tolist(),
                  'pred': pred[0].cpu().numpy().tolist(),
                  'gt': target_seq.tolist(),
                  'ann_tokens': ann_tokens
              })



  lengths = []
  errors = []

  for pred in debug_predictions:
      gt = np.array(pred['gt'])      # Ground truth trajectory (seq_len, 2)
      pr = np.array(pred['pred'])    # Predicted trajectory (seq_len, 2)

      lengths.append(trajectory_length(gt))
      errors.append(np.mean(np.linalg.norm(pr - gt, axis=1)))  # MAE per sequenza

  avg_length = np.mean(lengths)
  avg_mae = np.mean(errors)

  accuratezza_pct = (1 - (avg_mae / avg_length)) * 100

  print(f"Lunghezza media traiettorie: {avg_length:.4f}")
  print(f"MAE medio: {avg_mae:.4f}")
  print(f"Accuratezza percentuale: {accuratezza_pct:.2f}%")

  # ----------------------------
  # Verifica conteggio
  # ----------------------------
  expected = sum(1 for s in samples if not s[3])
  actual = len(debug_predictions)
  print(f"\nPredizioni salvate: {actual} / {expected} (non augmentati)\n")
  assert actual == expected, "⚠️ Mismatch tra sample non augmentati e predizioni salvate!"
  return model, debug_predictions