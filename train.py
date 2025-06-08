import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

def masked_smooth_l1(pred, target, mask):
    mask = mask.unsqueeze(-1).expand_as(pred)          
    diff = pred[mask] - target[mask]                  
    return F.smooth_l1_loss(diff, torch.zeros_like(diff))
class TrajectoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, ann_tokens_window, is_aug = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), ann_tokens_window, is_aug
class TrajectoryDatasetDrop(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
      x, y, ann_tokens, is_aug, mask = self.samples[idx]
      return (torch.tensor(x, dtype=torch.float32),
              torch.tensor(y, dtype=torch.float32),
              torch.tensor(mask, dtype=torch.bool),
              ann_tokens,
              is_aug)
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

def train_model(samples, num_epochs=50, batch_size= 4, lr= 0.001, mode = 'load', drop = 'no'):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if drop == 'no':
    dataset = TrajectoryDataset(samples)
  else:
    dataset = TrajectoryDatasetDrop(samples)
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
  if mode == 'save':
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
          if drop == 'target':  #only in target to avoid propagating the Nan in the loss
            x, y, mask, _, _ = batch                   
            x = torch.nan_to_num(x, nan=0.0)            
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = masked_smooth_l1(pred, y, mask)      
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
          else:
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
              if drop == 'no':
                x_val, y_val, _, _ = val_batch
                x_val, y_val = x_val.to(device), y_val.to(device)
                pred_val = model(x_val)
                val_loss += criterion(pred_val, y_val).item()
              elif drop == 'target':
                x_val, y_val,mask_val, _, _ = val_batch
                x_val, y_val = x_val.to(device), y_val.to(device)
                pred_val = model(x_val)
                val_loss += masked_smooth_l1(pred_val, y_val, mask_val.to(device)).item()
              else:
                x_val, y_val,mask_val, _, _ = val_batch
                x_val = torch.nan_to_num(x_val, nan=0.0)
                x_val, y_val = x_val.to(device), y_val.to(device)
                pred_val = model(x_val)
                val_loss += criterion(pred_val, y_val).item()
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

   
    if drop == 'no':
      torch.save(model.state_dict(), 'models/model_weights.pth')
    elif drop == 'target':
      torch.save(model.state_dict(), 'models/model_weights_drop.pth')
    else:
      torch.save(model.state_dict(), 'models/model_weights_drop_input.pth')
  elif mode == 'load':

    
    model = TrajectoryLSTM(input_size=2, hidden_size=64, num_layers=2, pred_len=7).to(device)
    if drop == 'no':
      model.load_state_dict(torch.load('models/model_weights.pth'))
    elif drop == 'target':
      model.load_state_dict(torch.load('models/model_weights_drop.pth'))
    else:
      model.load_state_dict(torch.load('models/model_weights_drop_input.pth'))
    

  debug_predictions = []
  model.eval()
  with torch.no_grad():
    if drop == 'no':
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
    elif drop == 'target':
        for input_seq, target_seq, ann_tokens, is_aug, mask in samples:
          if not is_aug:
              x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
              
              y = torch.tensor(target_seq, dtype=torch.float32).unsqueeze(0).to(device)
              pred = model(x)
              debug_predictions.append({
                  'input': input_seq.tolist(),
                  'pred': pred[0].cpu().numpy().tolist(),
                  'gt': target_seq.tolist(),
                  'mask': mask,  
                  'ann_tokens': ann_tokens
              })
    else:
        for input_seq, target_seq, ann_tokens, is_aug, mask in samples:
          if not is_aug:
              x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
              x = torch.nan_to_num(x, nan=0.0)
              y = torch.tensor(target_seq, dtype=torch.float32).unsqueeze(0).to(device)
              pred = model(x)
              debug_predictions.append({
                  'input': input_seq.tolist(),
                  'pred': pred[0].cpu().numpy().tolist(),
                  'gt': target_seq.tolist(),
                  'mask': mask,  
                  'ann_tokens': ann_tokens
              })
  lengths = []
  errors = []
  if drop == 'no':
    for pred in debug_predictions:
        gt = np.array(pred['gt'])      
        pr = np.array(pred['pred'])    

        lengths.append(trajectory_length(gt))
        errors.append(np.mean(np.linalg.norm(pr - gt, axis=1)))  
  else:
    for pred in debug_predictions:
        gt   = np.array(pred['gt'])     
        pr   = np.array(pred['pred'])    
        mask = np.array(pred['mask'])    

        valid_gt = gt[mask]
        valid_pr = pr[mask]

        if len(valid_gt) == 0:
            continue  
        lengths.append(trajectory_length(valid_gt))
        errors.append(np.mean(np.linalg.norm(valid_pr - valid_gt, axis=1)))  
  avg_length = np.mean(lengths)
  avg_mae = np.mean(errors)

  accuratezza_pct = (1 - (avg_mae / avg_length)) * 100

  print(f"Average trajectory length: {avg_length:.4f}")
  print(f"Mean Absolute Error (MAE): {avg_mae:.4f}")
  print(f"Accuracy percentage: {accuratezza_pct:.2f}%")

  expected = sum(1 for s in samples if not s[3])
  actual = len(debug_predictions)
  print(f"\nSaved predictions: {actual} / {expected} (non-augmented)\n")
  assert actual == expected, "⚠️ Mismatch between non-augmented samples and saved predictions!"
  return model, debug_predictions
