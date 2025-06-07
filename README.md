# 🚚 Trajectory Prediction in Truck Scenes

![Demo Video](media/video.gif)

> **Predict the next 7 positions of the nearest moving vehicle given the first 3 observations**
> using the **TruckScenes‑Mini** dataset and a custom LSTM model.

This repository supports the report:
📄 *“Trajectory Prediction in Truck Scenes using LSTM under Incomplete Observations.”*

It includes:

* ✅ `Trajectory_Prediction.ipynb`: End‑to‑end Colab notebook (data prep → training → evaluation → visualization)
* 🧠 Pretrained models for three experiment settings
* 🛠 `train.py`: Unified script for training and evaluation
* 🔧 Helper scripts for dataset parsing and plotting
* 🗂 Preselected samples for easy qualitative evaluation

---

## 1 · 🚀 Quick Start (Google Colab)

Run everything from [**this Colab notebook**](https://colab.research.google.com/github/FrancescaPorcellii/Trajectory-Prediction/blob/main/Trajectory_Prediction.ipynb).

| Step                      | Command                                                                                                                                                                                                                                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Launch notebook**    | [Open in Colab](https://colab.research.google.com/github/FrancescaPorcellii/Trajectory-Prediction/blob/main/Trajectory_Prediction.ipynb)                                                                                                                                                                        |
| **2. Clone repository**   | `!git clone https://github.com/FrancescaPorcellii/Trajectory-Prediction.git`                                                                                                                                                                                                                                    |
| **3. Download dataset**   | Paste this in one cell:<br><br>`bash<br>!wget https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/mini/man-truckscenes_metadata_v1.0-mini.zip<br>!wget https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/mini/man-truckscenes_sensordata_v1.0-mini.zip<br>!unzip "man-truckscenes_*.zip"` |
| **4. Install dev-kit**    | `!pip install -U "truckscenes-devkit[all]"`                                                                                                                                                                                                                                                                     |
| **5. Initialise dataset** | `python<br>from truckscenes import TruckScenes<br>trucksc = TruckScenes('v1.0-mini', '/content/man-truckscenes/man-truckscenes', verbose=True)`                                                                                                                                                                 |

---

## 2 · 🧪 Experiments & Training

This project explores **three experimental setups**:

| Setting        | Description                                 | Training Flag  |
| -------------- | ------------------------------------------- | -------------- |
| `standard`    | All 10 time steps are present               | ` no`     |
| `drop_target` | One missing point in the **future segment** | `target` |
| `drop_input`  | One missing point in the **past segment**   | `input`  |

You can use the unified `train.py` script to either train a model from scratch or load pretrained models:

### ▶ Train a new model

```bash
model, debug_preds = train_model(samples, num_epochs=100, batch_size=4, lr=0.001, mode = "save", drop = 'target')
```

### 🔁 Load and evaluate pretrained model

```bash
model, debug_preds = train_model(samples, num_epochs=100, batch_size=4, lr=0.001, mode = "load", drop = 'target')
```

> All three pretrained models are already saved inside the `models/` directory.

---

## 3 · 📊 Visual Examples

Two annotation samples are provided one for a straight trajectory and another for a curved trajectory:

<div align="center">
  <img src="assets/standard_straight.png" width="45%" />
  <img src="assets/drop_input_curved.png"  width="45%" />
  <br>
  <sub>🔵 Past · 🔴 Ground truth · 🟢 Prediction</sub>
</div>

You can also choose **any other annotation token** from the dataset and visualize it.

---

## 🧠 Model Overview

* Architecture: LSTM-based predictor
* Inputs: 3 past observations of the nearest moving agent
* Outputs: 7 future predicted positions
* Losses: Mean Squared Error (MSE), L1 Loss (optional)
* Framework: PyTorch

---

