import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from pyquaternion import Quaternion
def ego_to_global(ego_pos, ego_pose):
    """Converte coordinate da ego vehicle a global frame."""
    ego_rot = Quaternion(ego_pose['rotation'])
    ego_trans = np.array(ego_pose['translation'])
    rotated = ego_rot.rotate(ego_pos)
    return (rotated + ego_trans).tolist()
def plot_prediction(input_seq, target_seq, pred_seq, title="Global Coordinate Trajectory"):
    input_seq = np.array(input_seq)
    target_seq = np.array(target_seq)
    pred_seq = np.array(pred_seq)

    plt.figure(figsize=(8, 6))

    def draw_path_with_arrows(points, color, label, arrow_style='-|>', linewidth=2, linestyle='-'):
        if len(points) < 2:
            return

        plt.plot(points[:, 0], points[:, 1], linestyle=linestyle, color=color, label=label)

        # Calcola vettori direzionali
        dx = np.diff(points[:, 0])
        dy = np.diff(points[:, 1])
        x = points[:-1, 0]
        y = points[:-1, 1]

        plt.quiver(
            x, y, dx, dy,
            angles='xy', scale_units='xy', scale=1,
            width=0.005, color=color, headwidth=3, headlength=5
        )

    draw_path_with_arrows(input_seq, 'blue', f'Input trajectory (past) [{len(input_seq)}]')
    draw_path_with_arrows(target_seq, 'red', f'Ground Truth (future) [{len(target_seq)}]')
    draw_path_with_arrows(pred_seq, 'green', f'Predicted trajectory [{len(pred_seq)}]')

    plt.plot(0, 0, 'ks', markersize=10, label='Ego vehicle (origin)')

    plt.legend()
    plt.xlabel('X position (global)')
    plt.ylabel('Y position (global)')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()



def is_augmented(sample_ann_token):
    # sample_ann_token è un dict oppure una stringa/null
    if isinstance(sample_ann_token, dict):
        return sample_ann_token.get('is_augmented', False)
    return False  # se è una stringa reale non è augmented
def plot_prediction_with_ego(input_seq, target_seq, pred_seq, ego_traj, title="Global Coordinate Trajectory"):
    input_seq = np.array(input_seq)
    target_seq = np.array(target_seq)
    pred_seq = np.array(pred_seq)
    ego_traj = np.array(ego_traj)

    plt.figure(figsize=(8, 6))

    def draw_path_with_arrows(points, color, label, arrow_style='-|>', linewidth=2, linestyle='-'):
        if len(points) < 2:
            return
        plt.plot(points[:, 0], points[:, 1], linestyle=linestyle, color=color, label=label)
        dx = np.diff(points[:, 0])
        dy = np.diff(points[:, 1])
        x = points[:-1, 0]
        y = points[:-1, 1]
        plt.quiver(
            x, y, dx, dy,
            angles='xy', scale_units='xy', scale=1,
            width=0.005, color=color, headwidth=3, headlength=5
        )

    draw_path_with_arrows(input_seq, 'blue', f'Input trajectory (past) [{len(input_seq)}]')
    draw_path_with_arrows(target_seq, 'red', f'Ground Truth (future) [{len(target_seq)}]')
    draw_path_with_arrows(pred_seq, 'green', f'Predicted trajectory [{len(pred_seq)}]')
    draw_path_with_arrows(ego_traj, 'black', f'Ego vehicle trajectory', linestyle='--')

    plt.legend()
    plt.xlabel('X position (global)')
    plt.ylabel('Y position (global)')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()



def visualize_trajectory(trucksc, debug_preds, first_ann_token, mode="Ego"):
    with open("metadata.json") as f:
        metadata = json.load(f)

    matched_sample = None
    print("First annotation token:", first_ann_token)

    for sample in debug_preds:
        ann_token_entry = sample['ann_tokens'][0]

        if isinstance(ann_token_entry, dict):
            if ann_token_entry.get('is_augmented', False):
                continue
            token_to_check = ann_token_entry.get('annotation_token', None)
        else:
            token_to_check = ann_token_entry

        if token_to_check == first_ann_token:
            matched_sample = sample
            print("✅ matched sample:", matched_sample)
            break

    if matched_sample is None:
        print(f"❌ Nessuna predizione trovata contenente il token {first_ann_token}")
        return

    input_seq = torch.tensor(matched_sample['input'])
    target_seq = torch.tensor(matched_sample['gt'])
    pred_seq = torch.tensor(matched_sample['pred'])
    ann_tokens = matched_sample['ann_tokens']

    print(f"Input: {input_seq.shape}, Target: {target_seq.shape}, Prediction: {pred_seq.shape}")
    print(f"Matched annotation tokens: {ann_tokens}")

    if mode.lower() == "ego":
        plot_prediction(input_seq, target_seq, pred_seq,
                        title=f"[Ego] Prediction for annotation {first_ann_token}")
    elif mode.lower() == "global":
        # Converti tutto in global
        all_poses = []
        for token in ann_tokens:
            ann = trucksc.get("sample_annotation", token)
            sample = trucksc.get("sample", ann['sample_token'])
            ego_pose = trucksc.get("ego_pose", sample['ego_pose_token'])
            all_poses.append(ego_pose)

        # Conversione a global
        input_global = [ego_to_global([pt[0], pt[1], 0.0], all_poses[i]) for i, pt in enumerate(input_seq)]
        target_global = [ego_to_global([pt[0], pt[1], 0.0], all_poses[i + len(input_seq)]) for i, pt in enumerate(target_seq)]
        pred_global = [ego_to_global([pt[0], pt[1], 0.0], all_poses[i + len(input_seq)]) for i, pt in enumerate(pred_seq)]

        # Estrai anche traiettoria dell'ego vehicle
        ego_positions = [np.array(pose['translation'][:2]) for pose in all_poses]

        # Chiama plot
        plot_prediction_with_ego(input_global, target_global, pred_global, ego_positions,
                                 title=f"[Global] Prediction for annotation {first_ann_token}")
    else:
        print(f"⚠️ Modalità '{mode}' non supportata. Usa 'Ego' o 'Global'.")
