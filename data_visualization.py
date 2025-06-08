import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from pyquaternion import Quaternion
from truckscenes.utils.geometry_utils import view_points, transform_matrix, BoxVisibility
from truckscenes.utils.data_classes import Box
from PIL import Image
def ego_to_global(ego_pos, ego_pose):
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
    
    if isinstance(sample_ann_token, dict):
        return sample_ann_token.get('is_augmented', False)
    return False  
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

def get_cam_channel(trucksc, anntoken):
        ann_record = trucksc.get('sample_annotation', anntoken)
        sample_record = trucksc.get('sample', ann_record['sample_token'])
        cams = [key for key in sample_record['data'].keys() if 'CAMERA' in key]
        for cam in cams:
            _, boxes, _ = trucksc.get_sample_data(sample_record['data'][cam],
                                                      box_vis_level=BoxVisibility.ANY,
                                                      selected_anntokens=[anntoken])
            if len(boxes) > 0:          
                break                   
        return cam

def visualize_trajectory(trucksc, debug_preds, first_ann_token):
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
        print(f"❌ No prediction found containing the token {first_ann_token}")
        return

    input_seq = torch.tensor(matched_sample['input'])
    target_seq = torch.tensor(matched_sample['gt'])
    pred_seq = torch.tensor(matched_sample['pred'])
    ann_tokens = matched_sample['ann_tokens']

    print(f"Input: {input_seq.shape}, Target: {target_seq.shape}, Prediction: {pred_seq.shape}")
    print(f"Matched annotation tokens: {ann_tokens}")

    ann = trucksc.get("sample_annotation", ann_tokens[0])
    sample = trucksc.get("sample", ann['sample_token'])
    
    cam_channel = get_cam_channel(trucksc, first_ann_token)
    all_poses = []

    for token in ann_tokens:
        ann = trucksc.get("sample_annotation", token)
        sample = trucksc.get("sample", ann['sample_token'])
        
        
        cam_token = sample['data'].get(cam_channel, None)
        if cam_token is None:
            print(f"⚠️ No data for channel {cam_channel} in sample {sample['token']}")
            continue

        cam_data = trucksc.get("sample_data", cam_token)
        ego_pose = trucksc.get("ego_pose", cam_data['ego_pose_token'])

        all_poses.append(ego_pose)

    if len(all_poses) != len(ann_tokens):
        print(f"⚠️ Poses retrieved: {len(all_poses)} out of {len(ann_tokens)} annotations")

    
    input_global = [ego_to_global([pt[0], pt[1], 0.0], all_poses[i]) for i, pt in enumerate(input_seq)]
    target_global = [ego_to_global([pt[0], pt[1], 0.0], all_poses[i + len(input_seq)]) for i, pt in enumerate(target_seq)]
    pred_global = [ego_to_global([pt[0], pt[1], 0.0], all_poses[i + len(input_seq)]) for i, pt in enumerate(pred_seq)]

    
    ego_positions = [np.array(pose['translation'][:2]) for pose in all_poses]

    
    plot_prediction_with_ego(input_global, target_global, pred_global, ego_positions,
                            title=f"[Global] Prediction for annotation {first_ann_token}")

    return matched_sample, input_seq, target_seq, pred_seq
def create_box(center, size, orientation=[1, 0, 0, 0], name='prediction'):
    
    return Box(center=center, size=[size['width'], size['length'], size['height']],
               orientation=Quaternion(orientation), name=name)

def transform_global_to_sensor(global_pos, ego_pose, calibrated_sensor):
    
    # Global -> Ego
    translation_ego = np.array(global_pos) - np.array(ego_pose['translation'])
    rotation_ego = Quaternion(ego_pose['rotation']).inverse.rotate(translation_ego)

    # Ego -> Sensor (Camera)
    translation_sensor = rotation_ego - np.array(calibrated_sensor['translation'])
    rotation_sensor = Quaternion(calibrated_sensor['rotation']).inverse.rotate(translation_sensor)

    return rotation_sensor


def transform_orientation(orientation_quat, ego_pose, sensor_calib):
    q_global = Quaternion(orientation_quat)
    q_ego = Quaternion(ego_pose['rotation']).inverse * q_global
    q_sensor = Quaternion(sensor_calib['rotation']).inverse * q_ego
    return q_sensor

def render_gt_vs_prediction(trucksc, gt_ann_token, pred_xy, annotation_data):
    gt_ann = trucksc.get('sample_annotation', gt_ann_token)
    sample_token = gt_ann['sample_token']
    sample = trucksc.get('sample', sample_token)
    cam_channel = get_cam_channel(trucksc, gt_ann_token)
    cam_token = sample['data'].get(cam_channel, None)
    if cam_token is None:
        print(f"⚠️ No data for channel {cam_channel} in sample {sample['token']}")
    cam_data = trucksc.get('sample_data', cam_token)
    calib = trucksc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_intrinsic = np.array(calib['camera_intrinsic'])

    ego_pose = trucksc.get('ego_pose', cam_data['ego_pose_token'])

    def get_box_size(annotation_token, annotation_data):
        for ann in annotation_data:
            if ann['token'] == annotation_token:
                size = ann['size']
                if isinstance(size, dict):
                    return {
                        'width': float(size['width']),
                        'length': float(size['length']),
                        'height': float(size['height']),
                    }
                return {
                    'width': float(size[0]),
                    'length': float(size[1]),
                    'height': float(size[2]),
                }
        return None

    size = get_box_size(gt_ann_token, annotation_data)
    if size is None:
        print(f"Box dimensions not found for {gt_ann_token}")
        return

    # GT Box
    center_global_gt = gt_ann['translation']
    center_sensor_gt = transform_to_sensor_frame(center_global_gt, ego_pose, calib)
    orientation_sensor_gt = transform_orientation(gt_ann['rotation'], ego_pose, calib)

    box_gt = Box(
        center=center_sensor_gt,
        size=[size['width'], size['length'], size['height']],
        orientation=orientation_sensor_gt,
        name='GT'
    )

    # Prediction Box
    pred_ego = [pred_xy[0], pred_xy[1], 0.0]  
    pred_pos_global = ego_to_global(pred_ego, ego_pose)
    pred_pos_global[2] = center_global_gt[2]  
    center_sensor_pred = transform_to_sensor_frame(pred_pos_global, ego_pose, calib)
    orientation_sensor_pred = orientation_sensor_gt  

    box_pred = Box(
        center=center_sensor_pred,
        size=[size['width'], size['length'], size['height']],
        orientation=orientation_sensor_pred,
        name='Prediction'
    )
    print(f"Box GT: {box_gt}")
    print(f"Box Prediction: {box_pred}")
    
    trucksc.render_annotation(gt_ann_token,
                          box_vis_level=BoxVisibility.ANY,
                          )   

    
    fig  = plt.gcf()
    ax   = plt.gca()

    
    box_gt.render(ax, view=cam_intrinsic, normalize=True, colors=(np.array([1.0, 0.0, 0.0]),) * 3)
    box_pred.render(ax, view=cam_intrinsic, normalize=True, colors=(np.array([0.0, 1.0, 0.0]),) * 3)

    plt.title("GT (red) vs Prediction (green)")
    plt.tight_layout()
    plt.show()
def render_box(trucksc, pred_seq, matched_sample):
    with open("/content/man-truckscenes/man-truckscenes/v1.0-mini/sample_annotation.json") as f:
        annotation_data = json.load(f)

    for i in range(7):
        gt_token = matched_sample['ann_tokens'][3 + i]
        pred_xy = pred_seq[i].numpy()
        render_gt_vs_prediction(trucksc, gt_token, pred_xy, annotation_data)


def transform_to_sensor_frame(global_pos, ego_pose, sensor_calib):
    """Global ➝ Ego ➝ Sensor."""
    pos = np.array(global_pos)
    ego_rot = Quaternion(ego_pose['rotation'])
    ego_trans = np.array(ego_pose['translation'])
    pos_ego = np.dot(ego_rot.inverse.rotation_matrix, pos - ego_trans)

    sensor_rot = Quaternion(sensor_calib['rotation'])
    sensor_trans = np.array(sensor_calib['translation'])
    pos_sensor = np.dot(sensor_rot.inverse.rotation_matrix, pos_ego - sensor_trans)
    return pos_sensor

def render_trajectory(trucksc, matched_sample, pred_seq):
    gt_token = matched_sample['ann_tokens'][3]
    gt_ann = trucksc.get('sample_annotation', gt_token)
    sample_token = gt_ann['sample_token']
    sample = trucksc.get('sample', sample_token)

    cam_channel = get_cam_channel(trucksc, gt_token)
    cam_token = sample['data'].get(cam_channel, None)
    if cam_token is None:
        print(f"⚠️ Camera {cam_channel} non trovata.")
        return

    cam_data = trucksc.get('sample_data', cam_token)
    calib = trucksc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_intrinsic = np.array(calib['camera_intrinsic'])

    ego_pose = trucksc.get('ego_pose', cam_data['ego_pose_token'])

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    trucksc.render_sample_data(cam_token, box_vis_level=BoxVisibility.ANY, ax=ax)

    gt_points_sensor = []
    pred_points_sensor = []

    for i in range(len(pred_seq)):
      gt_token = matched_sample['ann_tokens'][3 + i]
      gt_ann = trucksc.get('sample_annotation', gt_token)
      center_global_gt = gt_ann['translation']

      
      sample_token = gt_ann['sample_token']
      sample_i = trucksc.get('sample', sample_token)
      cam_token_i = sample_i['data'].get(cam_channel, None)
      cam_data_i = trucksc.get('sample_data', cam_token_i)
      ego_pose_i = trucksc.get('ego_pose', cam_data_i['ego_pose_token'])

      # GT ➝ Sensor
      gt_sensor = transform_to_sensor_frame(center_global_gt, ego_pose, calib)
      gt_points_sensor.append(gt_sensor)

      # Pred ➝ Global  ➝ Sensor
      pred_xy = pred_seq[i].numpy()
      pred_ego = [pred_xy[0], pred_xy[1], 0.0]
      pred_global = ego_to_global(pred_ego, ego_pose_i)
      pred_global[2] = center_global_gt[2]  # match altezza
      pred_sensor = transform_to_sensor_frame(pred_global, ego_pose, calib)
      pred_points_sensor.append(pred_sensor)

    
    gt_img = view_points(np.array(gt_points_sensor).T, cam_intrinsic, normalize=True)
    pred_img = view_points(np.array(pred_points_sensor).T, cam_intrinsic, normalize=True)


    ax.plot(gt_img[0, :], gt_img[1, :], 'ro-', label='GT Trajectory')
    ax.plot(pred_img[0, :], pred_img[1, :], 'go-', label='Prediction Trajectory')

    ax.legend()
    plt.title("Trajectory: Ground Truth (red) vs Prediction (green)")
    plt.tight_layout()
    plt.show()

