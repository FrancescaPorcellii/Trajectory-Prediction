import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import uuid
import random

def augment_window(window, noise_std=0.1, translation_range=0.5):
    augmented = []
    delta = np.random.uniform(-translation_range, translation_range, size=2)

    for frame in window:
        new_frame = frame.copy()
        original_pos = np.array(frame['translation'][:2])
        noisy_pos = original_pos + delta + np.random.normal(scale=noise_std, size=2)
        z = frame['translation'][2] if len(frame['translation']) == 3 else 0.0

        new_frame['translation'] = [*noisy_pos, z]

        # Mantieni il vero annotation_token (per riferimento) ma indica che è augmentato
        new_frame['is_augmented'] = True

        # Opzionalmente: salva anche il token reale se vuoi
        new_frame['original_annotation_token'] = frame['annotation_token']

        # Annulla il valore se vuoi evitare errori diretti col renderer
        new_frame['annotation_token'] = frame['annotation_token']+"_"

        augmented.append(new_frame)

    return augmented

def global_to_ego(ego_translation, ego_rotation_quat, obj_global_pos):
    # ego_rotation_quat: [w, x, y, z]
    r = R.from_quat([ego_rotation_quat[1], ego_rotation_quat[2], ego_rotation_quat[3], ego_rotation_quat[0]])
    r_inv = r.inv()
    vec = np.array(obj_global_pos) - np.array(ego_translation)
    pos_ego = r_inv.apply(vec)
    return pos_ego.tolist()

def create_dataset(trucksc ):
  N = 10  # lunghezza della finestra temporale


  trajectory_data = []
  augmented_data = []

  num_aug = 3  # quante augmentazioni per finestra

  for scene_idx, scene in enumerate(trucksc.scene):
      sample_token = scene['first_sample_token']
      sample_sequence = []

      # Raccogli tutti i sample nella scena
      while sample_token:
          sample = trucksc.get('sample', sample_token)
          sample_sequence.append(sample)
          sample_token = sample['next']

      # Scorri la scena frame per frame
      for i in range(len(sample_sequence) - N + 1):
          window = sample_sequence[i:i + N]
          base_sample = window[0]

          # Trova il veicolo "car + moving" più vicino nel primo frame
          nearest = None
          min_dist = float('inf')

          lidar_token = next(v for v in base_sample['data'].values()
                            if 'LIDAR_TOP' in trucksc.get('sample_data', v)['filename'])
          ego_pose = trucksc.get('ego_pose', trucksc.get('sample_data', lidar_token)['ego_pose_token'])
          ego_translation = ego_pose['translation']

          for ann_token in base_sample['anns']:
              ann = trucksc.get('sample_annotation', ann_token)
              if not any(cat in ann['category_name'] for cat in ['vehicle.car', 'vehicle.truck', 'vehicle.motorcycle']):
                continue
              attr_tokens = ann.get('attribute_tokens', [])
              is_moving = any(trucksc.get('attribute', attr)['name'] == 'vehicle.moving'
                              for attr in attr_tokens)
              if not is_moving:
                  continue
              pos = ann['translation']
              dist = np.linalg.norm(np.array(pos[:2]) - np.array(ego_translation[:2]))
              if dist < min_dist:
                  min_dist = dist
                  nearest = ann

          if not nearest:
              continue

          instance_token = nearest['instance_token']
          traj = []
          valid = True

          # Verifica presenza di quell'istanza nei frame successivi
          for sample in window:
              found = False
              for ann_token in sample['anns']:
                  ann = trucksc.get('sample_annotation', ann_token)
                  if ann['instance_token'] != instance_token:
                      continue

                  sd_token = next(v for v in sample['data'].values()
                                  if 'LIDAR_TOP' in trucksc.get('sample_data', v)['filename'])
                  sd = trucksc.get('sample_data', sd_token)
                  ego_pose = trucksc.get('ego_pose', sd['ego_pose_token'])
                  ego_translation = ego_pose['translation']
                  ego_rotation = ego_pose['rotation']

                  pos_ego = global_to_ego(ego_translation, ego_rotation, ann['translation'])

                  traj.append({
                      'scene_idx': scene_idx,
                      'timestamp': sample['timestamp'] / 1e6,
                      'instance_token': instance_token,
                      'annotation_token': ann['token'],
                      'translation': pos_ego
                  })
                  found = True
                  break

              if not found:
                  valid = False
                  break

          if valid:
              trajectory_data.append(traj)  # ogni elemento è una finestra da N frame
  for traj in trajectory_data:
      augmented_data.append(traj)  # finestra originale
      for _ in range(num_aug):
          aug_traj = augment_window(traj)
          augmented_data.append(aug_traj)
  # Salvataggio
  with open('nearest_vehicle_trajectories.json', 'w') as f:
      json.dump(trajectory_data, f, indent=2)
  with open('nearest_vehicle_trajectories_augmented.json', 'w') as f:
      json.dump(augmented_data, f, indent=2)
  print(f"✅ Salvate {len(trajectory_data)} finestre da {N} frame con veicolo più vicino moving")
  print(f"✅ Salvate {len(augmented_data)} finestre da {N} frame con veicolo più vicino moving (augmented {num_aug}x)")
