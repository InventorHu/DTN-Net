import numpy as np
import pickle
import argparse
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

# --------------------------
# 1. Affordance Evaluation (with division-by-zero protection)
# --------------------------
def affordance_eval(affordance_list, result):
    """
    Evaluate affordance prediction performance.
    Fixed edge cases: division by zero, invalid weight averaging.
    Args:
        affordance_list: list of target affordance classes
        result: loaded result dictionary from detection
    Returns:
        mIoU: mean Intersection over Union
        Acc: overall point-wise accuracy
        mAcc: mean per-class accuracy
    """
    num_correct = 0
    num_all = 0
    num_points = {aff: 0 for aff in affordance_list}
    num_label_points = {aff: 0 for aff in affordance_list}
    num_correct_fg_points = {aff: 0 for aff in affordance_list}
    num_correct_bg_points = {aff: 0 for aff in affordance_list}
    num_union_points = {aff: 0 for aff in affordance_list}
    num_appearances = {aff: 0 for aff in affordance_list}

    for shape in result:
        for affordance in shape['affordance']:
            if affordance not in affordance_list:
                continue
            
            label = np.transpose(shape['full_shape']['label'][affordance])
            prediction = shape['result'][affordance][0]
            
            num_appearances[affordance] += 1
            num_correct += np.sum(label == prediction)
            num_all += 2048
            num_points[affordance] += 2048
            num_label_points[affordance] += np.sum(label == 1.0)
            num_correct_fg_points[affordance] += np.sum((label == 1.0) & (prediction == 1.0))
            num_correct_bg_points[affordance] += np.sum((label == 0.0) & (prediction == 0.0))
            num_union_points[affordance] += np.sum((label == 1.0) | (prediction == 1.0))
    
    # Convert to numpy arrays for safe calculation
    correct_fg = np.array(list(num_correct_fg_points.values()))
    union = np.array(list(num_union_points.values()))
    weights = np.array(list(num_appearances.values()))
    total_points = np.array(list(num_points.values()))
    
    # Safe IoU calculation (avoid division by zero)
    iou_values = np.where(union != 0, correct_fg / union, 0.0)
    if np.sum(weights) != 0:
        mIoU = np.average(iou_values, weights=weights)
    else:
        mIoU = np.mean(iou_values)
    
    # Overall accuracy
    Acc = num_correct / num_all if num_all != 0 else 0.0

    # Per-class accuracy
    total_correct_per_class = np.array(list(num_correct_fg_points.values())) + np.array(list(num_correct_bg_points.values()))
    acc_per_class = np.where(total_points != 0, total_correct_per_class / total_points, 0.0)
    mAcc = np.mean(acc_per_class)
    
    return mIoU, Acc, mAcc

# --------------------------
# 2. Pose Evaluation (identical to original version)
# --------------------------
def pose_eval(result):
    """
    Evaluate 6DoF pose estimation performance.
    Matches original implementation exactly — no modifications.
    Args:
        result: loaded result dictionary
    Returns:
        mESM: mean Euclidean distance minimum
        mCR: mean recall rate (distance <= 0.2)
    """
    all_min_dist = []
    all_rate = []
    
    for obj in result:
        for affordance in obj['affordance']:
            # Convert GT rotation matrices to quaternions + translation
            gt_poses = []
            for pose_mat in obj['pose'][affordance]:
                quat = R.from_matrix(pose_mat[:3, :3]).as_quat()
                trans = pose_mat[:3, 3]
                gt_poses.append(np.concatenate([quat, trans]))
            gt_poses = np.array(gt_poses)
            
            pred_poses = obj['result'][affordance][1]
            
            # Compute pose distance matrix
            distances = cdist(gt_poses, pred_poses)
            # Recall rate: distance <= 0.2
            recall = np.sum(np.any(distances <= 0.2, axis=1)) / len(gt_poses)
            all_rate.append(recall)
            
            # Minimum L2 distance between GT and predictions
            l2_dists = np.sqrt(np.sum((gt_poses[:, None, :] - pred_poses) ** 2, axis=2))
            min_dist = np.min(l2_dists)
            
            # Filter unstable cases (original logic preserved)
            if min_dist <= 1.0:
                all_min_dist.append(min_dist)
    
    mESM = np.mean(np.array(all_min_dist)) if all_min_dist else 0.0
    mCR = np.mean(np.array(all_rate)) if all_rate else 0.0
    return mESM, mCR

# --------------------------
# 3. Main Evaluation Pipeline
# --------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate affordance and pose prediction results (no data filtering)"
    )
    parser.add_argument("--result", required=True, help="Path to result .pkl file")
    args = parser.parse_args()

    # Official affordance label list used in training
    AFFORDANCE_LIST = [
        'grasp to pour', 'grasp to stab', 'stab', 'pourable', 'lift',
        'wrap_grasp', 'listen', 'contain', 'displaY', 'grasp to cut',
        'cut', 'wear', 'openable', 'grasp'
    ]

    # Load full detection results
    with open(args.result, 'rb') as f:
        result = pickle.load(f)
    
    # Run evaluation
    mIoU, Acc, mAcc = affordance_eval(AFFORDANCE_LIST, result)
    print(f"[Affordance] mIoU: {mIoU:.4f}, Acc: {Acc:.4f}, mAcc: {mAcc:.4f}")
    
    mESM, mCR = pose_eval(result)
    print(f"[Pose] mESM: {mESM:.4f}, mCR: {mCR:.4f}")

if __name__ == "__main__":
    main()