import os
import trimesh
import numpy as np
import pickle
import argparse
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import pyrender
from PIL import Image
from utils.visualization import create_gripper_marker  # 保留原始夹爪创建函数

# Set the environment variable to use EGL for offscreen rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"

# 原始颜色配置
color_code_1 = np.array([0, 0, 255])    # 蓝色：可用性点云
color_code_2 = np.array([0, 255, 0])    # 绿色：夹爪姿态
num_pose = 5  # 每个物体-功能对可视化的姿势数量
DEFAULT_MESM_THRESHOLD = 0.2  # mESM阈值

def parse_args():
    """解析命令行参数，保留原始参数+新增筛选相关参数"""
    parser = argparse.ArgumentParser(description="Visualize affordance and gripper poses (with evaluation set filter)")
    parser.add_argument("--result_file", required=True, help="Path to the result.pkl file")
    parser.add_argument("--visualize", choices=["result", "pose"], default="result", 
                        help="Choose to visualize 'result' or 'pose' data")
    parser.add_argument("--category", type=str, default="Mug", help="Filter by semantic class (e.g., 'Mug')")
    parser.add_argument("--mesm_threshold", type=float, default=DEFAULT_MESM_THRESHOLD, help="mESM threshold for valid poses")
    parser.add_argument("--output_dir", type=str, default="/home/coop/HuWei/original_model/30D_point_model/result_pictures", 
                        help="Output directory for visualization images")
    args = parser.parse_args()
    return args

def filter_evaluation_data(result_data):
    """筛选后20%的评估集数据（保留你的筛选规则）"""
    total = len(result_data)
    start_idx = int(total * 0.8)
    eval_data = result_data[start_idx:]
    print(f"\n=== 评估集筛选结果 ===")
    print(f"总数据量: {total} 个物体")
    print(f"评估集数据量: {len(eval_data)} 个物体 (80%~100%)")
    print(f"评估集索引范围: {start_idx} ~ {total-1}")
    return eval_data

def get_best_poses_by_mESM(pred_poses, gt_poses, top_k=5, mesm_threshold=0.2):
    """按mESM筛选最优姿态（保留你的筛选规则）"""
    if len(pred_poses) == 0 or len(gt_poses) == 0:
        return [], []
    # 提取平移向量
    gt_trans = np.array([pose[:3, 3] for pose in gt_poses if pose.shape == (4,4)])
    pred_trans = np.array([pose[4:7] for pose in pred_poses if len(pose) >= 7])
    if len(gt_trans) == 0 or len(pred_trans) == 0:
        return [], []
    # 计算mESM
    distance_matrix = cdist(gt_trans, pred_trans)
    pred_mesm = distance_matrix.min(axis=0)
    # 过滤并排序
    valid_mask = pred_mesm <= mesm_threshold
    valid_pred_indices = np.where(valid_mask)[0]
    if len(valid_pred_indices) == 0:
        return [], []
    valid_mesm = pred_mesm[valid_pred_indices]
    sorted_indices = valid_pred_indices[np.argsort(valid_mesm)[:top_k]]
    # 获取最优姿态
    best_poses = [pred_poses[i] for i in sorted_indices]
    return best_poses, [pred_mesm[i] for i in sorted_indices]

def render_scene(scene, image_path):
    """保留原始的渲染逻辑，不做修改"""
    pr_scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
    
    # Add lighting（原始光照配置）
    directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    point_light = pyrender.PointLight(color=np.ones(3), intensity=10.0)
    pr_scene.add(directional_light, pose=np.eye(4))
    pr_scene.add(point_light, pose=np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0.5],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ]))

    # Add geometry to the scene（原始几何添加逻辑）
    for name, geometry in scene.geometry.items():
        if isinstance(geometry, trimesh.PointCloud):
            vertices = geometry.vertices
            colors = geometry.colors
            if colors.dtype == np.float64:
                colors = (colors * 255).astype(np.uint8)
            cloud = pyrender.Mesh.from_points(vertices, colors=colors)
            pr_scene.add(cloud)
        elif isinstance(geometry, trimesh.Trimesh):
            mesh = pyrender.Mesh.from_trimesh(geometry, smooth=False)
            pr_scene.add(mesh)

    # Set up the camera（原始相机配置）
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.33)

    # 动态调整相机位置（原始逻辑）
    bounds = scene.bounds
    if bounds is not None:
        center = np.mean(bounds, axis=0)
        size = np.max(bounds[1] - bounds[0])
        distance = size * 2.0
        camera_pose = np.array([
            [1, 0, 0, center[0]],
            [0, 1, 0, center[1]],
            [0, 0, 1, center[2] + distance],
            [0, 0, 0, 1]
        ])
    else:
        camera_pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.12],
            [0, 0, 0, 1]
        ])

    pr_scene.add(camera, pose=camera_pose)

    # Render the scene（原始渲染逻辑）
    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    color, _ = renderer.render(pr_scene)
    
    # Save the image
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    Image.fromarray(color).save(image_path)
    renderer.delete()

if __name__ == "__main__":
    args = parse_args()
    
    # 加载结果文件
    try:
        with open(args.result_file, 'rb') as f:
            result = pickle.load(f)
        print(f"成功加载结果文件: {args.result_file}")
    except Exception as e:
        print(f"加载结果文件失败: {e}")
        exit(1)
    
    # 1. 筛选后20%的评估集数据（保留你的筛选规则）
    eval_data = filter_evaluation_data(result)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n=== 可视化配置 ===")
    print(f"目标类别: {args.category}")
    print(f"可视化模式: {args.visualize}")
    print(f"mESM阈值: {args.mesm_threshold}")
    print(f"输出目录: {args.output_dir}")
    
    vis_count = 0
    # 2. 遍历评估集数据（仅筛选指定类别）
    for i, obj_data in enumerate(eval_data):
        if obj_data['semantic class'] != args.category:
            continue
        
        # 遍历物体的每个可用性
        for affordance in obj_data['affordance']:
            try:
                # 生成带颜色的点云（原始逻辑）
                affordance_probs = obj_data['result'][affordance][0].flatten()
                colors = (affordance_probs[:, None] * color_code_1).astype(np.uint8)
                point_cloud = trimesh.points.PointCloud(
                    obj_data['full_shape']['coordinate'],
                    colors=colors
                )
                
                T_matrices = []
                if args.visualize == "result":
                    # 3. 按mESM筛选最优姿态（保留你的筛选规则）
                    pred_poses = obj_data['result'][affordance][1]
                    gt_poses = obj_data['pose'][affordance]
                    best_poses, _ = get_best_poses_by_mESM(
                        pred_poses, gt_poses, num_pose, args.mesm_threshold
                    )
                    # 原始的pose变换矩阵生成逻辑
                    for pose in best_poses:
                        quaternion = pose[:4]
                        translation = pose[4:7]
                        transform = np.eye(4)
                        transform[:3, :3] = R.from_quat(quaternion).as_matrix()
                        transform[:3, 3] = translation
                        T_matrices.append(transform)
                elif args.visualize == "pose":
                    # 原始的真实pose处理逻辑
                    poses_data = obj_data['pose'][affordance][:num_pose]
                    T_matrices = [np.eye(4) for _ in range(len(poses_data))]
                    for idx, pose in enumerate(poses_data):
                        T_matrices[idx][:3, :3] = pose[:3, :3]
                        T_matrices[idx][:3, 3] = pose[:3, 3]
                
                # 跳过无有效姿态的情况
                if len(T_matrices) == 0:
                    print(f"[物体{i}-{affordance}] 无有效姿态，跳过")
                    continue
                
                # 原始的夹爪创建逻辑（使用utils中的create_gripper_marker）
                grippers = [
                    create_gripper_marker(color=color_code_2/255).apply_transform(t) 
                    for t in T_matrices
                ]
                
                # 原始的场景创建和渲染逻辑
                scene = trimesh.Scene([point_cloud] + grippers)
                safe_affordance = affordance.replace(" ", "_")
                image_path = os.path.join(args.output_dir, f"{safe_affordance}_{i}_{args.visualize}.png")
                render_scene(scene, image_path)
                
                vis_count += 1
                print(f"[{vis_count}] 成功保存：{image_path}")
                
            except Exception as e:
                print(f"[物体{i}-{affordance}] 可视化失败: {e}")
                continue
    
    print(f"\n=== 可视化完成 ===")
    print(f"共生成 {vis_count} 张可视化图片")
    print(f"图片保存路径: {args.output_dir}")