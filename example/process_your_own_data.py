import os
import yaml
import torch
import mmcv
import open3d as o3d
import numpy as np
from collections import defaultdict
from pyquaternion import Quaternion
from copy import deepcopy


def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=n_threads
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities

def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original= None):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)

def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd


def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=None,
):
    # No deepcopy needed — callers create fresh point clouds
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        pcd.estimate_normals(params)
        pcd.orient_normals_towards_camera_location()

    return pcd


def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True
    )

def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's
        Returns:
            ([indices], [distances])

    """
    if len(verts1) == 0 or len(verts2) == 0:
        return [], []

    from scipy.spatial import KDTree
    tree = KDTree(verts1)
    distances, indices = tree.query(verts2, k=1)

    return indices.tolist(), distances.tolist()


def rotation_z(angle):
    """Construct a 3x3 rotation matrix around Z axis (replaces scipy Rotation.from_euler)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def _make_4x4(R, t):
    """Build a 4x4 homogeneous transform from 3x3 rotation and 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compute_transform_matrix(src_calibrated, src_ego_pose,
                              dst_calibrated, dst_ego_pose):
    """Precompute a single 4x4 matrix: src_lidar -> world -> dst_lidar."""
    T1 = _make_4x4(Quaternion(src_calibrated['rotation']).rotation_matrix,
                   np.array(src_calibrated['translation']))
    T2 = _make_4x4(Quaternion(src_ego_pose['rotation']).rotation_matrix,
                   np.array(src_ego_pose['translation']))
    T3 = _make_4x4(Quaternion(dst_ego_pose['rotation']).rotation_matrix.T,
                   -(Quaternion(dst_ego_pose['rotation']).rotation_matrix.T
                     @ np.array(dst_ego_pose['translation'])))
    T4 = _make_4x4(Quaternion(dst_calibrated['rotation']).rotation_matrix.T,
                   -(Quaternion(dst_calibrated['rotation']).rotation_matrix.T
                     @ np.array(dst_calibrated['translation'])))
    return T4 @ T3 @ T2 @ T1


def points_in_boxes_xpu(points, boxes):
    """
    Pure torch implementation of points in boxes. runs on CPU or XPU.
    points: (N, 3) tensor
    boxes: (M, 7) tensor [cx, cy, cz, dx, dy, dz, heading]
    (cx, cy, cz) is the bottom center of the box to match mmcv.
    Returns: (N, M) bool tensor
    """
    if len(points) == 0 or len(boxes) == 0:
        return torch.zeros((len(points), len(boxes)), dtype=torch.bool, device=points.device)
        
    heading = boxes[:, 6]
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    
    # Save memory by strictly doing broadcasting on 1D tensor values 
    shifted_x = points[:, 0].unsqueeze(1) - boxes[:, 0].unsqueeze(0)  # (N, M)
    shifted_y = points[:, 1].unsqueeze(1) - boxes[:, 1].unsqueeze(0)  # (N, M)
    shifted_z = points[:, 2].unsqueeze(1) - (boxes[:, 2] + 0.5 * boxes[:, 5]).unsqueeze(0)  # (N, M)
    
    local_x = shifted_x * cos_h + shifted_y * sin_h
    local_y = -shifted_x * sin_h + shifted_y * cos_h
    
    in_box = ((local_x.abs() <= boxes[:, 3] / 2) & 
              (local_y.abs() <= boxes[:, 4] / 2) & 
              (shifted_z.abs() <= boxes[:, 5] / 2))
    return in_box


if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()

    parse.add_argument('--data_path', type=str, required=True)
    parse.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for occupancy files (default: same as data_path)')
    parse.add_argument('--config_path', type=str, default='config.yaml')
    parse.add_argument('--len_sequence', type=int, required=True)    
    parse.add_argument('--to_mesh', action='store_true', default=False)
    parse.add_argument('--with_semantic', action='store_true', default=False)
    parse.add_argument('--whole_scene_to_mesh', action='store_true', default=False)


    args=parse.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.data_path
    os.makedirs(output_dir, exist_ok=True)

    # load config
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    voxel_size = config['voxel_size']
    pc_range = config['pc_range']
    occ_size = config['occ_size']


    path = args.data_path
    pc_path = os.path.join(path,'pc/')
    pc_seman_path = os.path.join(path,'pc_seman/')
    bbox_path = os.path.join(path,'bbox/')
    calib_path = os.path.join(path,'calib/')
    pose_path = os.path.join(path,'pose/')

    lidar_ego_pose0 = np.load(os.path.join(pose_path, 'lidar_ego_pose0.npy'), allow_pickle=True).item()
    lidar_calibrated_sensor0 = np.load(os.path.join(calib_path, 'lidar_calibrated_sensor0.npy'), allow_pickle=True).item()

    len_sequence = args.len_sequence
    frame_list = []

    device = torch.device('xpu') if torch.xpu.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    pc_range_t = torch.tensor(pc_range, dtype=torch.float32, device=device)
    voxel_size_t = torch.tensor(voxel_size, dtype=torch.float32, device=device)
    occ_size_t = torch.tensor(occ_size, dtype=torch.long, device=device)

    from concurrent.futures import ThreadPoolExecutor
    save_executor = ThreadPoolExecutor(max_workers=4)

    def load_frame_data(i):
        if args.with_semantic:
            pc0 = np.array(np.load(os.path.join(pc_seman_path, 'pc_seman_{}.npy'.format(i)), mmap_mode='r'))
        else:
            pc0 = np.array(np.load(os.path.join(pc_seman_path, 'pc_seman_{}.npy'.format(i)), mmap_mode='r')[:,:3])
        boxes = np.load(os.path.join(bbox_path, 'bbox{}.npy'.format(i)))
        object_category = np.load(os.path.join(bbox_path, 'object_category{}.npy'.format(i)))
        boxes_token = np.load(os.path.join(bbox_path, 'boxes_token{}.npy'.format(i)))
        lidar_ego_pose = np.load(os.path.join(pose_path, 'lidar_ego_pose{}.npy'.format(i)), allow_pickle=True).item()
        lidar_calib = np.load(os.path.join(calib_path, 'lidar_calibrated_sensor{}.npy'.format(i)), allow_pickle=True).item()
        return pc0, boxes, object_category, boxes_token, lidar_ego_pose, lidar_calib

    print("Pre-loading frame data...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        loaded_data = list(executor.map(load_frame_data, range(len_sequence)))

    # Convert initial frame processing to XPU
    for i in range(len_sequence):
        pc0, boxes, object_category, boxes_token, lidar_ego_pose, lidar_calibrated_sensor = loaded_data[i]

        pc0_t = torch.from_numpy(pc0).to(device)
        boxes_t = torch.from_numpy(boxes).to(device)

        pib = points_in_boxes_xpu(pc0_t[:, :3], boxes_t)

        if args.with_semantic and pc0_t.shape[1] > 4:
            # =================================================================
            # INSTANCE-LEVEL PIPELINE (HIGH PRECISION)
            # =================================================================
            boxes_token_t = torch.from_numpy(boxes_token).to(device)

            # 1. Anti-trailing (Outlier Removal): 
            # DYNAMICALLY identify which classes are actually tracked as moving objects
            # based on the ground truth bounding boxes in this scene.
            original_object_category = object_category.copy()
            dynamic_classes = torch.from_numpy(np.unique(original_object_category)).to(device)
            
            is_dynamic_point = torch.isin(pc0_t[:, 3], dynamic_classes)
            dynamic_instance_ids = torch.unique(pc0_t[is_dynamic_point, 4])

            # CRITICAL FIX for sparsity: Instance ID `0` is often used as a fallback or 
            # represents static background in CARLA/Occ3D. If even one point of a "dynamic class" 
            # (like a misclassified parked vehicle) has Instance ID `0`, stripping ID `0` 
            # will delete the ENTIRE static scene (roads, buildings, etc. which are all ID 0)!
            dynamic_instance_ids = dynamic_instance_ids[dynamic_instance_ids != 0]
            
            is_dynamic_instance = torch.isin(pc0_t[:, 4], dynamic_instance_ids)
            
            # PERFECT CULLING: 
            # 1. Remove points belonging to any explicitly tracked dynamic instance
            # 2. ALSO explicitly remove ANY point whose semantic class is one of the target dynamic classes.
            # Why? Because some distant/untracked vehicles might have instance ID 0 or be noise.
            # Leaving them in makes them look like floating cyan "ghost" cars in the background map.
            points_mask = ~(is_dynamic_instance | is_dynamic_point)

            # 2. Instance-level Visibility Filter
            # Check if points inside the box ACTUALLY match the box's instance ID
            instance_match = pc0_t[:, 4].unsqueeze(1) == boxes_token_t.unsqueeze(0)
            valid_pib = pib & instance_match
            
            # Box is visible if it contains ANY point belonging to its OWN instance ID
            visible_mask = valid_pib.any(dim=0)
            visible_mask_np = visible_mask.cpu().numpy()

            boxes = boxes[visible_mask_np]
            object_category = object_category[visible_mask_np]
            boxes_token = boxes_token[visible_mask_np]

            valid_pib = valid_pib[:, visible_mask]

            # 3. Object point collection (strictly assigned points only, immune to ground/noise leakage)
            pib_nz = valid_pib.nonzero(as_tuple=False)
            object_points_list = []
            for j in range(valid_pib.shape[1]):
                box_point_indices = pib_nz[pib_nz[:, 1] == j, 0]
                object_points_list.append(pc0_t[box_point_indices])

        else:
            # =================================================================
            # SEMANTIC / BASE PIPELINE (FALLBACK)
            # =================================================================
            original_object_category = object_category.copy()
            original_pib = pib
            
            # Anti-trailing first (using all original boxes)
            points_in_any_box = original_pib.any(dim=1)
            points_mask = ~points_in_any_box
            if args.with_semantic and pc0_t.shape[1] > 3:
                dynamic_categories = torch.from_numpy(np.unique(original_object_category)).to(device)
                is_dynamic_class = torch.isin(pc0_t[:, 3], dynamic_categories)
                points_mask = points_mask & (~is_dynamic_class)

            # Visibility culling
            if args.with_semantic and pc0_t.shape[1] > 3:
                object_category_t = torch.from_numpy(object_category).to(device)
                semantic_match = pc0_t[:, 3].unsqueeze(1) == object_category_t.unsqueeze(0)
                visible_mask = (original_pib & semantic_match).any(dim=0)
            else:
                visible_mask = original_pib.any(dim=0)

            visible_mask_np = visible_mask.cpu().numpy()
            boxes = boxes[visible_mask_np]
            object_category = object_category[visible_mask_np]
            boxes_token = boxes_token[visible_mask_np]
            
            pib = original_pib[:, visible_mask]
            pib_nz = pib.nonzero(as_tuple=False)
            object_points_list = []
            for j in range(pib.shape[1]):
                box_point_indices = pib_nz[pib_nz[:, 1] == j, 0]
                object_points_list.append(pc0_t[box_point_indices])

        ############################# get point mask of the vehicle itself ##########################
        self_range = config['self_range']
        oneself_mask = (torch.abs(pc0_t[:, 0]) > self_range[0]) | \
                       (torch.abs(pc0_t[:, 1]) > self_range[1]) | \
                       (torch.abs(pc0_t[:, 2]) > self_range[2])

        ############################# get static scene segment ##########################
        points_mask = points_mask & oneself_mask
        pc_t = pc0_t[points_mask]


        ################## coordinate conversion to the same (first) LiDAR coordinate  ##################
        T_to_frame0 = compute_transform_matrix(lidar_calibrated_sensor, lidar_ego_pose,
                                                lidar_calibrated_sensor0, lidar_ego_pose0)
        T_to_frame0_t = torch.from_numpy(T_to_frame0).float().to(device)
        
        lidar_pc_points = pc_t[:, :3] @ T_to_frame0_t[:3, :3].T + T_to_frame0_t[:3, 3] # More efficient contiguous tensor mul
        if args.with_semantic and pc_t.shape[1] > 3:
            lidar_pc_points = torch.cat([lidar_pc_points, pc_t[:, 3:4]], dim=1)
        
        lidar_pc_t = lidar_pc_points  # keep as tensor

        frame_data = {"object_tokens": boxes_token,
                      "object_points_list": object_points_list,
                      "lidar_pc": lidar_pc_t,
                      "lidar_ego_pose": lidar_ego_pose,
                      "lidar_calibrated_sensor": lidar_calibrated_sensor,
                      "gt_bbox_3d": boxes,
                      "converted_object_category": object_category,
                      "T_to_frame0": T_to_frame0,
                      "pc_file_name": i}
        frame_list.append(frame_data)

    ################## concatenate all static scene segments on device ################
    lidar_pc_t_base = torch.cat([fd['lidar_pc'] for fd in frame_list], dim=0)

    ################## process all object segments (vectorized on device) ################
    object_token_zoo = []
    object_semantic = []
    
    # Dictionary mapping token -> list of point tensors for that object across frames
    object_points_dict = {}
    
    # Pre-collect data for batched operations on device
    all_pts_to_rot_list = []
    all_rots_list = []
    all_tokens = []
    
    for fd in frame_list:
        for i, object_token in enumerate(fd['object_tokens']):
            # Filter objects stringently, avoid processing tokens that have zero points entirely
            object_points = fd['object_points_list'][i]
            if object_points.shape[0] == 0:
                continue
                
            if object_token not in object_points_dict:
                object_token_zoo.append(object_token)
                object_semantic.append(fd['converted_object_category'][i])
                object_points_dict[object_token] = []
                
            # Both object_points and gt_bbox_3d_t (partially converted) need to be on device.
            # gt_bbox_3d is still numpy in frame_data, convert it on the fly:
            bbox_t = torch.from_numpy(fd['gt_bbox_3d'][i]).to(device=device, dtype=torch.float32)
            centered_points = object_points[:, :3] - bbox_t[:3]
            
            all_pts_to_rot_list.append(centered_points)
            all_rots_list.append(-bbox_t[6]) # Negative heading for inverse rotation
            all_tokens.append(object_token)

    if all_pts_to_rot_list:
        # Perform batched rotation on the GPU
        pts_counts = [p.shape[0] for p in all_pts_to_rot_list]
        concat_pts = torch.cat(all_pts_to_rot_list, dim=0)
        
        all_rots_t = torch.stack(all_rots_list)
        c = torch.cos(all_rots_t)
        s = torch.sin(all_rots_t)
        
        # Repeat rotation coefficients for each point in the respective object
        repeats = torch.tensor(pts_counts, device=device)
        c_expanded = torch.repeat_interleave(c, repeats)
        s_expanded = torch.repeat_interleave(s, repeats)
        
        x = concat_pts[:, 0]
        y = concat_pts[:, 1]
        z = concat_pts[:, 2]
        
        rot_x = x * c_expanded - y * s_expanded
        rot_y = x * s_expanded + y * c_expanded
        rotated_pts = torch.stack([rot_x, rot_y, z], dim=1)
        
        # Split back into individual tensors
        split_sizes = pts_counts
        rotated_list = torch.split(rotated_pts, split_sizes)
        
        for k, token in enumerate(all_tokens):
            object_points_dict[token].append(rotated_list[k])

    for token in object_points_dict:
        object_points_dict[token] = torch.cat(object_points_dict[token], dim=0)

    # List of tensor points, one per unique object token
    object_points_vertice_t = [object_points_dict[token][:, :3] for token in object_token_zoo]
    
    if args.whole_scene_to_mesh:
        pcd_mesh = o3d.geometry.PointCloud()
        pcd_mesh.points = o3d.utility.Vector3dVector(lidar_pc_t_base[:, :3].cpu().numpy())
        preprocess(pcd_mesh, config)  # estimates normals in-place
        mesh, _ = create_mesh_from_map(None, 11, config['n_threads'],
                                       config['min_density'], pcd_mesh)
        lidar_pc_t_base = torch.from_numpy(np.asarray(mesh.vertices, dtype=float)).float().to(device)

    # Pre-build token -> index map for O(1) lookup
    token_to_idx = {token: k for k, token in enumerate(object_token_zoo)}

    # (No need to transfer massive background array; already on device as lidar_pc_t_base)
    # (No need to pre-transfer object points; already on device as object_points_vertice_t)
    
    corners_local = torch.tensor([
        [pc_range_t[0], pc_range_t[1], pc_range_t[2]],
        [pc_range_t[0], pc_range_t[1], pc_range_t[5]],
        [pc_range_t[0], pc_range_t[4], pc_range_t[2]],
        [pc_range_t[0], pc_range_t[4], pc_range_t[5]],
        [pc_range_t[3], pc_range_t[1], pc_range_t[2]],
        [pc_range_t[3], pc_range_t[1], pc_range_t[5]],
        [pc_range_t[3], pc_range_t[4], pc_range_t[2]],
        [pc_range_t[3], pc_range_t[4], pc_range_t[5]]
    ], dtype=torch.float32, device=device)
    
    for i in range(len(frame_list)):
        fd = frame_list[i]

        ################## convert the static scene to the target coordinate system ##############
        lidar_calibrated_sensor = fd['lidar_calibrated_sensor']
        lidar_ego_pose = fd['lidar_ego_pose']
        
        # --- NEW CODE: Broad phase filter in frame0 coordinates to accelerate to O(N) ---
        T_to_frame0 = fd['T_to_frame0']
        T_to_frame0_t = torch.from_numpy(T_to_frame0).float().to(device)
        
        T_frame = np.linalg.inv(T_to_frame0)
        T_frame_t = torch.from_numpy(T_frame).float().to(device)
        corners_global = corners_local @ T_to_frame0_t[:3, :3].T + T_to_frame0_t[:3, 3]
        min_req = corners_global.min(dim=0)[0]
        max_req = corners_global.max(dim=0)[0]
        
        broad_mask = (lidar_pc_t_base[:, 0] >= min_req[0]) & (lidar_pc_t_base[:, 0] <= max_req[0]) \
                   & (lidar_pc_t_base[:, 1] >= min_req[1]) & (lidar_pc_t_base[:, 1] <= max_req[1]) \
                   & (lidar_pc_t_base[:, 2] >= min_req[2]) & (lidar_pc_t_base[:, 2] <= max_req[2])
        local_lidar_pc_t_base = lidar_pc_t_base[broad_mask]
        # ------------------------------------------------------------
        
        point_cloud = local_lidar_pc_t_base[:, :3] @ T_frame_t[:3, :3].T + T_frame_t[:3, 3] # More efficient contiguous tensor mul
        if args.with_semantic and local_lidar_pc_t_base.shape[1] > 3:
            point_cloud_with_semantic = torch.cat([point_cloud, local_lidar_pc_t_base[:, 3:4]], dim=1)

        gt_bbox_3d_t = torch.from_numpy(fd['gt_bbox_3d']).float().to(device)

        ################## batched bbox placement and kernel eval ##############
        valid_indices = []
        valid_tokens_k = []
        pts_counts = []
        
        for j, object_token in enumerate(fd['object_tokens']):
            if object_token in token_to_idx:
                valid_indices.append(j)
                k = token_to_idx[object_token]
                valid_tokens_k.append(k)
                pts_counts.append(object_points_vertice_t[k].shape[0])
                
        if len(valid_indices) > 0:
            concat_pts = torch.cat([object_points_vertice_t[k] for k in valid_tokens_k], dim=0) # (N_tot, 3)
            repeats = torch.tensor(pts_counts, device=device)
            
            # Vectorized Transform without Batched Matrix Mult (saves immense GPU memory and time)
            boxes_expanded = torch.repeat_interleave(gt_bbox_3d_t[valid_indices], repeats, dim=0)
            
            x = concat_pts[:, 0]
            y = concat_pts[:, 1]
            z = concat_pts[:, 2]
            
            c_exp = torch.cos(boxes_expanded[:, 6])
            s_exp = torch.sin(boxes_expanded[:, 6])
            
            rot_x = x * c_exp - y * s_exp
            rot_y = x * s_exp + y * c_exp
            
            transformed_pts = torch.stack([rot_x, rot_y, z], dim=1) + boxes_expanded[:, :3]
            
            # Math proves: local_x = x, local_y = y. 
            local_z = z - 0.5 * boxes_expanded[:, 5]
            
            in_box = (x.abs() <= boxes_expanded[:, 3] / 2) & \
                     (y.abs() <= boxes_expanded[:, 4] / 2) & \
                     (local_z.abs() <= boxes_expanded[:, 5] / 2)
                     
            valid_transformed_pts = transformed_pts[in_box]
            scene_points = torch.cat([point_cloud, valid_transformed_pts])
            
            if args.with_semantic:
                semantics_active = torch.tensor([object_semantic[k] for k in valid_tokens_k], dtype=torch.float32, device=device)
                valid_semantics = torch.repeat_interleave(semantics_active, repeats, dim=0)[in_box].unsqueeze(1)
                valid_semantic_pts = torch.cat([valid_transformed_pts, valid_semantics], dim=1)
                scene_semantic_points = torch.cat([point_cloud_with_semantic, valid_semantic_pts])
        else:
            scene_points = point_cloud
            if args.with_semantic:
                scene_semantic_points = point_cloud_with_semantic

        ################## remain points with a spatial range ##############
        spatial_mask = (scene_points[:, 0] > pc_range_t[0]) & (scene_points[:, 0] < pc_range_t[3]) \
               & (scene_points[:, 1] > pc_range_t[1]) & (scene_points[:, 1] < pc_range_t[4]) \
               & (scene_points[:, 2] > pc_range_t[2]) & (scene_points[:, 2] < pc_range_t[5])
        scene_points = scene_points[spatial_mask]
        
        if args.with_semantic:
            scene_semantic_points = scene_semantic_points[spatial_mask]

        if args.to_mesh and not args.whole_scene_to_mesh:
            ################## get mesh via Possion Surface Reconstruction ##############
            # mesh ops still on CPU numpy
            pcd_mesh = o3d.geometry.PointCloud()
            pcd_mesh.points = o3d.utility.Vector3dVector(scene_points[:, :3].cpu().numpy())
            preprocess(pcd_mesh, config)
            mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'],
                                           config['min_density'], pcd_mesh)
            scene_points = torch.from_numpy(np.asarray(mesh.vertices, dtype=float)).float().to(device)

            ################## re-filter after mesh reconstruction ##############
            mask = (scene_points[:, 0] > pc_range_t[0]) & (scene_points[:, 0] < pc_range_t[3]) \
                   & (scene_points[:, 1] > pc_range_t[1]) & (scene_points[:, 1] < pc_range_t[4]) \
                   & (scene_points[:, 2] > pc_range_t[2]) & (scene_points[:, 2] < pc_range_t[5])
            scene_points = scene_points[mask]

        ################## convert points to voxels (optimized XPU unique) ##############
        coords = torch.floor((scene_points[:, :3] - pc_range_t[:3]) / voxel_size_t).long()
        # Keep valid voxel indices
        valid = (coords[:, 0] >= 0) & (coords[:, 0] < occ_size_t[0]) & \
                (coords[:, 1] >= 0) & (coords[:, 1] < occ_size_t[1]) & \
                (coords[:, 2] >= 0) & (coords[:, 2] < occ_size_t[2])
        coords = coords[valid]
        
        # O(N) uniqueness using boolean grid assignment (replaces O(N log N) torch.unique)
        occ_grid = torch.zeros(tuple(occ_size_t.tolist()), dtype=torch.bool, device=device)
        occ_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        coords = torch.nonzero(occ_grid) # Gets unique coordinates directly
        
        fov_voxels = (coords.float() + 0.5) * voxel_size_t + pc_range_t[:3]
        save_executor.submit(np.save, os.path.join(output_dir, 'occupancy_gt{}.npy'.format(i)), fov_voxels.cpu().numpy())


        if args.with_semantic:
            ################## O(N) direct grid assignment to replace KDTree  ##############
            semantic_voxel = torch.zeros(tuple(occ_size_t.tolist()), dtype=torch.float32, device=device)
            sem_coords = torch.floor((scene_semantic_points[:, :3] - pc_range_t[:3]) / voxel_size_t).long()
            
            valid_sem = (sem_coords[:, 0] >= 0) & (sem_coords[:, 0] < occ_size_t[0]) & \
                        (sem_coords[:, 1] >= 0) & (sem_coords[:, 1] < occ_size_t[1]) & \
                        (sem_coords[:, 2] >= 0) & (sem_coords[:, 2] < occ_size_t[2])
            sem_coords = sem_coords[valid_sem]
            sem_labels = scene_semantic_points[valid_sem, 3]
            
            # Scatter labels (last written wins for duplicates)
            semantic_voxel[sem_coords[:, 0], sem_coords[:, 1], sem_coords[:, 2]] = sem_labels
            
            # Read back labels for the unique occupied voxels we found earlier
            dense_semantic = semantic_voxel[coords[:, 0], coords[:, 1], coords[:, 2]]
            
            dense_voxels_with_semantic = torch.cat([coords.float(), dense_semantic.unsqueeze(1)], dim=1)
            save_executor.submit(np.save, os.path.join(output_dir, 'occupancy_gt_with_semantic{}.npy'.format(i)), dense_voxels_with_semantic.cpu().numpy())

        if torch.xpu.is_available():
            torch.xpu.empty_cache()

    save_executor.shutdown(wait=True)
    print('finish scene!')