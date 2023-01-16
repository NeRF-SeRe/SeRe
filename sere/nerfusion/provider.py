import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader
from sere.ngp.utils import generate_rand_transf

from .utils import get_rays, angle, IntrinsicsPoseToProjection

# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10, path=None, scene=None):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path if path is None else path
        self.load_to_mem = opt.load_to_mem
        self.scene = os.path.basename(self.root_path) if scene is None else scene
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.local_K = getattr(opt, 'local_K', 1)
        self.n_fusion_levels = getattr(opt, 'n_fusion_levels', 3)
        self.get_proj = None

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        elif os.path.exists(os.path.join(self.root_path, 'traj.txt')):
            self.mode = 'nice-slam'
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)
        elif self.mode =='nice-slam':
            transform = {"frames": []}
            with open(os.path.join(self.root_path, '..', 'cam_params.json'), 'r') as f:
                intrinsics = json.load(f)['camera']
                transform['fl_x'] = intrinsics['fx']
                transform['fl_y'] = intrinsics['fy']
                transform['cx'] = intrinsics['cx']
                transform['cy'] = intrinsics['cy']
                transform['w'] = intrinsics['w']
                transform['h'] = intrinsics['h']
            with open(os.path.join(self.root_path, 'traj.txt'), "r") as f:
                lines = f.readlines()
            for idx, line in enumerate(lines):
                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                c2w[:3, 1] *= -1
                c2w[:3, 2] *= -1
                if opt.noise_pose is not None:
                    noise_transf = generate_rand_transf(opt.noise_pose[0], opt.noise_pose[1], batch_size=1, device="cpu").to(torch.float64) # (1, 4, 4)
                    c2w = torch.matmul(noise_transf, torch.from_numpy(c2w).unsqueeze(0)).squeeze(0).numpy()
                transform['frames'].append({'transform_matrix': c2w,
                                            'file_path': os.path.join('results', f'frame{str(idx).zfill(6)}.jpg'),
                                            })
        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if (self.mode == 'colmap' or self.mode == 'nice-slam') and type == 'test':
            
            self.poses = []
            self.images = None

            # choose two random poses, and interpolate between.
            f0 = np.random.choice(frames)
            start = pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            section_cnt = 0
            while section_cnt < 3:
                f1 = np.random.choice(frames)
                pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]

                # compute the angle between the two poses
                deg = angle(pose0, pose1)
                if deg < 90: # if the angle is too small, we skip this section.
                    continue

                n_test = int(angle(pose0, pose1))

                rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                slerp = Slerp([0, 1], rots)

                for i in range(n_test + 1):
                    ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = slerp(ratio).as_matrix()
                    pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                    self.poses.append(pose)
                
                f0 = f1
                pose0 = pose1
                section_cnt += 1
            

            # Go back to the start pose
            pose1 = start
            n_test = int(angle(pose0, pose1))

            if n_test != 0:
                rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                slerp = Slerp([0, 1], rots)

                for i in range(n_test + 1):
                    ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = slerp(ratio).as_matrix()
                    pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                    self.poses.append(pose)

        else:  # train or val
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap' or self.mode == 'nice-slam':
                if type == 'train':
                    frames = frames[self.local_K:]
                elif type == 'val':
                    frames = frames[:self.local_K]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data' +
                                            (f' for scene {self.scene}' if self.scene is not None else '')):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = f_path
                if self.load_to_mem:
                    image = self.load_image(f_path)

                self.poses.append(pose)
                self.images.append(image)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            if self.load_to_mem:
                self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
            else:
                self.images = np.array(self.images) # [N,]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([len(self.images), 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [unit test] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [unit test] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        self.len = len(self.poses)
        self.len_fusion = len(self.poses) - self.local_K + 1

    def load_image(self, f_path):
        image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
        if self.H is None or self.W is None:
            self.H = image.shape[0] // self.downscale
            self.W = image.shape[1] // self.downscale

        # add support for the alpha channel as a mask.
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        if image.shape[0] != self.H or image.shape[1] != self.W:
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

        image = image.astype(np.float32) / 255  # [H, W, 3/4]

        return image

    def load_images(self, f_paths):
        images = []
        for f_path in f_paths.flatten():
            images.append(self.load_image(f_path))
        images = torch.from_numpy(np.stack(images, axis=0))
        return images.reshape(f_paths.shape + images.shape[1:])

    def collate(self, index):

        B = len(index) # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            if self.load_to_mem:
                images = self.images[index].to(self.device) # [B, H, W, 3/4]
            else:
                images = self.load_images(self.images[index]).to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def collate_fusion(self, index):

        B = len(index) # a list of length 1
        index = np.stack([np.arange(i, i + self.local_K) for i in index], 0) # [B, K]

        # random pose without gt images.
        if self.rand_pose == 0 or index[0][0] >= len(self.poses):

            poses = rand_poses(B * self.local_K, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
            }

        poses = self.poses[index].reshape(*index.shape, *self.poses.shape[-2:]).to(self.device) # [B, K, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]

        rays = get_rays(poses.view(-1, *poses.shape[-2:]), self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        # these are for nerfusion, we flip the z axis.
        extrinsics = poses.clone()  # [B, K, 4, 4]
        # extrinsics[:, :, :3, 2] *= -1
        fx, fy, cx, cy = self.intrinsics
        # these are for nerfusion, we flip the z axis.
        extrinsics = poses.view(B, self.local_K, *poses.shape[2:])  # [B, K, 4, 4]
        extrinsics[:, :, :3, 3] *= -1
        intrinsics = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], device=self.device).unsqueeze(0).repeat(B, self.local_K, 1, 1)  # [B, K, 3, 3]
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],  # [B * K, N, 3]
            'rays_d': rays['rays_d'],  # [B * K, N, 3]
            'extrinsics': extrinsics,  # [B, K, 4, 4]
            'intrinsics': intrinsics,  # [B, K, 3, 3]
        }

        if self.images is not None:
            if self.load_to_mem:
                images = self.images[index].to(self.device) # [B, H, W, 3/4]
            else:
                images = self.load_images(self.images[index]).to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                rgbs = torch.gather(images.view(B * self.local_K, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B * K, N, 3/4]
            else:
                rgbs = images
            results['images'] = rgbs
            results['images_orig'] = images

        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']

        # dummy voxel origin
        results['vol_origin'] = torch.zeros(B, 3, device=self.device)
        results['vol_origin_partial'] = torch.zeros(B, 3, device=self.device)

        # get projection matrix
        if self.get_proj is None:
            self.get_proj = IntrinsicsPoseToProjection(n_views=self.local_K, n_levels=self.n_fusion_levels)
        results = self.get_proj(results)

        # scene name
        if self.scene is not None:
            results['scene'] = self.scene

        return results

    def dataloader(self, fusion=False):
        if fusion:
            size = self.len_fusion
            collate = self.collate_fusion
        else:
            size = self.len
            if self.training and self.rand_pose > 0:
                size += size // self.rand_pose # index >= size means we use random pose.
            collate = self.collate
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=collate, shuffle=self.training and not fusion, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader


class NeRFusionDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()

        self.opt = opt
        self.root_path = opt.fusion_path
        self.training = type == 'train'
        scene_paths = np.array(glob.glob(os.path.join(self.root_path, '*') + os.path.sep))
        scenes = [os.path.basename(p[:-1]) for p in scene_paths]
        if opt.fusion_scenes is not None:
            scenes, indices, _ = np.intersect1d(scenes, opt.fusion_scenes, return_indices=True)
            scene_paths = scene_paths[indices]

        self.scene_datasets = []
        for scene_path, scene in zip(scene_paths, scenes):
            self.scene_datasets.append(NeRFDataset(opt, device, type, downscale, n_test, scene_path, scene,))
        self.scene_datasets = np.array(self.scene_datasets)
        self.scene_datasets_dict = {s: d for s, d in zip(scenes, self.scene_datasets)}
        self.scene_sizes = [d.len_fusion for d in self.scene_datasets]
        self.scene_indices = np.cumsum(self.scene_sizes)

    def collate(self, index):
        scene_index = np.searchsorted(self.scene_indices, index, side='right')
        scene_index = np.clip(scene_index, 0, len(self.scene_datasets) - 1)
        dataset_index = index - self.scene_indices[scene_index - 1] if scene_index > 0 else index
        return self.scene_datasets[scene_index].item().collate_fusion(dataset_index)

    def dataloader(self, stage='local'):
        # loader = DataLoader(list(range(self.scene_indices[-1])), batch_size=1, collate_fn=self.collate, shuffle=self.training and stage == 'local', num_workers=0)
        # Shuffling will trigger frequent grid update, which costs a lot of time. This could be solve by saving grid states for each scene.
        loader = DataLoader(list(range(self.scene_indices[-1])), batch_size=1, collate_fn=self.collate, shuffle=False, num_workers=0)
        loader._data = self
        loader.has_gt = np.array([data.images is not None for data in self.scene_datasets]).all()
        return loader

    def poses(self, scene):
        return self.scene_datasets_dict[scene].poses

    def intrinsics(self, scene):
        return self.scene_datasets_dict[scene].intrinsics

    def error_map(self, scene):
        return self.scene_datasets_dict[scene].error_map

