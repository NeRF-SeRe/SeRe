import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import tqdm


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_replica_data(basedir, half_res=False):
    splits = ['train', 'val', 'test']
    metas = {}
    transform = {"frames": []}
    with open(os.path.join(basedir, '..', 'cam_params.json'), 'r') as f:
        intrinsics = json.load(f)['camera']
        transform['fl_x'] = intrinsics['fx']
        transform['fl_y'] = intrinsics['fy']
        transform['cx'] = intrinsics['cx']
        transform['cy'] = intrinsics['cy']
        transform['w'] = intrinsics['w']
        transform['h'] = intrinsics['h']
    with open(os.path.join(basedir, 'traj.txt'), "r") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        transform['frames'].append({'transform_matrix': c2w,
                                    'file_path': os.path.join('results', f'frame{str(idx).zfill(6)}.jpg'),
                                    })

    H = int(transform['h']) 
    W = int(transform['w'])
    frames_all = transform["frames"]
    all_imgs = []
    all_poses = []
    counts = [0]
    #Process train_data
    for split in ["train","val","test"]:
        poses = []
        imgs = []
        if split == "train":
            frames = frames_all[:-3]
        if split == "val":
            frames = frames_all[-3:-2]
        if split == "test":
            frames = frames_all[-2:-1]

        for f in tqdm.tqdm(frames, desc=f'{split} Loading data'):

            f_path = os.path.join(basedir, f['file_path'])
            pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]

            if H is None or W is None:
                H = image.shape[0] 
                W = image.shape[1] 

            # add support for the alpha channel as a mask.
            if image.shape[-1] == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

            if image.shape[0] != H or image.shape[1] !=W:
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                
            image = image.astype(np.float32) / 255 # [H, W, 3/4]
            imgs.append(image)
            poses.append(pose)
            


        # imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        imgs = np.array(imgs)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

 

    # load intrinsics
    if 'fl_x' in transform or 'fl_y' in transform:
        fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) 
        fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) 
    elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
        # blender, assert in radians. already downscaled since we use H/W
        fl_x = W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
        fl_y = H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
        if fl_x is None: fl_x = fl_y
        if fl_y is None: fl_y = fl_x
    else:
        raise RuntimeError('Failed to load focal length, please check the transforms.json!')

    cx = (transform['cx'] ) if 'cx' in transform else (W / 2)
    cy = (transform['cy'] ) if 'cy' in transform else (H / 2)

    intrinsics = np.array([fl_x, fl_y, cx, cy])
    

    # H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta['camera_angle_x'])
    # focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        fl_x /=2
        fl_y /=2

        imgs_half_res = np.zeros((imgs.shape[0], H, W, imgs.shape[-1]))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    
    K = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])
        
    return imgs, poses, render_poses, [H, W, fl_x], K,i_split


