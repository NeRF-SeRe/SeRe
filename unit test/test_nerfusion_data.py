import argparse
from tqdm import tqdm

from sere.nerfusion.provider import NeRFusionDataset

opt = argparse.Namespace(**{
    'fusion_path': 'data/replica',
    'fusion_scenes': ['office0', 'room0'],
    'load_to_mem': False,
    'preload': False,
    'scale': 0.33,
    'patch_size': 1,
    'offset': [0, 0, 0],
    'bound': 2,
    'fp16': True,
    'local_K': 3,
    'n_fusion_levels': 8,
    'num_rays': 4096,
    'rand_pose': -1,
    'noise_pose': None,
    'error_map': False,
})

loader = NeRFusionDataset(opt, device='cuda').dataloader()

for _ in tqdm(loader):
    pass
