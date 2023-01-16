# ported from NeuralRecon (https://github.com/zju3dv/NeuralRecon)
import torch
import torch.nn as nn

from sere.nerfusion.fusion.models.backbone import MnasMulti
from sere.nerfusion.fusion.models.neucon_network import NeuConNet
from sere.nerfusion.fusion.utils.nerfusion_utils import tocuda


# local training stage
# trained in the stage: local reconstruction network, radiance field decoder
# global training stage
# trained in the stage (end2end): local reconstruction network, fusion network, radiance field decoder
class NeRFusion(nn.Module):
    def __init__(self, opt):
        super(NeRFusion, self).__init__()
        self.opt = opt
        # alpha = float(self.opt.BACKBONE2D.ARC.split('-')[-1])
        alpha = 1.0
        # other hparams
        self.pixel_mean = torch.Tensor(opt.pixel_norm_mean)
        self.pixel_std = torch.Tensor(opt.pixel_norm_std)
        self.n_scales = opt.n_fusion_levels  # len(self.opt.THRESHOLDS) - 1

        # networks
        self.backbone2d = MnasMulti(alpha)
        self.neucon_net = NeuConNet(opt)

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def forward(self, inputs, fuse='local', device='cuda'):
        '''

        :param inputs: dict: {
            'imgs':                    (Tensor), images,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            others: unused in network
        }
        :param fuse: str, 'local' or 'global'
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results:
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
        }
                 loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        '''
        inputs = tocuda(inputs, device)
        outputs = {}
        imgs = torch.unbind(inputs['images_orig'], 1)  # (Tuple of Tensor), images along views, (batch size, H, W, C)

        # image feature extraction
        # in: images; out: feature maps
        if not self.backbone2d.training:
            self.backbone2d.train()
        features = [self.backbone2d(self.normalizer(img).permute(0, 3, 1, 2)) for img in imgs]
        if not self.backbone2d.training:
            self.backbone2d.eval()
        # features = [self.backbone2d(self.normalizer(img).permute(0, 3, 1, 2)) for img in imgs]

        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs = self.neucon_net(features, inputs, outputs, fuse)

        return outputs
