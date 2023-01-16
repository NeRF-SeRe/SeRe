import argparse

from sere.nerfusion.provider import NeRFDataset, NeRFusionDataset
from sere.nerfusion.utils import *
from sere.nerfusion.network import NeRFusionNetwork

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--test_stage', type=str, default='local')
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='0')

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--local_lr', type=float, default=1e-2, help="initial learning rate for local fusion")
    parser.add_argument('--global_lr', type=float, default=1., help="initial learning rate for global fusion")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate for nerf (finetune)")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")
    ## fusion options
    parser.add_argument('--fusion_path', type=str, default='', help="path to the dataset for training fusion")
    parser.add_argument('--fusion_scenes', nargs='+', help="scenes to be used for training fusion, e.g., 'room0 room1'")
    parser.add_argument('--local_epochs', type=int, default=2, help="num epochs to train local fusion")
    parser.add_argument('--global_epochs', type=int, default=2, help="num epochs to train global fusion")
    parser.add_argument('--local_K', type=int, default=3, help="collect K frames to get local voxel")
    parser.add_argument('--n_fusion_levels', type=int, default=3, help="number of scale levels for fusion")
    parser.add_argument('--n_voxel_levels', type=int, default=1, help="number of scale levels for the voxel grid")
    parser.add_argument('--sparse_conv_dropout', action='store_true', help="use dropout in sparse conv (of voxel fusion)")
    parser.add_argument('--fusion_full', type=bool, default=True, help="use full fusion")
    parser.add_argument('--no_fusion_full', action='store_false', dest='fusion_full', help="not use full gru fusion")

    ### feature grid options
    parser.add_argument('--base_resolution_rate', type=int, default=48, help="base grid resolution is base_resolution_rate * bound")
    parser.add_argument('--desired_resolution_rate', type=int, default=48, help="desired final grid resolution is base_resolution_rate * bound")
    parser.add_argument('--log2_hashmap_size', type=int, default=40, help="log2 of hashmap size for instant-ngp, large enough to hold all coordinates")
    ## advanced options (for dimensional alignment, shall not be changed without changing the 2D backbone)
    parser.add_argument('--num_levels', type=int, default=3, help="number of levels of resolution")
    parser.add_argument('--level_dim', type=int, default=8, help="feature dimension of each level")

    ### network backbone options
    ## nerf backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    ## nerfusion 2d backbone
    parser.add_argument('--pixel_norm_mean', type=list, default=[103.53, 116.28, 123.675], help="normalizing mean of 2d pixel")
    parser.add_argument('--pixel_norm_std', type=list, default=[1., 1., 1.], help="normalizing std of 2d pixel")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--load_to_mem', action='store_true', help="load all images to memory")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options (This is NOT implemented!!!)
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")
    parser.add_argument('--noise_pose', type=float, nargs='*', default=None, help="noise for rand pose")

    ### Visualize options
    parser.add_argument('--video', action='store_true', help="write video")

    opt = parser.parse_args()

    # Choose device
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.device

    opt.base_resolution = opt.bound * opt.base_resolution_rate
    opt.desired_resolution = opt.bound * opt.desired_resolution_rate

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    print(opt)

    seed_everything(opt.seed)

    model = NeRFusionNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        encoder_opt={
            'base_resolution': opt.base_resolution,
            'desired_resolution': opt.desired_resolution,
            'log2_hashmap_size': opt.log2_hashmap_size,
            'num_levels': opt.num_levels,
            'level_dim': opt.level_dim,
        },
        fusion_opt=argparse.Namespace(**{
            'pixel_norm_mean': opt.pixel_norm_mean,
            'pixel_norm_std': opt.pixel_norm_std,
            'bound': opt.bound,
            'resolution': opt.base_resolution,
            # 'n_voxel_levels': opt.n_voxel_levels,
            'n_fusion_levels': opt.n_fusion_levels,
            'sparse_conv_dropout': opt.sparse_conv_dropout,
            'fusion_full': opt.fusion_full,
        })
    )

    print(model)

    criterion = torch.nn.MSELoss(reduction='none')
    #criterion = partial(huber_loss, reduction='none')
    #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fusion_device = torch.device(1) if torch.cuda.device_count() > 1 else device

    if opt.test:
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]

        trainer = Trainer('ngp', opt, model, device=device, fusion_device=fusion_device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt, stage=opt.test_stage)

        if opt.gui:
            from sere.nerfusion.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            # -- prepare dataloaders --
            if trainer.stage == 'finetune':
                if opt.video:
                    test_loader = NeRFDataset(opt, device=device, type='test', n_test=300).dataloader()
                else:
                    test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            else:
                if trainer.stage == 'global':
                    fusion_loader = NeRFusionDataset(opt, device=device, type='train').dataloader()
                test_loader = NeRFusionDataset(opt, device=device, type='val', downscale=1).dataloader()

            # TODO: staged test support
            if test_loader.has_gt:
                trainer.evaluate(test_loader, fusion_loader)  # blender has gt, so evaluate it.

            if trainer.stage == 'finetune':
                trainer.test(test_loader, write_video=opt.video)  # test

                trainer.save_mesh(resolution=256, threshold=10)

    else:  # train
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]

        trainer = None
        # -- prepare dataloaders --
        if opt.local_epochs > 0 or opt.global_epochs > 0:
            train_dataset = NeRFusionDataset(opt, device=device, type='train')
            if not opt.gui:
                valid_loader = NeRFusionDataset(opt, device=device, type='val', downscale=1).dataloader()
                # also test
                if opt.video:
                    test_loader = NeRFusionDataset(opt, device=device, type='test', n_test=300).dataloader()
                else:
                    test_loader = NeRFusionDataset(opt, device=device, type='test').dataloader()

        if opt.iters > 0:
            nerf_train_dataset = NeRFDataset(opt, device=device, type='train')
            fusion_loader = nerf_train_dataset.dataloader(fusion=True)
            nerf_train_loader = nerf_train_dataset.dataloader()
            if not opt.gui:
                nerf_valid_loader = NeRFDataset(opt, device=device, type='val').dataloader()
                # also test
                if opt.video:
                    nerf_test_loader = NeRFDataset(opt, device=device, type='test', n_test=300).dataloader()
                else:
                    nerf_test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

        # -- train local --
        if opt.local_epochs > 0:
            train_loader = train_dataset.dataloader('local')

            local_iters = opt.local_epochs * len(train_loader)

            # !!! ALWAYS remember to enable & unable optimization for the grid !!!
            model.encoder_unable_update()

            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.local_lr), betas=(0.9, 0.99), eps=1e-15)

            # decay to 0.1 * init_lr at last iter step
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
                                                                      lambda iter: 0.1 ** min(iter / local_iters, 1))

            trainer = Trainer('ngp', opt, model, device=device, fusion_device=fusion_device, workspace=opt.workspace, stage='local', optimizer=optimizer,
                              criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                              scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=1)

            if opt.gui:
                from sere.nerfusion.gui import NeRFGUI
                gui = NeRFGUI(opt, trainer, train_loader)
                gui.render()

            else:
                trainer.train(train_loader, valid_loader, opt.local_epochs, stage='local')

                if test_loader.has_gt:
                    trainer.evaluate(valid_loader)  # blender has gt, so evaluate it.

                trainer.test(valid_loader, write_video=opt.video)  # test and save video

                # trainer.save_mesh(resolution=256, threshold=10)

        # -- train global --
        if opt.global_epochs > 0:
            train_loader = train_dataset.dataloader('global')

            global_iters = opt.global_epochs * len(train_loader)

            model.encoder_unable_update()

            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.global_lr), betas=(0.9, 0.99), eps=1e-15)

            # decay to 0.1 * init_lr at last iter step
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
                                                                      lambda iter: 0.1 ** min(iter / global_iters, 1))

            if trainer is None:
                trainer = Trainer('ngp', opt, model, device=device, fusion_device=fusion_device, workspace=opt.workspace, stage='global', optimizer=optimizer,
                                  criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                                  scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt,
                                  eval_interval=1)
            else:
                trainer.set_optimization_stats(optimizer, scheduler)

            if opt.gui:
                from sere.nerfusion.gui import NeRFGUI
                gui = NeRFGUI(opt, trainer, train_loader)
                gui.render()

            else:
                if valid_loader is None:
                    valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

                # max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
                trainer.train(train_loader, valid_loader, opt.global_epochs, stage='global')

                # also test
                if test_loader.has_gt:
                    trainer.evaluate(valid_loader, fusion_loader=train_loader) # blender has gt, so evaluate it.

                trainer.test(valid_loader, write_video=opt.video) # test and save video

                # trainer.save_mesh(resolution=256, threshold=10)

        # -- train nerf (per scene finetune) --
        if opt.iters > 0:
            model.encoder_enable_update()

            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

            # decay to 0.1 * init_lr at last iter step
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
                                                                      lambda iter: 0.1 ** min(iter / opt.iters, 1))

            if trainer is None:
                trainer = Trainer('ngp', opt, model, device=device, fusion_device=fusion_device, workspace=opt.workspace, stage='finetune', optimizer=optimizer,
                                  criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                                  scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt,
                                  eval_interval=10)
            else:
                trainer.eval_interval = 10
                trainer.set_optimization_stats(optimizer, scheduler)

            if opt.gui:
                from sere.nerfusion.gui import NeRFGUI
                gui = NeRFGUI(opt, trainer, nerf_train_loader)
                gui.render()

            else:
                max_epoch = np.ceil(opt.iters / len(nerf_train_loader)).astype(np.int32)
                trainer.train(nerf_train_loader, nerf_valid_loader, max_epoch, stage='finetune', fusion_loader=fusion_loader)

                # also test
                if nerf_test_loader.has_gt:
                    trainer.evaluate(nerf_test_loader)  # blender has gt, so evaluate it.

                trainer.test(nerf_test_loader, write_video=opt.video)  # test and save video

                # trainer.save_mesh(resolution=256, threshold=10)

