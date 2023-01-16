<h1 align="center">
NeRF4SeRe: Neural Radiance Fields for Scene Reconstruction
</h1>
<p align="center">
Project of AI3604 Computer Vision, Shanghai Jiao Tong University.
</p>

## TODO
- [x] installation guideline
- [x] training and testing code for NeRF, INeRF and Torch-NGP
- [x] implementation of NeRFusion
- [ ] NICE-SLAM integration

## Installation

1. Environment setup
    You should install google-sparsehash first following the instructions in torchsparse (e.g., `sudo apt-get install libsparsehash-dev` on Ubuntu and `brew install google-sparsehash` on Mac OS).

    Then install the dependencies:
    ```sh
    pip install -r requirements.txt
    pip install git+https://github.com/facebookresearch/pytorch3d.git
    pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
    export PYTHONPATH="/path/to/{ENV_NAME}:$PYTHONPATH"
    ```

2. Dataset preparation
   Please refer to https://github.com/cvg/nice-slam to download the Replica Dataset. For SJTUers, you could download it from [Replica-JBox](https://jbox.sjtu.edu.cn/l/41duY3).

    Then link the dataset
    ```sh
    ln -s /path/to/Replica data/replica
    ```
## Training Pipeline
Adding `--device x` to specify the GPU device (default: 0).

1. NeRF

    ```sh
    python run/main_nerf.py --config configs/nerf/replica/room0.txt
    ```

2. Torch-NGP

    ```sh
    python run/main_ngp.py data/replica/room0/ --workspace output/ngp/replica/room0/ --fp16 --cuda_ray
    ```

3. NeRFusion

    ```sh
    python run/main_nerfusion.py data/replica/room0 --fusion_path data/replica --workspace output/nerfusion/replica/room0 --cuda_ray --fp16 

    python run/main_nerfusion.py data/replica/room0 --fusion_path data/replica --fusion_scenes office0 office1 --workspace output/nerfusion/replica/room0 --cuda_ray --fp16 # change fusion_scenes to specify the scenes to be trained
    ```

4. Refine camera pose (INeRF)

   After you have trained a NeRF model, please put the checkpoints (its name should keep the same as model_name in config) under /ckpts/inerf. Then run:

   ```sh
   python run/main_inerf.py --config configs/inerf/replica/room0.txt
    ```


## Evaluation

If you have already trained a NeRF model, just add `--test` to test it. eg.
```sh
python run/main_ngp.py data/replica/room0/ --workspace output/ngp/replica/room0/ --fp16 --cuda_ray --test # write images

python run/main_ngp.py data/replica/room0/ --workspace output/ngp/replica/room0/ --fp16 --cuda_ray --test --video # generate video
```

## Acknowledgement

* nerf-pytorch
* inerf
* torch-ngp
* NeRFusion
* NeuralRecon
