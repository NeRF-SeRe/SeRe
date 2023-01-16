<h1 align="center">
NeRF4SeRe: Neural Radiance Fields for Scene Reconstruction
</h1>
<p align="center">
    Project of AI3604 Computer Vision, SJTU.
    <br />
    <a href="https://github.com/Ailon-Island"><strong>Wei Jiang *</strong></a>
    ·
    <a href="https://www.linkedin.cn/incareer/in/kangrui-mao-376694239"><strong>Kangrui Mao *</strong></a>
    ·
    <a href="https://github.com/blakery-star"><strong>Gu Zhang *</strong></a>
    ·
    <a href="https://haoyuzhen.com"><strong>Haoyu Zhen *</strong></a>
    <br />
    \star = equal contribution
</p>
<p align="center">
  <a href='https://github.com/NeRF-SeRe/SeRe/docs/sere.pdf'>
    <img src='https://img.shields.io/badge/Project%20Paper-PDF-blue?style=flat&logo=Googlescholar&logoColor=blue' alt='Paper PDF'>
  </a>
  <a href="https://sere-cv-project.netlify.app">
    <img alt="Pre" src="https://img.shields.io/badge/Presentation-Slides-green?logo=Hugo&logoColor=green">
  </a>
<a href="https://pytorch.org/get-started/locally/">
    <img alt="PyTorch" src="https://img.shields.io/badge/Pytorch-1.11.0-red?logo=pytorch&logoColor=red">
  </a>
</p>

## TODO
- [x] installation guideline
- [x] training and testing code for NeRF, INeRF and Instant-NGP
- [x] implementation of NeRFusion
- [ ] NICE-SLAM integration
- [ ] visualizations

## Installation
  ```sh
  git clone --recursive https://github.com/NeRF-SeRe/SeRe.git
  cd SeRe
  ```

1. Environment setup
    You should install google-sparsehash first following the instructions in torchsparse (e.g., `sudo apt-get install libsparsehash-dev` on Ubuntu and `brew install google-sparsehash` on Mac OS).

    Then install the dependencies:
    ```sh
    conda create -c conda-forge -n sere python=3.9
    pip install -r requirements.txt
    pip install git+https://github.com/facebookresearch/pytorch3d.git
    pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
    export PYTHONPATH="/path/to/sere:$PYTHONPATH"
    ```

2. Dataset preparation
   Please refer to https://github.com/cvg/nice-slam to download the Replica Dataset. For SJTUers, you could download it from [Replica-JBox](https://jbox.sjtu.edu.cn/l/41duY3).

    Then link the dataset
    ```sh
    ln -s /path/to/Replica data/replica
    ```
## Training Pipeline
Adding `--device x` to specify the GPU device (default: 0).

1. Vanilla NeRF

    ```sh
    python run/main_nerf.py --config configs/nerf/replica/room0.txt
    ```

2. Instant-NGP

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

* [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)
* [inerf](https://github.com/yenchenlin/iNeRF-public)
* [torch-ngp](https://github.com/ashawkey/torch-ngp)
* [NeRFusion](https://github.com/jetd1/NeRFusion)
* [NeuralRecon](https://github.com/zju3dv/NeuralRecon)

## For TA

Our latest report and appendices are in `doc`.
