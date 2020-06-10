# Image Restoration Toolkit in PyTorch

## Usage

### Dependencies
[PyTorch](https://pytorch.org/get-started/locally/) >= 1.6
and
```bash
conda install tensorboard h5py scikit-image
```

### Train
```bash
python trainer.py --dataset [dataset&options] --eval_datasets [datasets&options] --model [model&options] --job_dir [dir]
```
e.g.
```bash
python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --scale 2 --model wdsr --num_blocks 16 --job_dir ./wdsr_x2_b16
```
We also support multi-GPU and mixed precision training:
```bash
python -m torch.distributed.launch --nproc_per_node=[\#gpu] trainer.py --amp [options]
```

### Evaluation
```bash
python trainer.py --eval_only --dataset [dataset&options] --eval_datasets [datasets&options] --model [model&options] --job_dir [dir]
# or
python trainer.py --eval_only --dataset [dataset&options] --eval_datasets [datasets&options] --model [model&options] --job_dir X --ckpt [path]
```
e.g.
```bash
python trainer.py --eval_only --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --scale 2 --model wdsr --num_blocks 16 --job_dir ./wdsr_x2_b16
# or
python trainer.py --eval_only --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --scale 2 --model wdsr --num_blocks 16 --job_dir X --ckpt ./wdsr_x2_b16/latest.pth
```

## Datasets
[DIV2K dataset: DIVerse 2K resolution high quality images as used for the NTIRE challenge on super-resolution @ CVPR 2017](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

[Benchmarks (Set5, BSDS100, Urban100)](http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip)

Download and organize data like: 
```bash
pt_ir/data/DIV2K/
├── DIV2K_train_HR
├── DIV2K_train_LR_bicubic
│   └── X2
│   └── X3
│   └── X4
├── DIV2K_valid_HR
└── DIV2K_valid_LR_bicubic
    └── X2
    └── X3
    └── X4
pt_ir/data/Set5/*.png
pt_ir/data/BSDS100/*.png
pt_ir/data/Urban100/*.png
```

## Related repos
[[NSR]](https://github.com/ychfan/nsr) Neural Sparse Representation for Image Restoration

[[SCN]](https://github.com/ychfan/scn) Scale-wise Convolution for Image Restoration

[[WDSR]](https://github.com/ychfan/wdsr) Wide Activation for Efficient Image and Video Super-Resolution
