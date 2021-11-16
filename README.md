# Contrastive Clustering

A revised version of the official repository that makes it easy to run on your own images.

## Installation

```bash
conda create -n cc python=3.9 cudatoolkit=11.3 cudnn pytorch==1.10.0 torchvision==0.11 -c nvidia -c pytorch -c conda-forge
conda activate cc
pip install scikit-learn munkres numpy opencv-python tqdm
```

## Usage

```bash
python train.py --help
```

```bash
python cluster.py --help
```

## Citation

If you find CC useful in your research, please consider citing:
```
@article{li2020contrastive,
  title={Contrastive Clustering},
  author={Li, Yunfan and Hu, Peng and Liu, Zitao and Peng, Dezhong and Zhou, Joey Tianyi and Peng, Xi},
  journal={arXiv preprint arXiv:2009.09687},
  year={2020}
}
```
