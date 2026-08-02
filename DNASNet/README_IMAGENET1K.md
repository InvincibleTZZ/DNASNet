# DNASNet ImageNet-1k Training

This folder contains the code needed to train DNASNet on ImageNet-1k.
It intentionally excludes checkpoints, logs, caches, datasets, and paper files.

## Environment

Tested environment:

- Python 3.9.25
- CUDA wheel: PyTorch CUDA 11.8
- Recommended NVIDIA driver: compatible with CUDA 11.8 or newer

Create the environment:

```bash
conda create -n dnasnet-imagenet python=3.9 -y
conda activate dnasnet-imagenet
pip install -r requirements.txt
```

If your server uses a different CUDA version, install the matching `torch`
and `torchvision` wheels first, then install the remaining packages from
`requirements.txt`.

## Data Layout

Pass the parent dataset directory with `--data-dir`. The loader will search for
ImageNet-1k using any of these layouts:

```text
/path/to/data/imagenet-1k/train/n01440764/*.JPEG
/path/to/data/imagenet-1k/val/n01440764/*.JPEG

/path/to/data/ILSVRC2012/train/n01440764/*.JPEG
/path/to/data/ILSVRC2012/val/n01440764/*.JPEG

/path/to/data/ILSVRC2012/ILSVRC/Data/CLS-LOC/train/n01440764/*.JPEG
/path/to/data/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/n01440764/*.JPEG
```

For the Kaggle ImageNet layout, prepare validation class folders first:

```bash
python tools/prepare_imagenet_kaggle_layout.py --root /path/to/data/ILSVRC2012
```

Optional tar-shard format:

```bash
python tools/imagenet_to_tar_shards.py \
  --train-dir /path/to/data/ILSVRC2012/train \
  --val-dir /path/to/data/ILSVRC2012/val \
  --output-dir /path/to/data/imagenet-1k-tar-shards \
  --shard-size 2048
```

When tar shards exist under `/path/to/data/imagenet-1k-tar-shards`, the loader
uses them automatically.

## Single-GPU Training

```bash
cd DNASNet
python DNASNet_train.py \
  --dataset imagenet-1k \
  --data-dir /path/to/data \
  --batch-size 64 \
  --workers 8 \
  --prefetch-factor 4 \
  --epochs 300 \
  --step 4 \
  --layers 16 \
  --arch cifar_final \
  --amp \
  --output ./output/imagenet1k
```

`--dataset imagenet-1k` automatically selects:

- `NetworkImageNet`
- 1000 classes
- 224x224 input size
- 56 initial channels, unless `--init-channels` is explicitly set

## Multi-GPU Training

The batch size is per process/GPU. The learning rate is linearly scaled by
`batch_size * world_size / 1024` unless `--no-scale-lr-by-batch` is used.

```bash
cd DNASNet
torchrun --nproc_per_node=8 DNASNet_train.py \
  --dataset imagenet-1k \
  --data-dir /path/to/data \
  --batch-size 64 \
  --workers 8 \
  --prefetch-factor 4 \
  --epochs 300 \
  --step 4 \
  --layers 16 \
  --arch cifar_final \
  --amp \
  --output ./output/imagenet1k
```

## Resume or Evaluate

```bash
python DNASNet_train.py \
  --dataset imagenet-1k \
  --data-dir /path/to/data \
  --resume ./output/imagenet1k/train/EXP_DIR/last.pth.tar \
  --amp
```

```bash
python DNASNet_train.py \
  --dataset imagenet-1k \
  --data-dir /path/to/data \
  --resume ./output/imagenet1k/train/EXP_DIR/model_best.pth.tar \
  --eval \
  --amp
```

