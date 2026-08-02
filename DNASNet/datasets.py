import logging
import os
import inspect
import io
import json
import math
import multiprocessing as mp
import random
import re
import tarfile
import torch
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as datasets
from torchvision import transforms
from PIL import Image

import tonic
from tonic import DiskCachedDataset
from timm.data import ImageDataset, create_loader, create_transform

from braincog.datasets.TinyImageNet import TinyImageNet
from cut_mix import CutMix, EventMix, MixUp
from rand_aug import RandAugment
from datasets_utils import dvs_channel_check_expend

DVSCIFAR10_MEAN_16 = [0.3290, 0.4507]
DVSCIFAR10_STD_16 = [1.8398, 1.6549]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(PROJECT_ROOT, 'data', 'datasets'))

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
CIFAR10_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_DEFAULT_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_DEFAULT_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_DEFAULT_STD = (0.2675, 0.2565, 0.2761)
SHD_SENSOR_SIZE = (700, 1, 1)
DVSGESTURE_TRAIN_URL = 'https://zenodo.org/records/8060604/files/ibmGestureTrain.tar.gz?download=1'
DVSGESTURE_TEST_URL = 'https://zenodo.org/records/8060604/files/ibmGestureTest.tar.gz?download=1'

_logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def _patch_dvsgesture_download_urls():
    # The figshare links used by older tonic releases can return an empty file.
    tonic.datasets.DVSGesture.train_url = DVSGESTURE_TRAIN_URL
    tonic.datasets.DVSGesture.test_url = DVSGESTURE_TEST_URL


def _first_existing_dir(candidates):
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def _unique_paths(paths):
    ordered_paths = []
    seen = set()
    for path in paths:
        if not path:
            continue
        normalized = os.path.normpath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered_paths.append(normalized)
    return ordered_paths


def _expand_dataset_roots(root):
    normalized_root = os.path.normpath(root)
    root_candidates = [normalized_root]

    if os.path.basename(normalized_root).lower() == 'datasets':
        root_candidates.append(os.path.dirname(normalized_root))
    else:
        root_candidates.append(os.path.join(normalized_root, 'datasets'))

    for env_name in ('DATA_DIR', 'TORCH_DATA_ROOT', 'TORCH_HOME'):
        env_root = os.environ.get(env_name)
        if not env_root:
            continue
        env_root = os.path.normpath(env_root)
        root_candidates.append(env_root)
        if os.path.basename(env_root).lower() == 'datasets':
            root_candidates.append(os.path.dirname(env_root))
        else:
            root_candidates.append(os.path.join(env_root, 'datasets'))

    return _unique_paths(root_candidates)


def _resolve_image_size(size, data_config=None, default=224):
    if size is not None:
        return int(size)
    if data_config and data_config.get('input_size'):
        return int(data_config['input_size'][-1])
    return int(default)


def build_loader_kwargs(num_workers, pin_memory=True, prefetch_factor=4):
    num_workers = int(num_workers)
    loader_kwargs = dict(pin_memory=pin_memory, num_workers=num_workers)
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = int(prefetch_factor)
    return loader_kwargs


def _dist_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _load_tar_shard_metadata(shard_root):
    metadata_path = os.path.join(shard_root, 'metadata.json')
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, 'r') as f:
        return json.load(f)


def _is_image_member(member_name):
    return member_name.lower().endswith(IMAGE_EXTENSIONS)


def _label_from_tar_member(member_name):
    stem = os.path.splitext(os.path.basename(member_name))[0]
    return int(stem.rsplit('_', 1)[1])


def _decode_image_bytes(data):
    with Image.open(io.BytesIO(data)) as img:
        return img.convert('RGB')


class ImageNetTarShardIterable(torch.utils.data.IterableDataset):
    def __init__(self, shard_root, split, transform=None, shuffle=True, seed=42, num_workers=8):
        super().__init__()
        self.shard_root = shard_root
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.seed = int(seed)
        self.num_workers = max(1, int(num_workers))
        self.epoch = 0
        self._shared_epoch = mp.Value('i', 0)

        metadata = _load_tar_shard_metadata(shard_root)
        if metadata is None or split not in metadata.get('splits', {}):
            raise FileNotFoundError(f'Invalid ImageNet tar-shard metadata under "{shard_root}"')
        self.metadata = metadata
        self.shards = metadata['splits'][split]['shards']
        self.full_shard_size = int(metadata['splits'][split].get('full_shard_size', metadata.get('shard_size', 0)))
        if self.full_shard_size <= 0:
            self.full_shard_size = max(int(shard['samples']) for shard in self.shards)
        self.full_shards = [
            shard for shard in self.shards
            if int(shard.get('samples', 0)) == self.full_shard_size
        ]
        if not self.full_shards:
            raise RuntimeError(f'No full tar shards found for split "{split}" in "{shard_root}"')

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = self.epoch

    def _current_epoch(self):
        return int(self._shared_epoch.value)

    def _effective_shard_count(self):
        _, world_size = _dist_info()
        return (len(self.full_shards) // max(1, world_size)) * max(1, world_size)

    def __len__(self):
        _, world_size = _dist_info()
        effective_shards = self._effective_shard_count()
        return effective_shards * self.full_shard_size // max(1, world_size)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        workers_per_rank = worker_info.num_workers if worker_info is not None else 1
        rank, world_size = _dist_info()

        shard_ids = list(range(len(self.full_shards)))
        rng = random.Random(self.seed + self._current_epoch())
        if self.shuffle:
            rng.shuffle(shard_ids)
        effective_count = (len(shard_ids) // max(1, world_size)) * max(1, world_size)
        shard_ids = shard_ids[:effective_count]
        rank_shard_ids = shard_ids[rank::max(1, world_size)]

        for shard_id in rank_shard_ids[worker_id::workers_per_rank]:
            shard = self.full_shards[shard_id]
            shard_path = os.path.join(self.shard_root, shard['name'])
            with tarfile.open(shard_path, 'r') as tar:
                for member in tar:
                    if not member.isfile() or not _is_image_member(member.name):
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    image = _decode_image_bytes(extracted.read())
                    target = _label_from_tar_member(member.name)
                    if self.transform is not None:
                        image = self.transform(image)
                    yield image, target


class ImageNetTarShardMap(torch.utils.data.Dataset):
    def __init__(self, shard_root, split, transform=None):
        self.shard_root = shard_root
        self.split = split
        self.transform = transform
        metadata = _load_tar_shard_metadata(shard_root)
        if metadata is None or split not in metadata.get('splits', {}):
            raise FileNotFoundError(f'Invalid ImageNet tar-shard metadata under "{shard_root}"')
        self.metadata = metadata
        self.samples = []
        for shard in metadata['splits'][split]['shards']:
            shard_name = shard['name']
            shard_path = os.path.join(shard_root, shard_name)
            with tarfile.open(shard_path, 'r') as tar:
                for member in tar.getmembers():
                    if member.isfile() and _is_image_member(member.name):
                        self.samples.append((shard_name, member.name, _label_from_tar_member(member.name)))
        self._tar_cache = {}

    def __len__(self):
        return len(self.samples)

    def _get_tar(self, shard_name):
        tar = self._tar_cache.get(shard_name)
        if tar is None:
            tar = tarfile.open(os.path.join(self.shard_root, shard_name), 'r')
            self._tar_cache[shard_name] = tar
        return tar

    def __getitem__(self, index):
        shard_name, member_name, target = self.samples[index]
        tar = self._get_tar(shard_name)
        extracted = tar.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f'Missing member "{member_name}" in "{shard_name}"')
        image = _decode_image_bytes(extracted.read())
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _resolve_imagenet_tar_shard_root(root, dataset_roots):
    candidates = []
    for candidate_root in _expand_dataset_roots(root):
        for dataset_root in dataset_roots:
            candidates.extend([
                os.path.join(candidate_root, f'{dataset_root}-tar-shards'),
                os.path.join(candidate_root, f'{dataset_root}_tar_shards'),
                os.path.join(candidate_root, dataset_root, 'tar_shards'),
            ])
    for candidate in _unique_paths(candidates):
        metadata = _load_tar_shard_metadata(candidate)
        if metadata and {'train', 'val'}.issubset(set(metadata.get('splits', {}).keys())):
            return candidate
    return None


def _resolve_image_dataset_dirs(root, dataset_roots, eval_splits=('val', 'validation', 'test')):
    base_candidates = []
    for candidate_root in _expand_dataset_roots(root):
        base_candidates.append(candidate_root)
        for dataset_root in dataset_roots:
            dataset_base = os.path.join(candidate_root, dataset_root)
            base_candidates.extend([
                dataset_base,
                os.path.join(dataset_base, 'Data', 'CLS-LOC'),
                os.path.join(dataset_base, 'ILSVRC', 'Data', 'CLS-LOC'),
            ])
    base_candidates = _unique_paths(base_candidates)

    train_dir = _first_existing_dir(os.path.join(base_dir, 'train') for base_dir in base_candidates)
    eval_dir = _first_existing_dir(os.path.join(base_dir, split) for base_dir in base_candidates for split in eval_splits)

    if train_dir is None or eval_dir is None:
        expected_roots = ', '.join(base_candidates)
        raise FileNotFoundError(
            f'Unable to locate dataset folders under: {expected_roots}. '
            'Pass `--data-dir /path/to/imagenet-root` and make sure the dataset uses one of these layouts: '
            '`train` + `val`, `ILSVRC2012/train` + `ILSVRC2012/val`, or '
            '`ILSVRC2012/ILSVRC/Data/CLS-LOC/train` + `.../val`.'
        )
    return train_dir, eval_dir


def _resolve_imagenet_archive_root(root, dataset_roots):
    required_archives = (
        'ILSVRC2012_devkit_t12.tar.gz',
        'ILSVRC2012_img_train.tar',
        'ILSVRC2012_img_val.tar',
    )
    archive_candidates = []
    for candidate_root in _expand_dataset_roots(root):
        archive_candidates.append(candidate_root)
        archive_candidates.extend(os.path.join(candidate_root, dataset_root) for dataset_root in dataset_roots)

    for archive_root in _unique_paths(archive_candidates):
        if all(os.path.isfile(os.path.join(archive_root, archive_name)) for archive_name in required_archives):
            return archive_root
    return None


def _ensure_imagefolder_layout(split_dir, split_name):
    if any(entry.is_dir() for entry in os.scandir(split_dir)):
        return
    raise FileNotFoundError(
        f'{split_name} directory "{split_dir}" does not contain class subfolders. '
        'This loader uses torchvision ImageFolder, so ImageNet must be arranged like '
        '`train/n01440764/*.JPEG` and `val/n01440764/*.JPEG`.'
    )


def _build_imagefolder_loaders(batch_size, train_dir, eval_dir, size, num_workers=8, same_da=False, prefetch_factor=4):
    _ensure_imagefolder_layout(train_dir, 'Train')
    _ensure_imagefolder_layout(eval_dir, 'Validation')
    train_transform = build_transform(False, size) if same_da else build_transform(True, size)
    eval_transform = build_transform(False, size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    eval_dataset = datasets.ImageFolder(eval_dir, transform=eval_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, shuffle=True,
        **build_loader_kwargs(num_workers, prefetch_factor=prefetch_factor)
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, drop_last=False, shuffle=False,
        **build_loader_kwargs(num_workers, prefetch_factor=prefetch_factor)
    )
    return train_loader, eval_loader, False, None


def _build_imagenet_archive_loaders(batch_size, archive_root, size, num_workers=8, same_da=False, prefetch_factor=4):
    train_transform = build_transform(False, size) if same_da else build_transform(True, size)
    eval_transform = build_transform(False, size)

    train_dataset = datasets.ImageNet(archive_root, split='train', transform=train_transform)
    eval_dataset = datasets.ImageNet(archive_root, split='val', transform=eval_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, shuffle=True,
        **build_loader_kwargs(num_workers, prefetch_factor=prefetch_factor)
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, drop_last=False, shuffle=False,
        **build_loader_kwargs(num_workers, prefetch_factor=prefetch_factor)
    )
    return train_loader, eval_loader, False, None


def _build_imagenet_tar_shard_loaders(
    batch_size,
    shard_root,
    size,
    num_workers=8,
    same_da=False,
    seed=42,
    prefetch_factor=4,
):
    train_transform = build_transform(False, size) if same_da else build_transform(True, size)
    eval_transform = build_transform(False, size)

    train_dataset = ImageNetTarShardMap(shard_root, 'train', transform=train_transform)
    eval_dataset = ImageNetTarShardMap(shard_root, 'val', transform=eval_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, shuffle=True,
        **build_loader_kwargs(num_workers, prefetch_factor=prefetch_factor)
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, drop_last=False, shuffle=False,
        **build_loader_kwargs(num_workers, prefetch_factor=prefetch_factor)
    )
    return train_loader, eval_loader, False, None


def _build_fake_image_loaders(
    batch_size,
    size,
    num_classes,
    num_workers=8,
    same_da=False,
    train_samples=2048,
    eval_samples=512,
):
    train_transform = build_transform(False, size) if same_da else build_transform(True, size)
    eval_transform = build_transform(False, size)

    train_dataset = datasets.FakeData(
        size=train_samples,
        image_size=(3, size, size),
        num_classes=num_classes,
        transform=train_transform,
    )
    eval_dataset = datasets.FakeData(
        size=eval_samples,
        image_size=(3, size, size),
        num_classes=num_classes,
        transform=eval_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, **build_loader_kwargs(num_workers)
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, drop_last=False, shuffle=False, **build_loader_kwargs(num_workers)
    )
    return train_loader, eval_loader, False, None

def unpack_mix_param(args):
    mix_up = args.get('mix_up', False)
    cut_mix = args.get('cut_mix', False)
    event_mix = args.get('event_mix', False)
    beta = args.get('beta', 1.)
    prob = args.get('prob', .5)
    num = args.get('num', 1)
    num_classes = args.get('num_classes', 10)
    noise = args.get('noise', 0.)
    gaussian_n = args.get('gaussian_n', None)
    return mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n

def build_transform(is_train, img_size, mean=None, std=None):
    resize_im = img_size > 32
    mean = mean or IMAGENET_DEFAULT_MEAN
    std = std or IMAGENET_DEFAULT_STD
    if is_train:
        transform = create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(img_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * img_size)
        t.append(transforms.Resize(size, interpolation=3))
        t.append(transforms.CenterCrop(img_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_dataset(is_train, img_size, dataset, path, same_da=False):
    os.makedirs(path, exist_ok=True)

    if dataset == 'CIFAR10':
        mean, std = CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD
        nb_classes = 10
    elif dataset == 'CIFAR100':
        mean, std = CIFAR100_DEFAULT_MEAN, CIFAR100_DEFAULT_STD
        nb_classes = 100
    else:
        raise NotImplementedError

    transform = build_transform(False, img_size, mean=mean, std=std) if same_da else build_transform(
        is_train, img_size, mean=mean, std=std
    )
    if dataset == 'CIFAR10':
        dataset_obj = datasets.CIFAR10(path, train=is_train, transform=transform, download=True)
    else:
        dataset_obj = datasets.CIFAR100(path, train=is_train, transform=transform, download=True)

    return dataset_obj, nb_classes


class SHDToDense:
    """Convert tonic SHD events/frames to a dense [time, 700] tensor."""

    def __init__(self, time_bins=100, clamp=True):
        self.time_bins = int(time_bins)
        self.clamp = clamp
        self.to_frame = tonic.transforms.ToFrame(
            sensor_size=SHD_SENSOR_SIZE,
            n_time_bins=self.time_bins,
        )

    def __call__(self, events):
        if not torch.is_tensor(events):
            if hasattr(events, 'dtype') and getattr(events.dtype, 'names', None):
                events = self.to_frame(events)
            events = torch.as_tensor(events, dtype=torch.float32)
        else:
            events = events.float()

        # tonic SHD frames are usually [T, 1, 700, 1]. The model expects [T, 700].
        if events.dim() == 5:
            if events.size(1) == 1:
                events = events.squeeze(1)
            if events.dim() == 4 and events.size(-1) == 1:
                events = events.squeeze(-1)
            if events.dim() == 5:
                events = events.reshape(events.size(0), -1)
        if events.dim() == 4:
            if events.size(1) == 1:
                events = events.squeeze(1)
            if events.dim() == 3 and events.size(-1) == 1:
                events = events.squeeze(-1)
            if events.dim() == 4:
                events = events.reshape(events.size(0), -1)
        if events.dim() == 3:
            if events.size(1) == 1:
                events = events.squeeze(1)
            elif events.size(-1) == 1:
                events = events.squeeze(-1)
            else:
                events = events.reshape(events.size(0), -1)
        if events.dim() == 1:
            events = events.unsqueeze(0)

        if events.size(-1) != SHD_SENSOR_SIZE[0]:
            if events.numel() % SHD_SENSOR_SIZE[0] != 0:
                raise ValueError(f'Cannot reshape SHD sample with shape {tuple(events.shape)} to [T, 700].')
            events = events.reshape(-1, SHD_SENSOR_SIZE[0])

        if events.size(0) != self.time_bins:
            if events.size(0) > self.time_bins:
                events = events[:self.time_bins]
            else:
                pad = events.new_zeros(self.time_bins - events.size(0), SHD_SENSOR_SIZE[0])
                events = torch.cat([events, pad], dim=0)

        if self.clamp:
            events = events.clamp(0, 1)
        return events


def build_shd_datasets(root=DATA_DIR, step=100, clamp=True):
    transform = SHDToDense(time_bins=step, clamp=clamp)
    dataset_root = os.path.join(root, 'SHD')
    train_dataset = tonic.datasets.SHD(dataset_root, train=True, transform=transform)
    test_dataset = tonic.datasets.SHD(dataset_root, train=False, transform=transform)
    return train_dataset, test_dataset


def get_shd_data(batch_size, step=100, root=DATA_DIR, **kwargs):
    num_workers = kwargs.get('num_workers', 8)
    train_dataset, test_dataset = build_shd_datasets(
        root=root,
        step=step,
        clamp=kwargs.get('shd_clamp', True),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, **build_loader_kwargs(num_workers)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, drop_last=False, shuffle=False,
        **build_loader_kwargs(max(1, num_workers // 4))
    )
    return train_loader, test_loader, False, None

def get_cifar10_data(batch_size, num_workers=8, same_da=False, root=DATA_DIR, **kwargs):
    size = kwargs.get('size', 32)
    train_datasets, _ = build_dataset(True, size, 'CIFAR10', root, same_da)
    test_datasets, _ = build_dataset(False, size, 'CIFAR10', root, same_da)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size, drop_last=True, shuffle=True, **build_loader_kwargs(num_workers)
    )
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, drop_last=False, **build_loader_kwargs(num_workers)
    )
    return train_loader, test_loader, None, None

def get_cifar100_data(batch_size, num_workers=8, same_data=False, root=DATA_DIR, *args, **kwargs):
    if 'root' in kwargs:
        root = kwargs['root']
    elif 'data' in kwargs:
        root = kwargs['data']
    elif args and hasattr(args[0], 'data'):
        root = args[0].data

    size = kwargs.get('size', 32)
    train_datasets, _ = build_dataset(True, size, 'CIFAR100', root, same_data)
    test_datasets, _ = build_dataset(False, size, 'CIFAR100', root, same_data)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size, drop_last=True, shuffle=True, **build_loader_kwargs(num_workers)
    )
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, drop_last=False, **build_loader_kwargs(num_workers)
    )
    return train_loader, test_loader, False, None

def get_TinyImageNet_data(batch_size, num_workers=8, same_da=False, root=DATA_DIR, *args, **kwargs):
    size = kwargs.get("size", 224)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(size * 8 // 7),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    root = os.path.join(root, 'TinyImageNet')
    train_datasets = TinyImageNet(root=root, split="train", transform=test_transform if same_da else train_transform, download=True)
    test_datasets = TinyImageNet(root=root, split="val", transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size, drop_last=True, shuffle=True, **build_loader_kwargs(num_workers)
    )
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, drop_last=False, **build_loader_kwargs(num_workers)
    )
    return train_loader, test_loader, False, None

def get_tiny_imagenet_data(batch_size, num_workers=8, same_da=False, root=DATA_DIR, *args, **kwargs):
    return get_TinyImageNet_data(batch_size, num_workers=num_workers, same_da=same_da, root=root, *args, **kwargs)

def get_imagenet_1k_data(batch_size, num_workers=8, same_da=False, root=DATA_DIR, *args, **kwargs):
    size = _resolve_image_size(kwargs.get('size', None), kwargs.get('data_config', None), default=224)
    prefetch_factor = kwargs.get('prefetch_factor', 4)
    archive_roots = ('ILSVRC2012', 'imagenet-1k', 'imagenet1k', 'imagenet')
    logger = kwargs.get('_logger', _logger)
    shard_root = _resolve_imagenet_tar_shard_root(root, archive_roots)
    if shard_root is not None:
        logger.info('Using ImageNet tar-shard dataset at "%s"', shard_root)
        return _build_imagenet_tar_shard_loaders(
            batch_size=batch_size,
            shard_root=shard_root,
            size=size,
            num_workers=num_workers,
            same_da=same_da,
            seed=getattr(kwargs.get('args', None), 'seed', kwargs.get('seed', 42)),
            prefetch_factor=prefetch_factor,
        )
    try:
        train_dir, eval_dir = _resolve_image_dataset_dirs(
            root,
            dataset_roots=archive_roots,
        )
        return _build_imagefolder_loaders(
            batch_size=batch_size, train_dir=train_dir, eval_dir=eval_dir, size=size,
            num_workers=num_workers, same_da=same_da, prefetch_factor=prefetch_factor,
        )
    except FileNotFoundError as exc:
        archive_root = _resolve_imagenet_archive_root(root, archive_roots)
        if archive_root is not None:
            logger.warning(
                'Found official ImageNet archives in "%s". Preparing extracted train/val folders on first use; '
                'this may take a while.',
                archive_root,
            )
            try:
                return _build_imagenet_archive_loaders(
                    batch_size=batch_size,
                    archive_root=archive_root,
                    size=size,
                    num_workers=num_workers,
                    same_da=same_da,
                    prefetch_factor=prefetch_factor,
                )
            except Exception as archive_exc:
                logger.warning(
                    'Failed to prepare ImageNet from archives in "%s". Falling back to synthetic FakeData. '
                    'Please verify the official archive files are complete. Details: %s',
                    archive_root, archive_exc
                )
        logger.warning(
            'ImageNet-1k dataset not found under "%s". Falling back to synthetic FakeData so training can start. '
            'Provide `--data-dir /path/to/imagenet` with either extracted folders or the official archives '
            '`ILSVRC2012_devkit_t12.tar.gz`, `ILSVRC2012_img_train.tar`, and `ILSVRC2012_img_val.tar`. Details: %s',
            root, exc
        )
        return _build_fake_image_loaders(
            batch_size=batch_size,
            size=size,
            num_classes=1000,
            num_workers=num_workers,
            same_da=same_da,
            train_samples=kwargs.get('fake_train_samples', 2048),
            eval_samples=kwargs.get('fake_eval_samples', 512),
        )

def get_imagenet_mini_data(batch_size, num_workers=8, same_da=False, root=DATA_DIR, *args, **kwargs):
    size = _resolve_image_size(kwargs.get('size', None), kwargs.get('data_config', None), default=224)
    prefetch_factor = kwargs.get('prefetch_factor', 4)
    logger = kwargs.get('_logger', _logger)
    shard_root = _resolve_imagenet_tar_shard_root(
        root, ('imagenet-mini', 'mini-imagenet', 'mini_imagenet', 'imagenet_mini')
    )
    if shard_root is not None:
        logger.info('Using ImageNet-mini tar-shard dataset at "%s"', shard_root)
        return _build_imagenet_tar_shard_loaders(
            batch_size=batch_size,
            shard_root=shard_root,
            size=size,
            num_workers=num_workers,
            same_da=same_da,
            seed=getattr(kwargs.get('args', None), 'seed', kwargs.get('seed', 42)),
            prefetch_factor=prefetch_factor,
        )
    try:
        train_dir, eval_dir = _resolve_image_dataset_dirs(
            root,
            dataset_roots=('imagenet-mini', 'mini-imagenet', 'mini_imagenet', 'imagenet_mini'),
        )
        return _build_imagefolder_loaders(
            batch_size=batch_size, train_dir=train_dir, eval_dir=eval_dir, size=size,
            num_workers=num_workers, same_da=same_da, prefetch_factor=prefetch_factor,
        )
    except FileNotFoundError as exc:
        logger.warning(
            'ImageNet-mini dataset not found under "%s". Falling back to synthetic FakeData so training can start. '
            'Provide `--data-dir /path/to/imagenet-mini` to train on real data. Details: %s',
            root, exc
        )
        return _build_fake_image_loaders(
            batch_size=batch_size,
            size=size,
            num_classes=100,
            num_workers=num_workers,
            same_da=same_da,
            train_samples=kwargs.get('fake_train_samples', 1024),
            eval_samples=kwargs.get('fake_eval_samples', 256),
        )

def get_imnet_data(args, _logger, data_config, num_aug_splits, root=DATA_DIR, **kwargs):
    return get_imagenet_1k_data(
        batch_size=args.batch_size,
        num_workers=getattr(args, 'workers', 8),
        root=root,
        size=data_config['input_size'][-1] if data_config and data_config.get('input_size') else kwargs.get('size', 224),
        data_config=data_config,
    )


class DVSGestureFrameTransform:
    def __init__(self, size=48, train=False):
        self.size = int(size)
        self.train = bool(train)
        self.crop = transforms.RandomCrop(self.size, padding=self.size // 12)

    def __call__(self, frames):
        frames = torch.as_tensor(frames, dtype=torch.float)
        if frames.dim() != 4:
            raise ValueError(f'DVS-Gesture frames should be [T, C, H, W], got {tuple(frames.shape)}')
        if frames.size(1) != 2:
            frames = dvs_channel_check_expend(frames)
        if frames.size(-2) != self.size or frames.size(-1) != self.size:
            frames = F.interpolate(frames, size=[self.size, self.size],
                                   mode='bilinear', align_corners=True)
        if self.train:
            frames = self.crop(frames)
        return frames


def _resolve_spikingjelly_dvsg_root(root):
    candidates = [
        os.path.join(root, 'DVS/DVSGesture/DVSGesture'),
        os.path.join(root, 'DVS/DVSGesture'),
        os.path.join(root, 'DVS128Gesture'),
        root,
    ]
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        if (os.path.isdir(os.path.join(candidate, 'events_np')) or
                os.path.isdir(os.path.join(candidate, 'DvsGesture', 'DvsGesture')) or
                os.path.isdir(os.path.join(candidate, 'extract', 'DvsGesture'))):
            return candidate
    return candidates[0]


def _prepare_spikingjelly_dvsg(root):
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    dataset_root = _resolve_spikingjelly_dvsg_root(root)
    events_np_root = os.path.join(dataset_root, 'events_np')
    if os.path.isdir(events_np_root):
        return dataset_root

    extract_candidates = [
        os.path.join(dataset_root, 'DvsGesture'),
        os.path.join(dataset_root, 'extract'),
    ]
    extract_root = None
    for candidate in extract_candidates:
        if os.path.isfile(os.path.join(candidate, 'DvsGesture', 'trials_to_train.txt')):
            extract_root = candidate
            break
    if extract_root is None:
        raise FileNotFoundError(
            'Unable to locate DVS-Gesture raw files for SpikingJelly. Expected '
            '`<root>/DVS/DVSGesture/DVSGesture/DvsGesture/DvsGesture/trials_to_train.txt` '
            'or an existing `<root>/.../events_np` cache.')

    os.makedirs(events_np_root, exist_ok=False)
    DVS128Gesture.create_events_np_files(extract_root, events_np_root)
    return dataset_root


def _build_spikingjelly_dvsg_dataset(root, train, step, size, train_transform):
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    dataset_root = _prepare_spikingjelly_dvsg(root)
    return DVS128Gesture(
        root=dataset_root,
        train=train,
        data_type='frame',
        frames_number=int(step),
        split_by='number',
        transform=DVSGestureFrameTransform(size=size, train=train_transform),
    )


def get_dvsg_data(batch_size, step, root=DATA_DIR, **kwargs):
    size = kwargs.get('size', 48)
    num_workers = kwargs.get('num_workers', 8)
    train_dataset = _build_spikingjelly_dvsg_dataset(
        root=root, train=True, step=step, size=size, train_transform=True)
    test_dataset = _build_spikingjelly_dvsg_dataset(
        root=root, train=False, step=step, size=size, train_transform=False)

    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mixup_active = cut_mix | event_mix | mix_up

    if cut_mix:
        train_dataset = CutMix(train_dataset, beta=beta, prob=prob, num_mix=num, num_class=num_classes, noise=noise)
    if event_mix:
        train_dataset = EventMix(train_dataset, beta=beta, prob=prob, num_mix=num, num_class=num_classes, noise=noise, gaussian_n=gaussian_n)
    if mix_up:
        train_dataset = MixUp(train_dataset, beta=beta, prob=prob, num_mix=num, num_class=num_classes, noise=noise)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, **build_loader_kwargs(num_workers)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, drop_last=False, shuffle=False,
        **build_loader_kwargs(max(1, num_workers // 4))
    )

    return train_loader, test_loader, mixup_active, None


def _split_indices_from_labels(labels, portion, seed, num_samples=None, num_classes=None):
    generator = torch.Generator().manual_seed(int(seed))
    if labels is None or (num_samples is not None and len(labels) != num_samples):
        total = int(num_samples if num_samples is not None else len(labels))
        indices = torch.randperm(total, generator=generator).tolist()
        split = int(round(total * portion))
        return indices[:split], indices[split:]

    train_indices, valid_indices = [], []
    class_ids = range(num_classes) if num_classes is not None else sorted(set(int(label) for label in labels))
    for class_id in class_ids:
        class_indices = [idx for idx, label in enumerate(labels) if int(label) == int(class_id)]
        if not class_indices:
            continue
        perm = torch.randperm(len(class_indices), generator=generator).tolist()
        split = int(round(len(class_indices) * portion))
        train_indices.extend(class_indices[idx] for idx in perm[:split])
        valid_indices.extend(class_indices[idx] for idx in perm[split:])
    return train_indices, valid_indices


def get_dvsg_search_data(batch_size, step, root=DATA_DIR, **kwargs):
    """
    Fair DVS-Gesture NAS split.

    Only the official training split is partitioned into weight-update and
    architecture-update subsets. The official test split is held out for
    retraining/evaluation and is never returned here.
    """
    size = kwargs.get('size', 48)
    num_workers = kwargs.get('num_workers', 8)
    portion = kwargs.get('portion', 0.5)
    args = kwargs.get('args', None)
    seed = getattr(args, 'seed', kwargs.get('seed', 42))
    num_classes = kwargs.get('num_classes', 11)
    official_train = _build_spikingjelly_dvsg_dataset(
        root=root, train=True, step=step, size=size, train_transform=True)

    labels = list(getattr(official_train, 'targets', []))
    train_indices, arch_indices = _split_indices_from_labels(
        labels if labels else None,
        portion=portion,
        seed=seed,
        num_samples=len(official_train),
        num_classes=num_classes,
    )
    overlap = set(train_indices) & set(arch_indices)
    if overlap:
        raise RuntimeError(f'DVS-Gesture search split overlap detected: {len(overlap)} samples')
    if not train_indices or not arch_indices:
        raise RuntimeError(
            f'Invalid DVS-Gesture search split: train={len(train_indices)}, arch={len(arch_indices)}')

    train_subset = torch.utils.data.Subset(official_train, train_indices)
    arch_source = _build_spikingjelly_dvsg_dataset(
        root=root, train=True, step=step, size=size, train_transform=False)
    arch_dataset = torch.utils.data.Subset(arch_source, arch_indices)

    logger = kwargs.get('_logger', _logger)
    if logger is not None:
        logger.info(
            'dvsg fair search split via SpikingJelly: official_train=%d, '
            'weight_train=%d, arch_val=%d, overlap=0, official_test_held_out=True',
            len(official_train), len(train_indices), len(arch_indices))

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, drop_last=True, shuffle=True, **build_loader_kwargs(num_workers))
    arch_loader = torch.utils.data.DataLoader(
        arch_dataset, batch_size=batch_size, drop_last=False, shuffle=False,
        **build_loader_kwargs(max(1, num_workers // 4)))
    return train_loader, arch_loader, False, None


def get_dvsc10_data(batch_size, step, root=DATA_DIR, **kwargs):
    size = kwargs.get('size', 48)
    num_workers = kwargs.get('num_workers', 8)
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    
    train_transform = transforms.Compose([tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step)])
    test_transform = transforms.Compose([tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step)])
    
    def _create_cifar10dvs(dataset_root, transform, download=None):
        try:
            sig = inspect.signature(tonic.datasets.CIFAR10DVS)
            if 'download' in sig.parameters:
                return tonic.datasets.CIFAR10DVS(dataset_root, transform=transform, download=download) if download is None else tonic.datasets.CIFAR10DVS(dataset_root, transform=transform, download=download)
        except Exception:
            pass
        return tonic.datasets.CIFAR10DVS(dataset_root, transform=transform)

    args = kwargs.get('args', None)
    dataset_root = os.path.join(root, 'DVS/DVS_Cifar10')
    distributed = bool(getattr(args, 'distributed', False))
    dist_ready = distributed and torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = getattr(args, 'rank', 0)

    dataset_exists = os.path.isdir(dataset_root) and any(
        os.path.isdir(os.path.join(dataset_root, cls_name))
        for cls_name in ('airplane', 'automobile', 'bird', 'cat', 'deer',
                         'dog', 'frog', 'horse', 'ship', 'truck', 'CIFAR10DVS')
    )

    if dist_ready:
        if rank == 0:
            train_dataset = _create_cifar10dvs(dataset_root, train_transform, download=not dataset_exists)
            test_dataset = _create_cifar10dvs(dataset_root, test_transform, download=not dataset_exists)
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            train_dataset = _create_cifar10dvs(dataset_root, train_transform, download=False)
            test_dataset = _create_cifar10dvs(dataset_root, test_transform, download=False)
    else:
        train_dataset = _create_cifar10dvs(dataset_root, train_transform, download=not dataset_exists)
        test_dataset = _create_cifar10dvs(dataset_root, test_transform, download=not dataset_exists)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        transforms.RandomCrop(size, padding=size // 12),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])

    if kwargs.get('rand_aug', False):
        train_transform.transforms.insert(2, RandAugment(m=kwargs.get('randaug_m', 15), n=kwargs.get('randaug_n', 3)))

    labels = next((list(getattr(train_dataset, attr)) for attr in ('targets', 'labels') if getattr(train_dataset, attr, None) is not None), None)
    data_paths = list(getattr(train_dataset, 'data', []))

    train_dataset = DiskCachedDataset(train_dataset, cache_path=os.path.join(root, f'DVS/DVS_Cifar10/train_cache_{step}'), transform=train_transform)
    test_dataset = DiskCachedDataset(test_dataset, cache_path=os.path.join(root, f'DVS/DVS_Cifar10/test_cache_{step}'), transform=test_transform)

    num_train = len(train_dataset)
    portion = kwargs.get('portion', .9)
    split_mode = kwargs.get('split_mode', getattr(args, 'dvsc10_split_mode', 'random'))
    seed = getattr(args, 'seed', 42)
    g = torch.Generator().manual_seed(int(seed))

    if split_mode == 'tet':
        split_point = int(round(1000 * portion))
        if not data_paths or len(data_paths) != num_train:
            raise RuntimeError('TET split mode requires CIFAR10DVS file paths for deterministic train/test partition.')
        indices_train, indices_test = [], []
        for idx, sample_path in enumerate(data_paths):
            match = re.search(r'_(\d+)\.aedat4$', os.path.basename(sample_path))
            if match is None:
                raise RuntimeError(f'Unable to parse CIFAR10DVS sample id from path: {sample_path}')
            sample_id = int(match.group(1))
            if sample_id < split_point:
                indices_train.append(idx)
            else:
                indices_test.append(idx)
    elif labels is not None:
        num_classes = kwargs.get('num_classes', 10)
        indices_train, indices_test = [], []
        for c in range(num_classes):
            cls_indices = [i for i, y in enumerate(labels) if int(y) == c]
            if not cls_indices: continue
            perm = torch.randperm(len(cls_indices), generator=g).tolist()
            split = int(round(len(cls_indices) * portion))
            indices_train.extend([cls_indices[i] for i in perm[:split]])
            indices_test.extend([cls_indices[i] for i in perm[split:]])
    else:
        all_indices = torch.randperm(num_train, generator=g).tolist()
        split = int(round(num_train * portion))
        indices_train, indices_test = all_indices[:split], all_indices[split:]

    if set(indices_train) & set(indices_test):
        raise RuntimeError("Train/Test split overlap detected")

    logger = kwargs.get('_logger', _logger)
    if logger is not None:
        logger.info(f'dvsc10 split_mode={split_mode}, train={len(indices_train)}, test={len(indices_test)}')

    train_dataset = torch.utils.data.Subset(train_dataset, indices_train)
    test_dataset = torch.utils.data.Subset(test_dataset, indices_test)

    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mix_indices = list(range(len(train_dataset)))
    mixup_active = (cut_mix > 0.0) | (event_mix > 0.0) | (mix_up > 0.0)

    if cut_mix:
        train_dataset = CutMix(train_dataset, beta=beta, prob=prob, num_mix=num, num_class=num_classes, indices=mix_indices, noise=noise)
    if event_mix:
        train_dataset = EventMix(train_dataset, beta=beta, prob=prob, num_mix=num, num_class=num_classes, indices=mix_indices, noise=noise, gaussian_n=gaussian_n)
    if mix_up:
        train_dataset = MixUp(train_dataset, beta=beta, prob=prob, num_mix=num, num_class=num_classes, indices=mix_indices, noise=noise)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **build_loader_kwargs(num_workers))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **build_loader_kwargs(max(1, num_workers // 4)))

    return train_loader, test_loader, mixup_active, None
