import os
import inspect
import torch
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as datasets
from torchvision import transforms

import tonic
from tonic import DiskCachedDataset
from timm.data import ImageDataset, create_loader, create_transform

from braincog.datasets.TinyImageNet import TinyImageNet
from cut_mix import CutMix, EventMix, MixUp
from rand_aug import RandAugment
from datasets_utils import dvs_channel_check_expend

DVSCIFAR10_MEAN_16 = [0.3290, 0.4507]
DVSCIFAR10_STD_16 = [1.8398, 1.6549]

DATA_DIR = os.environ.get('DATA_DIR', './data/datasets')

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
CIFAR10_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_DEFAULT_STD = (0.2023, 0.1994, 0.2010)


def _first_existing_dir(candidates):
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def _resolve_image_size(size, data_config=None, default=224):
    if size is not None:
        return int(size)
    if data_config and data_config.get('input_size'):
        return int(data_config['input_size'][-1])
    return int(default)


def _resolve_image_dataset_dirs(root, dataset_roots, eval_splits=('val', 'validation', 'test')):
    base_candidates = [root]
    base_candidates.extend(os.path.join(root, dataset_root) for dataset_root in dataset_roots)

    train_dir = _first_existing_dir(os.path.join(base_dir, 'train') for base_dir in base_candidates)
    eval_dir = _first_existing_dir(os.path.join(base_dir, split) for base_dir in base_candidates for split in eval_splits)

    if train_dir is None or eval_dir is None:
        expected_roots = ', '.join(base_candidates)
        raise FileNotFoundError(
            f'Unable to locate dataset folders under: {expected_roots}. '
            f'Expected train/val-style directories such as train + val, validation, or test.'
        )
    return train_dir, eval_dir


def _build_imagefolder_loaders(batch_size, train_dir, eval_dir, size, num_workers=8, same_da=False):
    train_transform = build_transform(False, size) if same_da else build_transform(True, size)
    eval_transform = build_transform(False, size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    eval_dataset = datasets.ImageFolder(eval_dir, transform=eval_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=False, num_workers=num_workers
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

def build_transform(is_train, img_size):
    resize_im = img_size > 32
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
    if img_size > 32:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    else:
        t.append(transforms.Normalize(CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD))
    return transforms.Compose(t)

def build_dataset(is_train, img_size, dataset, path, same_da=False):
    transform = build_transform(False, img_size) if same_da else build_transform(is_train, img_size)
    os.makedirs(path, exist_ok=True)

    if dataset == 'CIFAR10':
        dataset_obj = datasets.CIFAR10(path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif dataset == 'CIFAR100':
        dataset_obj = datasets.CIFAR100(path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    else:
        raise NotImplementedError

    return dataset_obj, nb_classes

def get_cifar10_data(batch_size, num_workers=8, same_da=False, root=DATA_DIR, **kwargs):
    train_datasets, _ = build_dataset(True, 32, 'CIFAR10', root, same_da)
    test_datasets, _ = build_dataset(False, 32, 'CIFAR10', root, same_da)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size, pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, pin_memory=True, drop_last=False, num_workers=num_workers
    )
    return train_loader, test_loader, None, None

def get_cifar100_data(batch_size, num_workers=8, same_data=False, root=DATA_DIR, *args, **kwargs):
    if 'root' in kwargs:
        root = kwargs['root']
    elif 'data' in kwargs:
        root = kwargs['data']
    elif args and hasattr(args[0], 'data'):
        root = args[0].data

    train_datasets, _ = build_dataset(True, 32, 'CIFAR100', root, same_data)
    test_datasets, _ = build_dataset(False, 32, 'CIFAR100', root, same_data)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size, pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, pin_memory=True, drop_last=False, num_workers=num_workers
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
        train_datasets, batch_size=batch_size, pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, pin_memory=True, drop_last=False, num_workers=num_workers
    )
    return train_loader, test_loader, False, None

def get_tiny_imagenet_data(batch_size, num_workers=8, same_da=False, root=DATA_DIR, *args, **kwargs):
    return get_TinyImageNet_data(batch_size, num_workers=num_workers, same_da=same_da, root=root, *args, **kwargs)

def get_imagenet_1k_data(batch_size, num_workers=8, same_da=False, root=DATA_DIR, *args, **kwargs):
    size = _resolve_image_size(kwargs.get('size', None), kwargs.get('data_config', None), default=224)
    train_dir, eval_dir = _resolve_image_dataset_dirs(
        root,
        dataset_roots=('ILSVRC2012', 'imagenet-1k', 'imagenet1k', 'imagenet'),
    )
    return _build_imagefolder_loaders(
        batch_size=batch_size, train_dir=train_dir, eval_dir=eval_dir, size=size,
        num_workers=num_workers, same_da=same_da,
    )

def get_imagenet_mini_data(batch_size, num_workers=8, same_da=False, root=DATA_DIR, *args, **kwargs):
    size = _resolve_image_size(kwargs.get('size', None), kwargs.get('data_config', None), default=224)
    train_dir, eval_dir = _resolve_image_dataset_dirs(
        root,
        dataset_roots=('imagenet-mini', 'mini-imagenet', 'mini_imagenet', 'imagenet_mini'),
    )
    return _build_imagefolder_loaders(
        batch_size=batch_size, train_dir=train_dir, eval_dir=eval_dir, size=size,
        num_workers=num_workers, same_da=same_da,
    )

def get_imnet_data(args, _logger, data_config, num_aug_splits, root=DATA_DIR, **kwargs):
    return get_imagenet_1k_data(
        batch_size=args.batch_size,
        num_workers=getattr(args, 'workers', 8),
        root=root,
        size=data_config['input_size'][-1] if data_config and data_config.get('input_size') else kwargs.get('size', 224),
        data_config=data_config,
    )

def get_dvsg_data(batch_size, step, root=DATA_DIR, **kwargs):
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    size = kwargs.get('size', 48)
    num_workers = kwargs.get('num_workers', 8)

    train_transform = transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])
    test_transform = transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])

    train_dataset = tonic.datasets.DVSGesture(os.path.join(root, 'DVS/DVSGesture'), transform=train_transform, train=True)
    test_dataset = tonic.datasets.DVSGesture(os.path.join(root, 'DVS/DVSGesture'), transform=test_transform, train=False)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        lambda x: dvs_channel_check_expend(x),
        transforms.RandomCrop(size, padding=size // 12),
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        lambda x: dvs_channel_check_expend(x),
    ])

    if kwargs.get('rand_aug', False):
        train_transform.transforms.insert(2, RandAugment(m=kwargs.get('randaug_m', 15), n=kwargs.get('randaug_n', 3)))

    train_dataset = DiskCachedDataset(train_dataset, cache_path=os.path.join(root, f'DVS/DVSGesture/train_cache_{step}'), transform=train_transform, num_copies=3)
    test_dataset = DiskCachedDataset(test_dataset, cache_path=os.path.join(root, f'DVS/DVSGesture/test_cache_{step}'), transform=test_transform, num_copies=3)

    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mixup_active = cut_mix | event_mix | mix_up

    if cut_mix:
        train_dataset = CutMix(train_dataset, beta=beta, prob=prob, num_mix=num, num_class=num_classes, noise=noise)
    if event_mix:
        train_dataset = EventMix(train_dataset, beta=beta, prob=prob, num_mix=num, num_class=num_classes, noise=noise, gaussian_n=gaussian_n)
    if mix_up:
        train_dataset = MixUp(train_dataset, beta=beta, prob=prob, num_mix=num, num_class=num_classes, noise=noise)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, drop_last=False, num_workers=max(1, num_workers // 4), shuffle=False)

    return train_loader, test_loader, mixup_active, None

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

    if dist_ready:
        if rank == 0:
            train_dataset = _create_cifar10dvs(dataset_root, train_transform, download=True)
            test_dataset = _create_cifar10dvs(dataset_root, test_transform, download=True)
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            train_dataset = _create_cifar10dvs(dataset_root, train_transform, download=False)
            test_dataset = _create_cifar10dvs(dataset_root, test_transform, download=False)
    else:
        train_dataset = _create_cifar10dvs(dataset_root, train_transform, download=True)
        test_dataset = _create_cifar10dvs(dataset_root, test_transform, download=True)

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

    train_dataset = DiskCachedDataset(train_dataset, cache_path=os.path.join(root, f'DVS/DVS_Cifar10/train_cache_{step}'), transform=train_transform)
    test_dataset = DiskCachedDataset(test_dataset, cache_path=os.path.join(root, f'DVS/DVS_Cifar10/test_cache_{step}'), transform=test_transform)

    num_train = len(train_dataset)
    portion = kwargs.get('portion', .9)
    seed = getattr(args, 'seed', 42)
    g = torch.Generator().manual_seed(int(seed))

    if labels is not None:
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=max(1, num_workers // 4))

    return train_loader, test_loader, mixup_active, None
