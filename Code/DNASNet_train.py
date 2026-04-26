import argparse
import time
import os
import yaml
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import resolve_data_config
from timm.models import create_model, resume_checkpoint
from timm.utils import CheckpointSaver, AverageMeter, accuracy, reduce_tensor, setup_default_logging, get_outdir, ModelEma, distribute_bn, NativeScaler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import DNASNet_model  # noqa: F401
from braincog.base.node.node import *
from braincog.base.utils.criterions import *
from datasets import *
from utils import random_gradient, save_feature_map, setup_seed
import genotypes

os.environ.setdefault('TORCH_HOME', './data')
os.environ.setdefault('TORCH_DATA_ROOT', './data')

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE')

parser = argparse.ArgumentParser(description='DNASNet Training')
parser.add_argument('--dataset', '--datasets', dest='dataset', default='cifar10', type=str)
parser.add_argument('--model', default='NetworkCIFAR', type=str)
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--num-classes', type=int, default=None)
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('--opt', default='adamw', type=str)
parser.add_argument('--opt-eps', default=1e-8, type=float)
parser.add_argument('--opt-betas', default=None, type=float, nargs='+')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0.01)
parser.add_argument('--clip-grad', type=float, default=None)
parser.add_argument('--sched', default='cosine', type=str)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--warmup-lr', type=float, default=1e-6)
parser.add_argument('--min-lr', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--warmup-epochs', type=int, default=5)
parser.add_argument('--cooldown-epochs', type=int, default=10)
parser.add_argument('--smoothing', type=float, default=0.1)
parser.add_argument('--drop', type=float, default=0.0)
parser.add_argument('--drop-path', type=float, default=0.1)
parser.add_argument('--model-ema', action='store_true', default=False)
parser.add_argument('--model-ema-decay', type=float, default=0.99996)
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log-interval', type=int, default=50)
parser.add_argument('-j', '--workers', type=int, default=4)
parser.add_argument('--num-gpu', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--amp', action='store_true', default=False)
parser.add_argument('--output', default='./output/', type=str)
parser.add_argument('--eval-metric', default='top1', type=str)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--step', type=int, default=4)
parser.add_argument('--encode', type=str, default='direct')
parser.add_argument('--node-type', type=str, default='PLIFNode')
parser.add_argument('--act-fun', type=str, default='QGateGrad')
parser.add_argument('--threshold', type=float, default=.5)
parser.add_argument('--tau', type=float, default=2.)
parser.add_argument('--requires-thres-grad', action='store_true')
parser.add_argument('--sigmoid-thres', action='store_true')
parser.add_argument('--loss-fn', type=str, default='ce')
parser.add_argument('--noisy-grad', type=float, default=0.)
parser.add_argument('--spike-output', action='store_true', default=False)
parser.add_argument('--n_groups', type=int, default=1)
parser.add_argument('--layer-by-layer', action='store_true')
parser.add_argument('--temporal-flatten', action='store_true')

parser.add_argument('--mix-up', action='store_true')
parser.add_argument('--cut-mix', action='store_true')
parser.add_argument('--event-mix', action='store_true')
parser.add_argument('--cutmix_beta', type=float, default=1.0)
parser.add_argument('--cutmix_prob', type=float, default=0.5)
parser.add_argument('--cutmix_num', type=int, default=1)
parser.add_argument('--cutmix_noise', type=float, default=0.)
parser.add_argument('--rand-aug', action='store_true')
parser.add_argument('--randaug_n', type=int, default=3)
parser.add_argument('--randaug_m', type=int, default=15)
parser.add_argument('--train-portion', type=float, default=0.9)
parser.add_argument('--event-size', '--image-size', dest='event_size', default=32, type=int)
parser.add_argument('--data-dir', '--data', dest='data_dir', default=None, type=str)

parser.add_argument('--init-channels', type=int, default=36)
parser.add_argument('--layers', type=int, default=16)
parser.add_argument('--auxiliary', action='store_true', default=False)
parser.add_argument('--arch', default='cifar_final', type=str)
parser.add_argument('--parse_method', default='darts', type=str)
parser.add_argument('--drop_path_prob', type=float, default=0.2)
parser.add_argument('--back-connection', action='store_true', default=True)
parser.add_argument('--k-bilinear', type=float, default=0.1)
parser.add_argument('--use-bilinear', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True, nargs='?', const=True)
parser.add_argument('--suffix', type=str, default='')

DATASET_ALIASES = {
    'cifar-10': 'cifar10',
    'cifar-100': 'cifar100',
    'tinyimagenet': 'TinyImageNet',
    'tiny-imagenet': 'TinyImageNet',
    'tinyimagenet-200': 'TinyImageNet',
    'imagenet': 'imagenet-1k',
    'imagenet1k': 'imagenet-1k',
    'imagenet_1k': 'imagenet-1k',
    'imagenet-1k': 'imagenet-1k',
    'imnet': 'imagenet-1k',
    'mini-imagenet': 'imagenet-mini',
    'imagenetmini': 'imagenet-mini',
    'imagenet_mini': 'imagenet-mini',
    'imagenet-mini': 'imagenet-mini',
}

DATASET_LOADERS = {
    'cifar10': get_cifar10_data,
    'cifar100': get_cifar100_data,
    'TinyImageNet': get_TinyImageNet_data,
    'imagenet-1k': get_imagenet_1k_data,
    'imagenet-mini': get_imagenet_mini_data,
    'dvsg': get_dvsg_data,
    'dvsc10': get_dvsc10_data,
}

DATASET_NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'TinyImageNet': 200,
    'imagenet-1k': 1000,
    'imagenet-mini': 100,
    'dvsg': 11,
    'dvsc10': 10,
}

IMAGENET_DATASETS = {'imagenet-1k', 'imagenet-mini'}
LARGE_IMAGE_DATASETS = {'TinyImageNet', 'imagenet-1k', 'imagenet-mini'}


def normalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.strip()
    return DATASET_ALIASES.get(key.lower(), key)


def infer_num_classes(dataset_name: str, num_classes):
    if num_classes is not None:
        return num_classes
    return DATASET_NUM_CLASSES.get(dataset_name, 10)


def resolve_dataset_loader(dataset_name: str):
    dataset_loader = DATASET_LOADERS.get(dataset_name)
    if dataset_loader is None:
        supported = ', '.join(DATASET_LOADERS.keys())
        raise ValueError(f'Unsupported dataset "{dataset_name}". Supported datasets: {supported}')
    return dataset_loader


def apply_dataset_defaults(args):
    args.dataset = normalize_dataset_name(args.dataset)
    args.num_classes = infer_num_classes(args.dataset, args.num_classes)

    if args.event_size == parser.get_default('event_size') and args.dataset in LARGE_IMAGE_DATASETS:
        args.event_size = 224

    if args.model == parser.get_default('model') and args.dataset in IMAGENET_DATASETS:
        args.model = 'NetworkImageNet'

    if args.data_dir is None:
        args.data_dir = DATA_DIR

    return args

def safe_set_requires_fp(model, flag: bool):
    model_to_set = model.module if hasattr(model, 'module') else model
    if hasattr(model_to_set, 'set_requires_fp'):
        with suppress(Exception):
            model_to_set.set_requires_fp(flag)

def ensure_model_on_device(model, device=torch.device('cuda:0')):
    mod = model.module if hasattr(model, 'module') else model
    try:
        mod.to(device)
        for sub in mod.modules():
            for bname, buf in list(getattr(sub, '_buffers', {}).items()):
                if buf is not None and buf.device != device:
                    sub._buffers[bname] = buf.to(device)
    except Exception:
        pass

def strict_check_model_device(model, device=torch.device('cuda:0')):
    mod = model.module if hasattr(model, 'module') else model
    mismatches = [(name, p.device) for name, p in mod.named_parameters(recurse=True) if p.device != device]
    if mismatches:
        raise RuntimeError(f'Model device mismatch detected (expected {device})')

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    args, args_text = _parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args = apply_dataset_defaults(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    args.no_spike_output = True
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), args.model, args.dataset, args.arch, str(args.step), args.suffix])
        output_dir = get_outdir(output_base, 'train', exp_name)
        args.output_dir = output_dir
        setup_default_logging(log_path=os.path.join(output_dir, 'log.txt'))
    else:
        setup_default_logging()

    args.distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    args.world_size = 1
    args.rank = 0
    if args.distributed:
        args.num_gpu = 1
        args.device = args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    else:
        torch.cuda.set_device(args.device)

    setup_seed(args.seed + args.rank)
    genotype = eval('genotypes.%s' % args.arch)
    k_bilinear_value = args.k_bilinear if args.use_bilinear else 0.0

    model = create_model(
        args.model, pretrained=args.pretrained, num_classes=args.num_classes, dataset=args.dataset,
        step=args.step, encode_type=args.encode, node_type=eval(args.node_type), threshold=args.threshold,
        tau=args.tau, sigmoid_thres=args.sigmoid_thres, requires_thres_grad=args.requires_thres_grad,
        spike_output=not args.no_spike_output, C=args.init_channels, layers=args.layers,
        auxiliary=args.auxiliary, genotype=genotype, parse_method=args.parse_method,
        back_connection=args.back_connection, act_fun=args.act_fun, temporal_flatten=args.temporal_flatten,
        layer_by_layer=args.layer_by_layer, n_groups=args.n_groups, k_bilinear=k_bilinear_value,
        use_sigmoid_transform=True,
    )

    dataset_name_lower = args.dataset.lower()
    args.channels = 2 if 'dvs' in dataset_name_lower else (1 if 'mnist' in dataset_name_lower else 3)
    linear_scaled_lr = args.lr * args.batch_size * args.world_size / 1024.0
    args.lr = linear_scaled_lr

    primary_dev = torch.device(f'cuda:{args.device}')
    model = model.to(primary_dev)
    ensure_model_on_device(model, primary_dev)
    if args.num_gpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu)))
        ensure_model_on_device(model, primary_dev)

    optimizer = create_optimizer(args, model)
    amp_autocast = suppress
    loss_scaler = None
    if args.amp and hasattr(torch.cuda.amp, 'autocast'):
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    resume_epoch = None
    if args.resume:
        args.eval = True
        try:
            resume_epoch = resume_checkpoint(
                model, args.resume, optimizer=None, loss_scaler=None, log_info=args.local_rank == 0)
        except RuntimeError:
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('state_dict_ema', ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt)))
            filtered = {k: v for k, v in state_dict.items() if not ('total_ops' in k or 'total_params' in k)}
            model.load_state_dict(filtered, strict=False)

    safe_set_requires_fp(model, True)

    model_ema = ModelEma(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume=args.resume) if args.model_ema else None
    model_without_ddp = model

    if args.distributed:
        model = NativeDDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
        model_without_ddp = model.module

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = args.start_epoch if args.start_epoch is not None else (resume_epoch if resume_epoch is not None else 0)
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    dataset_loader = resolve_dataset_loader(args.dataset)
    loader_kwargs = dict(
        batch_size=args.batch_size, step=args.step, args=args, data_config=data_config, num_aug_splits=0,
        size=args.event_size, mix_up=args.mix_up, cut_mix=args.cut_mix, event_mix=args.event_mix,
        beta=args.cutmix_beta, prob=args.cutmix_prob, num=args.cutmix_num, noise=args.cutmix_noise,
        num_classes=args.num_classes, rand_aug=args.rand_aug, randaug_n=args.randaug_n, randaug_m=args.randaug_m,
        temporal_flatten=args.temporal_flatten, portion=args.train_portion, _logger=_logger, num_workers=args.workers,
        root=args.data_dir,
    )
    loader_train, loader_eval, mixup_active, mixup_fn = dataset_loader(**loader_kwargs)

    if args.distributed:
        from torch.utils.data.distributed import DistributedSampler
        if not isinstance(getattr(loader_train, 'sampler', None), DistributedSampler):
            loader_train = torch.utils.data.DataLoader(loader_train.dataset, batch_size=args.batch_size, sampler=DistributedSampler(loader_train.dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True), num_workers=args.workers, collate_fn=getattr(loader_train, 'collate_fn', None))
            loader_eval = torch.utils.data.DataLoader(loader_eval.dataset, batch_size=args.batch_size, sampler=DistributedSampler(loader_eval.dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False), num_workers=args.workers, collate_fn=getattr(loader_eval, 'collate_fn', None))

    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda() if args.smoothing else nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    eval_metric = args.eval_metric
    best_metric, best_epoch = None, None

    if args.eval:
        validate(start_epoch, model, loader_eval, validate_loss_fn, args)
        return

    saver = None
    if args.local_rank == 0:
        saver = CheckpointSaver(model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler, checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=(eval_metric == 'loss'))
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, args.epochs):
            if args.distributed and hasattr(getattr(loader_train, 'sampler', None), 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_epoch(epoch, model, loader_train, optimizer, train_loss_fn, args, lr_scheduler, saver, output_dir, amp_autocast, loss_scaler, model_ema, mixup_fn)
            eval_metrics = validate(epoch, model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                ema_eval_metrics = validate(epoch, model_ema.ema, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if saver is not None:
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=eval_metrics[eval_metric])
    except KeyboardInterrupt:
        pass

def train_epoch(epoch, model, loader, optimizer, loss_fn, args, lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress, loss_scaler=None, model_ema=None, mixup_fn=None):
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    losses_m, top1_m = AverageMeter(), AverageMeter()
    model.train()
    strict_check_model_device(model, torch.device(f'cuda:{args.device}'))

    for batch_idx, (inputs, target) in enumerate(loader):
        inputs, target = inputs.type(torch.FloatTensor).cuda(), target.cuda()
        if mixup_fn is not None:
            inputs, target = mixup_fn(inputs, target)
        with amp_autocast():
            output = model(inputs)
            output = output[0] if isinstance(output, (tuple, list)) else output
            loss = loss_fn(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 5)) if not args.mix_up else (torch.tensor([0.]), torch.tensor([0.]))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        if args.distributed:
            loss = reduce_tensor(loss.data, args.world_size)
            acc1 = reduce_tensor(acc1, args.world_size)

        losses_m.update(loss.item(), inputs.size(0))
        top1_m.update(acc1.item() if isinstance(acc1, torch.Tensor) else acc1, inputs.size(0))

        if args.local_rank == 0 and (batch_idx == len(loader) - 1 or batch_idx % args.log_interval == 0):
            _logger.info(f'Train: {epoch} [{batch_idx}/{len(loader)}] Loss: {losses_m.val:.4f} Acc@1: {top1_m.val:.4f}')

    return OrderedDict([('loss', losses_m.avg)])

def validate(epoch, model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    losses_m, top1_m = AverageMeter(), AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
            inputs, target = inputs.type(torch.FloatTensor).cuda(), target.cuda()
            with amp_autocast():
                output = model(inputs)
            output = output[0] if isinstance(output, (tuple, list)) else output
            loss = loss_fn(output, target)
            acc1, _ = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)

            losses_m.update(loss.item(), inputs.size(0))
            top1_m.update(acc1.item() if isinstance(acc1, torch.Tensor) else acc1, output.size(0))

            if args.local_rank == 0 and (batch_idx == len(loader) - 1 or batch_idx % args.log_interval == 0):
                _logger.info(f'Test{log_suffix}: [{batch_idx}/{len(loader)}] Loss: {losses_m.val:.4f} Acc@1: {top1_m.val:.4f}')

    return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg)])

if __name__ == '__main__':
    main()
