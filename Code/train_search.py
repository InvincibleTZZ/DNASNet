"""
架构搜索训练脚本（支持 Pure STDP / Hybrid / Bilevel 三种模式）

"""

import os
import sys
import time
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset

from torch.autograd import Variable
from model_search import NetworkWithSTDP, calc_weight, calc_loss
from separate_loss import ConvSeparateLoss, TriSeparateLoss
import utils

from timm.loss import LabelSmoothingCrossEntropy
from datasets import build_transform

parser = argparse.ArgumentParser("SNN Architecture Search")
parser.add_argument('--data', type=str, default='/data/datasets',
                    help='数据集路径')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='数据集: cifar10 | cifar100')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.005,
                    help='权重学习率')
parser.add_argument('--learning_rate_min', type=float, default=0.001,
                    help='最小学习率')
parser.add_argument('--weight_decay', type=float, default=3e-4,
                    help='权重衰减')
parser.add_argument('--report_freq', type=int, default=50,
                    help='日志打印频率')
parser.add_argument('--aux_loss_weight', type=float, default=10.0,
                    help='辅助损失权重')
parser.add_argument('--device', type=int, default=0, help='GPU设备ID')
parser.add_argument('--epochs', type=int, default=50, help='搜索轮数')
parser.add_argument('--init-channels', type=int, default=16,
                    help='初始通道数')
parser.add_argument('--layers', type=int, default=6, help='网络层数')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--grad_clip', type=float, default=5,
                    help='梯度裁剪')
parser.add_argument('--train_portion', type=float, default=0.5,
                    help='训练集比例（剩余为验证集）')
parser.add_argument('--img_size', default=32, type=int)
parser.add_argument('--step', default=8, type=int, help='SNN仿真步长')
parser.add_argument('--node-type', default='BiasPLIFNode', type=str,
                    help='神经元类型')
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--back-connection', action='store_true', default=False)
parser.add_argument('--save', type=str, default='EXP', help='实验名称')
parser.add_argument('--suffix', default='', type=str)

# STDP 相关参数
parser.add_argument('--stdp-type', type=str, default='full',
                    choices=['full', 'hybrid'],
                    help='STDP类型: full=纯STDP, hybrid=混合(STDP+梯度)')
parser.add_argument('--stdp-update-freq', type=int, default=10,
                    help='STDP更新频率（每N个batch更新一次架构参数）')
parser.add_argument('--arch_learning_rate', type=float, default=1e-3,
                    help='架构参数学习率（仅hybrid模式使用）')

args = parser.parse_args()

args.save = './logs/search-{}-{}-{}-{}'.format(
    args.stdp_type, args.save, time.strftime("%Y%m%d-%H%M%S"), args.suffix)
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.device)
    logging.info("args = %s", args)

    # 记录初始显存
    torch.cuda.reset_peak_memory_stats(args.device)

    # ========== 损失函数 ==========
    criterion_train = LabelSmoothingCrossEntropy().cuda()
    criterion_val = nn.CrossEntropyLoss().cuda()
    criterion_train = ConvSeparateLoss(criterion_train, weight=args.aux_loss_weight)

    # ========== 数据集 ==========
    num_classes = 100 if args.dataset == 'cifar100' else 10
    args.num_classes = num_classes

    train_transform = build_transform(True, args.img_size)
    valid_transform = build_transform(False, args.img_size)

    if args.dataset == 'cifar100':
        train_data = dset.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True, num_workers=4)

    # ========== 模型 ==========
    model = NetworkWithSTDP(
        args.init_channels, args.num_classes, args.layers, criterion_train,
        stem_multiplier=3,
        parse_method='bio_darts',
        step=args.step,
        node_type=args.node_type,
        use_stdp=True,
        stdp_type=args.stdp_type,
        dataset=args.dataset,
        spike_output=False,
        back_connection=args.back_connection,
    )
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info("stdp_type = %s", args.stdp_type)

    model_optimizer = torch.optim.AdamW(
        model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    # Hybrid 模式需要验证集迭代器
    valid_iter = None
    if args.stdp_type == 'hybrid':
        valid_iter = iter(valid_queue)

    run_start = time.time()

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        logging.info(calc_weight(model.alphas_normal))
        model.update_history()

        # 训练 + 架构搜索
        train_acc, train_obj = train(
            epoch, train_queue, valid_queue, valid_iter,
            model, criterion_train, criterion_val, model_optimizer)
        logging.info('train_acc %f', train_acc)

        # 验证
        valid_acc, valid_obj = infer(valid_queue, model, criterion_val)
        logging.info('valid_acc %f, valid_loss %f', valid_acc, valid_obj)

        # 记录显存峰值
        peak_mem = torch.cuda.max_memory_allocated(args.device) / (1024 ** 3)
        logging.info('peak_gpu_memory %.2f GB', peak_mem)

        # 保存检查点
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'dur_time': time.time() - run_start,
            'scheduler': scheduler.state_dict(),
            'model_optimizer': model_optimizer.state_dict(),
            'network_states': model.states(),
        }, is_best=False, save=args.save)

        # 保存操作权重历史图
        utils.save_file(recoder=model.alphas_normal_history,
                        path=os.path.join(args.save, 'normal'),
                        back_connection=args.back_connection)

    # 保存最终权重
    np.save(os.path.join(args.save, 'normal_weight.npy'),
            calc_weight(model.alphas_normal).data.cpu().numpy())

    peak_mem_final = torch.cuda.max_memory_allocated(args.device) / (1024 ** 3)
    logging.info('Final peak GPU memory: %.2f GB', peak_mem_final)
    logging.info('search done, total time: %s',
                 utils.calc_time(time.time() - run_start))


def train(epoch, train_queue, valid_queue, valid_iter,
          model, criterion, criterion_val, model_optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step_idx, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # ============================================================
        # Step 1: STDP 更新架构参数（每 stdp_update_freq 步）
        # ============================================================
        if step_idx % args.stdp_update_freq == 0:
            model.eval()
            with torch.no_grad():
                _ = model(input, record_spikes=True)
            model.update_arch_with_stdp()
            model.train()

        # ============================================================
        # Step 2: Hybrid 模式额外计算架构梯度
        # ============================================================
        if args.stdp_type == 'hybrid' and model.stdp_optimizer is not None:
            try:
                val_input, val_target = next(valid_iter)
            except StopIteration:
                valid_iter = iter(valid_queue)
                val_input, val_target = next(valid_iter)
            val_input = val_input.cuda(non_blocking=True)
            val_target = val_target.cuda(non_blocking=True)

            # 需要让 alphas 参与梯度计算
            model.alphas_normal.requires_grad_(True)
            weights = calc_weight(model.alphas_normal)
            val_logits = model(val_input, record_spikes=False)
            val_loss = criterion_val(val_logits, val_target)
            arch_grad = torch.autograd.grad(val_loss, model.alphas_normal,
                                            retain_graph=False)[0]
            model.stdp_optimizer.record_gradient(arch_grad)
            model.alphas_normal.requires_grad_(False)

        # ============================================================
        # Step 3: STBP 更新网络权重
        # ============================================================
        model_optimizer.zero_grad()
        logits = model(input, record_spikes=False)
        aux_input = torch.cat([calc_loss(model.alphas_normal)], dim=0)
        loss, _, _ = criterion(logits, target, aux_input)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        model_optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step_idx % args.report_freq == 0:
            logging.info('train %03d loss: %e top1: %f top5: %f',
                         step_idx, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step_idx, (input, target) in enumerate(valid_queue):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            logits = model(input, record_spikes=False)
            loss = criterion(logits, target)
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step_idx % args.report_freq == 0:
                logging.info('valid %03d %e %f %f',
                             step_idx, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
