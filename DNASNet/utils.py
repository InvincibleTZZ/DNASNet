import math
import os
import random
import shutil
from typing import Any, Dict

import numpy as np
import torch
from torch import nn


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_gradient(model: nn.Module, sigma: float) -> None:
    if sigma <= 0:
        return
    for param in model.parameters():
        if param.grad is None:
            continue
        param.grad.add_(torch.randn_like(param.grad) * sigma)


class AvgrageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / max(self.cnt, 1)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        k = min(k, output.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_parameters_in_MB(model: nn.Module) -> float:
    return float(np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6)


def save_checkpoint(state: Dict[str, Any], is_best: bool, save: str) -> None:
    os.makedirs(save, exist_ok=True)
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save, 'model_best.pth.tar'))


def save(model: nn.Module, model_path: str) -> None:
    torch.save(model.state_dict(), model_path)


def load(model: nn.Module, model_path: str) -> None:
    model.load_state_dict(torch.load(model_path, map_location='cpu'))


def drop_path(x: torch.Tensor, drop_prob: float):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = torch.empty(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        x = x.div(keep_prob).mul(mask)
    return x


def create_exp_dir(path: str, scripts_to_save=None) -> None:
    os.makedirs(path, exist_ok=True)
    print(f'Experiment dir : {path}')
    if scripts_to_save:
        script_path = os.path.join(path, 'scripts')
        os.makedirs(script_path, exist_ok=True)
        for script in scripts_to_save:
            shutil.copyfile(script, os.path.join(script_path, os.path.basename(script)))


def save_feature_map(x, dir: str = '') -> None:
    import matplotlib.pyplot as plt

    os.makedirs(dir, exist_ok=True)
    for idx, layer in enumerate(x):
        layer = layer.detach().cpu()
        for batch in range(layer.shape[0]):
            for channel in range(layer.shape[1]):
                fname = f'{idx}_{batch}_{channel}_{layer.shape[-1]}.jpg'
                fp = layer[batch, channel]
                plt.tight_layout()
                plt.axis('off')
                plt.imshow(fp, cmap='inferno')
                plt.savefig(os.path.join(dir, fname), bbox_inches='tight', pad_inches=0)
                plt.close()


def calc_time(seconds: float):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return {'day': d, 'hour': h, 'minute': m, 'second': int(s)}


def save_file(recoder, path: str = './', back_connection: bool = False, split_by_operation: bool = False,
              plots_per_page: int = 27) -> None:
    import matplotlib.pyplot as plt

    if not recoder:
        return

    os.makedirs(path, exist_ok=True)
    keys = sorted(recoder.keys())
    cols = 3
    rows = max(1, math.ceil(plots_per_page / cols))

    for page_begin in range(0, len(keys), plots_per_page):
        page_keys = keys[page_begin:page_begin + plots_per_page]
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = np.atleast_1d(axes).reshape(-1)

        for axis, key in zip(axes, page_keys):
            values = recoder.get(key, [])
            axis.plot(range(len(values)), values)
            axis.set_title(str(key), fontsize=8)
            axis.grid(True, linestyle='--', alpha=0.3)

        for axis in axes[len(page_keys):]:
            axis.axis('off')

        fig.tight_layout()
        fig.savefig(os.path.join(path, f'weights_{page_begin // plots_per_page:03d}.png'))
        plt.close(fig)
