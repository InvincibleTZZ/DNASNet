from functools import partial
from typing import List, Type
import torch
import torch.nn.functional as F

from DNASNet_operations import *
from genotypes import Genotype
from utils import drop_path
from timm.models import register_model
from braincog.base.node import *
from braincog.base.connection.layer import *
from DNASNet_model_zoo_base_module import BaseModule


class BilinearClusterPotential(torch.autograd.Function):
    """
    内存优化的双线性簇膜电位计算
    使用自定义autograd函数，在backward中只计算必要的梯度，减少内存占用
    """
    @staticmethod
    def forward(ctx, neuron1, neuron2, k_bilinear):
        # 前向传播：计算簇膜电位
        # cluster_potential = neuron1 + neuron2 + k_bilinear * neuron1 * neuron2
        # 为了减少内存，我们保存最少的中间结果
        ctx.save_for_backward(neuron1, neuron2, k_bilinear)
        # 直接计算，避免存储中间结果
        result = neuron1 + neuron2 + k_bilinear * neuron1 * neuron2
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：只计算必要的梯度
        neuron1, neuron2, k_bilinear = ctx.saved_tensors
        
        # 计算梯度
        # ∂s/∂neuron1 = 1 + k_bilinear * neuron2
        # ∂s/∂neuron2 = 1 + k_bilinear * neuron1
        # ∂s/∂k_bilinear = neuron1 * neuron2
        
        # 使用grad_output广播，避免存储大的中间tensor
        grad_neuron1 = grad_output * (1.0 + k_bilinear * neuron2)
        grad_neuron2 = grad_output * (1.0 + k_bilinear * neuron1)
        grad_k = grad_output * (neuron1 * neuron2)
        
        # 如果k_bilinear是标量，需要sum
        if grad_k.dim() > 0:
            grad_k = grad_k.sum()
        
        return grad_neuron1, grad_neuron2, grad_k


class BilinearClusterPotentialSigmoid(torch.autograd.Function):
    """
    内存优化的双线性簇膜电位计算（带sigmoid变换）
    优化sigmoid的梯度计算，减少内存占用
    """
    @staticmethod
    def forward(ctx, neuron1, neuron2, k_bilinear_raw):
        # 前向传播：k_bilinear = sigmoid(k_bilinear_raw)
        k_bilinear = torch.sigmoid(k_bilinear_raw)
        ctx.save_for_backward(neuron1, neuron2, k_bilinear, k_bilinear_raw)
        # 计算簇膜电位
        result = neuron1 + neuron2 + k_bilinear * neuron1 * neuron2
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：优化梯度计算
        neuron1, neuron2, k_bilinear, k_bilinear_raw = ctx.saved_tensors
        
        # 计算对neuron1和neuron2的梯度
        grad_neuron1 = grad_output * (1.0 + k_bilinear * neuron2)
        grad_neuron2 = grad_output * (1.0 + k_bilinear * neuron1)
        
        # 计算对k_bilinear_raw的梯度
        # ∂s/∂k_raw = ∂s/∂k · ∂k/∂k_raw
        #            = (neuron1 * neuron2) · sigmoid(k_raw) · (1 - sigmoid(k_raw))
        #            = (neuron1 * neuron2) · k_bilinear · (1 - k_bilinear)
        # 使用已保存的k_bilinear，避免重新计算sigmoid
        sigmoid_derivative = k_bilinear * (1.0 - k_bilinear)
        grad_k_raw = grad_output * (neuron1 * neuron2) * sigmoid_derivative
        
        # 如果k_bilinear_raw是标量，需要sum
        if grad_k_raw.dim() > 0:
            grad_k_raw = grad_k_raw.sum()
        
        return grad_neuron1, grad_neuron2, grad_k_raw


def neuron_cluster_membrane_potential(neuron_outputs, k_bilinear=0.0, use_memory_efficient=True):
    """
    计算神经元簇的膜电位，包含双线性项
    公式：簇膜电位 = 神经元1膜电位 + 神经元2膜电位 + k * 神经元1膜电位 * 神经元2膜电位
    
    Args:
        neuron_outputs: 神经元输出列表，每两个神经元组成一个簇
        k_bilinear: 双线性项系数（可以是标量或tensor）
        use_memory_efficient: 是否使用内存优化的版本（默认True）
    
    Returns:
        计算后的神经元簇膜电位
    """
    if len(neuron_outputs) < 2:
        return sum(neuron_outputs)
    
    # 将神经元输出按两个一组进行分组
    cluster_outputs = []
    for i in range(0, len(neuron_outputs), 2):
        if i + 1 < len(neuron_outputs):
            # 两个神经元组成一个簇
            neuron1 = neuron_outputs[i]
            neuron2 = neuron_outputs[i + 1]
            
            # 使用内存优化的版本
            if use_memory_efficient and isinstance(k_bilinear, torch.Tensor) and k_bilinear.requires_grad:
                cluster_potential = BilinearClusterPotential.apply(neuron1, neuron2, k_bilinear)
            else:
                # 对于不需要梯度的参数，使用普通计算
                cluster_potential = neuron1 + neuron2 + k_bilinear * neuron1 * neuron2
            cluster_outputs.append(cluster_potential)
        else:
            # 如果剩余一个神经元，直接使用
            cluster_outputs.append(neuron_outputs[i])
    
    return sum(cluster_outputs)


def neuron_cluster_membrane_potential_sigmoid(neuron1, neuron2, k_bilinear_raw, use_memory_efficient=True):
    """
    计算神经元簇的膜电位（带sigmoid变换），内存优化版本
    
    Args:
        neuron1: 神经元1的输出
        neuron2: 神经元2的输出
        k_bilinear_raw: 原始的双线性系数参数（需要sigmoid变换）
        use_memory_efficient: 是否使用内存优化的版本（默认True）
    
    Returns:
        计算后的神经元簇膜电位
    """
    if use_memory_efficient and k_bilinear_raw.requires_grad:
        # 使用内存优化的自定义autograd函数
        return BilinearClusterPotentialSigmoid.apply(neuron1, neuron2, k_bilinear_raw)
    else:
        # 普通计算
        k_bilinear = torch.sigmoid(k_bilinear_raw)
        return neuron1 + neuron2 + k_bilinear * neuron1 * neuron2
class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, act_fun, back_connection, k_bilinear=0.0, use_sigmoid_transform=True):
        # print(C_prev_prev, C_prev, C, reduction)
        super(Cell, self).__init__()
        self.act_fun = act_fun
        self.back_connection = back_connection
        self.reduction = reduction
        self.k_bilinear_init = k_bilinear  # 保存初始值用于创建参数
        self.use_sigmoid_transform = use_sigmoid_transform  # 是否使用sigmoid变换，防止梯度消失
        if reduction:
            self.fun = FactorizedReduce(
                C_prev, C * 3, act_fun=act_fun
            )
            self.multiplier = 3
            # reduction cell 不需要 k_bilinear 参数
            self.k_bilinear_params = None
        else:
            if reduction_prev:
                self.preprocess0 = FactorizedReduce(
                    C_prev_prev, C, act_fun=act_fun)
            else:
                self.preprocess0 = ReLUConvBN(
                    C_prev_prev, C, 1, 1, 0, act_fun=act_fun)
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, act_fun=act_fun)

            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
            self._compile(C, op_names, indices, concat, reduction)
            # 在 _compile 之后创建 k_bilinear 参数，为每个 step（每对神经元）创建独立的参数
            if self.use_sigmoid_transform:
                # 使用sigmoid变换：将原始参数w通过sigmoid映射到(0,1)
                # 如果初始k_bilinear=k_init，则w_init应该使得sigmoid(w_init)≈k_init
                # 当k_init接近0时，w_init应该是一个较小的负数
                # 使用logit变换：w = log(k/(1-k))，但当k接近0时会趋向负无穷
                # 因此使用近似：w_init = -log(1/k_init - 1) if k_init > 0 else -10
                if k_bilinear > 0 and k_bilinear < 1:
                    init_w = torch.logit(torch.tensor(k_bilinear, dtype=torch.float32))
                else:
                    # 对于k_bilinear=0或超出范围的情况，初始化为接近0的值
                    init_w = torch.tensor(-10.0, dtype=torch.float32)  # sigmoid(-10) ≈ 0
                self.k_bilinear_params = nn.ParameterList([
                    nn.Parameter(init_w.clone())
                    for _ in range(self._steps)
                ])
            else:
                # 直接使用，不经过sigmoid变换
                self.k_bilinear_params = nn.ParameterList([
                    nn.Parameter(torch.tensor(float(k_bilinear), dtype=torch.float32))
                    for _ in range(self._steps)
                ])

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        # self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        self._ops_back = nn.ModuleList()
        back_begin_index = 0
        for i, (name, index) in enumerate(zip(op_names, indices)):
            # print(name, index)
            if '_back' in name:
                back_begin_index = i
                break
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True, act_fun=self.act_fun)
            self._ops += [op]

        if self.back_connection:
            for name, index in zip(op_names[back_begin_index:], indices[back_begin_index:]):
                op = OPS[name.replace('_back', '')](
                    C, 1, True, act_fun=self.act_fun)
                self._ops_back += [op]

        if self.back_connection:
            self._indices_forward = indices[:back_begin_index]
            self._indices_backward = indices[back_begin_index:]
        else:
            self._indices_backward = []
            self._indices_forward = indices
        self._steps = len(self._indices_forward) // 2

    def forward(self, s0, s1, drop_prob):
        if self.reduction:
            return self.fun(s1)

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices_forward[2 * i]]
            h2 = states[self._indices_forward[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            # 使用新的神经元簇膜电位计算方式，内存优化版本
            # 使用对应 step 的 k_bilinear 参数
            if self.k_bilinear_params is not None:
                k_bilinear_raw = self.k_bilinear_params[i]
                # 如果使用sigmoid变换，使用内存优化的sigmoid版本
                if self.use_sigmoid_transform:
                    s = neuron_cluster_membrane_potential_sigmoid(h1, h2, k_bilinear_raw, use_memory_efficient=True)
                else:
                    s = BilinearClusterPotential.apply(h1, h2, k_bilinear_raw)
            else:
                # reduction cell或没有k_bilinear参数的情况
                s = h1 + h2
            if self.back_connection:
                if i != 0:
                    s_back = self._ops_back[i - 1](s)
                    states[self._indices_backward[i - 1]
                           ] = states[self._indices_backward[i - 1]] + s_back
            states += [s]
        outputs = torch.cat([states[i]
                            for i in self._concat], dim=1)  # N，C，H, W
        return outputs
        # return self.node(outputs)


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes, act_fun):
        """assuming inputs size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.act_fun = act_fun
        self.features = nn.Sequential(
            # nn.ReLU(inplace=True),
            self.act_fun(),
            # image size = 2 x 2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            self.act_fun(),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            # nn.ReLU(inplace=True)
            self.act_fun()
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming inputs size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


@register_model
class NetworkCIFAR(BaseModule):

    def __init__(self,
                 C,
                 num_classes,
                 layers,
                 auxiliary,
                 genotype,
                 parse_method='darts',
                 step=1,
                 node_type='ReLUNode',
                 k_bilinear=0.0,
                 use_sigmoid_transform=True,
                 **kwargs):
        super(NetworkCIFAR, self).__init__(
            step=step,
            num_classes=num_classes,
            **kwargs
        )
        if isinstance(node_type, str):
            self.act_fun = eval(node_type)
        else:
            self.act_fun = node_type
        self.act_fun = partial(self.act_fun, **kwargs)

        if 'back_connection' in kwargs.keys():
            self.back_connection = kwargs['back_connection']
        else:
            self.back_connection = False

        self.spike_output = kwargs['spike_output'] if 'spike_output' in kwargs else True
        self.dataset = kwargs['dataset']
        self.k_bilinear = k_bilinear  # 双线性项系数
        self.use_sigmoid_transform = use_sigmoid_transform  # 是否使用sigmoid变换，防止梯度消失

        if self.layer_by_layer:
            self.flatten = nn.Flatten(start_dim=1)
        else:
            self.flatten = nn.Flatten()

        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        if self.dataset == 'dvsg' or self.dataset == 'dvsc10' or self.dataset == 'NCALTECH101':
            in_channels = 2 * self.init_channel_mul
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            # self.reduce_idx = [
            #     layers // 4,
            #     layers // 2,
            #     3 * layers // 4
            # ]
            self.reduce_idx = [1, 3, 5, 7]
        elif self.dataset == 'mnist' or self.dataset == 'fashion-mnist':
            # MNIST/Fashion-MNIST数据集：1通道（灰度图）
            in_channels = 1 * self.init_channel_mul
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            self.reduce_idx = [layers // 4,
                               layers // 2,
                               3 * layers // 4]
        else:
            in_channels = 3 * self.init_channel_mul
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            self.reduce_idx = [layers // 4,
                               layers // 2,
                               3 * layers // 4]

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in self.reduce_idx:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            if parse_method == 'darts':
                cell = Cell(genotype, C_prev_prev, C_prev, C_curr,
                            reduction, reduction_prev,
                            act_fun=self.act_fun, back_connection=self.back_connection, 
                            k_bilinear=self.k_bilinear, use_sigmoid_transform=self.use_sigmoid_transform)
            
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(
                C_to_auxiliary, num_classes, act_fun=self.act_fun)
        self.global_pooling = nn.Sequential(
            self.act_fun(), nn.AdaptiveAvgPool2d(1))

        if self.spike_output:
            self.classifier = nn.Sequential(
                nn.Linear(C_prev, 10 * num_classes),
                self.act_fun())
            self.vote = VotingLayer(10)
        else:
            self.classifier = nn.Linear(C_prev, num_classes)
            self.vote = nn.Identity()

        # self.classifier = nn.Linear(C_prev, num_classes)
        # self.vote = nn.Identity()

    def forward(self, inputs):
        logits_aux = None
        inputs = self.encoder(inputs)
        if not self.layer_by_layer:
            outputs = []
            output_aux = []
            self.reset()
            for t in range(self.step):
                x = inputs[t]
                s0 = s1 = self.stem(x)
                for i, cell in enumerate(self.cells):
                    s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
                    # print(s0.shape, s1.shape)
                    if i == 2 * self._layers // 3:
                        if self._auxiliary and self.training:
                            logits_aux = self.auxiliary_head(s1)
                out = self.global_pooling(s1)
                out = self.classifier(self.flatten(out))
                logits = self.vote(out)
                outputs.append(logits)
                output_aux.append(logits_aux)
            main_logits = sum(outputs) / len(outputs)
            aux_logits = logits_aux if logits_aux is None else (sum(output_aux) / len(output_aux))
            return main_logits, aux_logits
        else:
            s0 = s1 = self.stem(inputs)
            for i, cell in enumerate(self.cells):
                s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
                if i == 2 * self._layers // 3:
                    if self._auxiliary and self.training:
                        logits_aux = self.auxiliary_head(s1)
            out = self.global_pooling(s1)
            out = self.classifier(self.flatten(out))
            out = rearrange(out, '(t b) c -> t b c', t=self.step).mean(0)
            logits = self.vote(out)
            return logits, logits_aux

    def get_all_k_bilinear_values(self):
        """
        收集所有Cell中的k_bilinear参数值
        返回一个字典，键为(layer_idx, step_idx)，值为k_bilinear的实际值（如果使用sigmoid变换则转换）
        """
        k_bilinear_dict = {}
        total_count = 0
        
        for layer_idx, cell in enumerate(self.cells):
            if hasattr(cell, 'k_bilinear_params') and cell.k_bilinear_params is not None:
                # 这是一个normal cell，有k_bilinear参数
                for step_idx, k_log_param in enumerate(cell.k_bilinear_params):
                    if self.use_sigmoid_transform:
                        # 如果使用sigmoid变换，需要将k_log转换为k_bilinear
                        k_bilinear_value = torch.sigmoid(k_log_param).item()
                    else:
                        # 直接使用参数值
                        k_bilinear_value = k_log_param.item()
                    k_bilinear_dict[(layer_idx, step_idx)] = k_bilinear_value
                    total_count += 1
        
        return k_bilinear_dict, total_count

    def print_k_bilinear_values(self, epoch, logger=None):
        """
        打印所有k_bilinear参数的值，包括梯度信息
        """
        k_bilinear_dict, total_count = self.get_all_k_bilinear_values()
        
        if total_count == 0:
            return
        
        # 收集梯度信息
        grad_dict = {}
        for layer_idx, cell in enumerate(self.cells):
            if hasattr(cell, 'k_bilinear_params') and cell.k_bilinear_params is not None:
                for step_idx, k_log_param in enumerate(cell.k_bilinear_params):
                    if k_log_param.grad is not None:
                        grad_dict[(layer_idx, step_idx)] = k_log_param.grad.abs().item()
                    else:
                        grad_dict[(layer_idx, step_idx)] = 0.0
        
        if logger is not None:
            logger.info('=' * 80)
            logger.info(f'Epoch {epoch}: k_bilinear参数值 (共{total_count}个)')
            logger.info('-' * 80)
            
            # 按layer分组打印
            current_layer = -1
            for (layer_idx, step_idx), k_value in sorted(k_bilinear_dict.items()):
                if layer_idx != current_layer:
                    if current_layer >= 0:
                        logger.info('')  # 换行分隔不同layer
                    logger.info(f'Layer {layer_idx} (Normal Cell):')
                    current_layer = layer_idx
                grad_info = f', grad={grad_dict.get((layer_idx, step_idx), 0.0):.6e}' if (layer_idx, step_idx) in grad_dict else ', grad=None'
                logger.info(f'  Step {step_idx}: k_bilinear = {k_value:.6f}{grad_info}')
            
            logger.info('-' * 80)
            logger.info(f'总计: {total_count}个k_bilinear参数')
            logger.info('=' * 80)
        else:
            # 如果没有logger，使用print
            print('=' * 80)
            print(f'Epoch {epoch}: k_bilinear参数值 (共{total_count}个)')
            print('-' * 80)
            
            current_layer = -1
            for (layer_idx, step_idx), k_value in sorted(k_bilinear_dict.items()):
                if layer_idx != current_layer:
                    if current_layer >= 0:
                        print('')  # 换行分隔不同layer
                    print(f'Layer {layer_idx} (Normal Cell):')
                    current_layer = layer_idx
                print(f'  Step {step_idx}: k_bilinear = {k_value:.6f}')
            
            print('-' * 80)
            print(f'总计: {total_count}个k_bilinear参数')
            print('=' * 80)
    



@register_model
class NetworkImageNet(BaseModule):

    def __init__(self,
                 C,
                 num_classes,
                 layers,
                 auxiliary,
                 genotype,
                 step=1,
                 node_type='ReLUNode',
                 k_bilinear=0.1,
                 use_sigmoid_transform=True,
                 **kwargs):
        super(NetworkImageNet, self).__init__(
            step=step,
            num_classes=num_classes,
            **kwargs)

        if isinstance(node_type, str):
            self.act_fun = eval(node_type)
        else:
            self.act_fun = node_type
        self.act_fun = partial(self.act_fun, **kwargs)

        if 'back_connection' in kwargs.keys():
            self.back_connection = kwargs['back_connection']
        else:
            self.back_connection = False

        self.spike_output = kwargs['spike_output'] if 'spike_output' in kwargs else True
        self.k_bilinear = k_bilinear  # 双线性项系数
        self.use_sigmoid_transform = use_sigmoid_transform  # 是否使用sigmoid变换，防止梯度消失

        if self.layer_by_layer:
            self.flatten = nn.Flatten(start_dim=1)
        else:
            self.flatten = nn.Flatten()

        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            # nn.ReLU(inplace=True),
            self.act_fun(),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            # nn.ReLU(inplace=True),
            self.act_fun(),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev,
                        C_curr, reduction, reduction_prev,
                        act_fun=self.act_fun, back_connection=self.back_connection, 
                        k_bilinear=self.k_bilinear, use_sigmoid_transform=self.use_sigmoid_transform)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, inputs):
        outputs = []
        self.reset()
        for t in range(self.step):
            s0 = self.stem0(inputs)
            s1 = self.stem1(s0)
            for i, cell in enumerate(self.cells):
                s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            out = self.global_pooling(s1)
            logits = self.classifier(self.flatten(out))
            outputs.append(logits)
        return sum(outputs) / len(outputs)
