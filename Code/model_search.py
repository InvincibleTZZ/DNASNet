"""
基于STDP的神经架构搜索模型
==========================

关键特性：
1. 使用纯STDP无监督局部规则优化架构参数（无梯度信息）
2. 操作空间包含_p（兴奋性）和_n（抑制性）变体，由搜索决定
3. Cell包含3个计算节点+2个输入节点，共9条边（2+3+4）
4. 每个节点选top-2边，每边选top-1操作 → 共6个最终操作
5. Cell输出为3个节点输出的拼接
"""

from functools import partial
import logging
import numpy as np
from DNASNet_operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES, Genotype
from _init_ import parse, forward_edge_num, edge_num

from braincog.base.connection.layer import VotingLayer
from braincog.base.node.node import *
from braincog.model_zoo.base_module import BaseModule

try:
    from stdp_optimizer import STDPArchOptimizer, HybridArchOptimizer
except Exception:
    STDPArchOptimizer = None
    HybridArchOptimizer = None
    print("Warning: STDP optimizers not found")


def calc_weight(x):
    """计算架构权重（按节点分组 softmax 后求和）"""
    tmp0 = torch.split(x[0], edge_num, dim=0)
    tmp1 = torch.split(x[1], edge_num, dim=0)
    res = []
    for i in range(len(edge_num)):
        res.append(
            torch.softmax(tmp0[i].view(-1), dim=-1).view(tmp0[i].shape)
            + torch.softmax(tmp1[i].view(-1), dim=-1).view(tmp1[i].shape)
        )
    return torch.cat(res, dim=0)


def calc_loss(x):
    """计算辅助损失（按节点分组 softmax 后求差）"""
    tmp0 = torch.split(x[0], edge_num, dim=0)
    tmp1 = torch.split(x[1], edge_num, dim=0)
    res = []
    for i in range(len(edge_num)):
        res.append(
            torch.softmax(tmp0[i].view(-1), dim=-1).view(tmp0[i].shape)
            - torch.softmax(tmp1[i].view(-1), dim=-1).view(tmp1[i].shape)
        )
    return torch.cat(res, dim=0)


class darts_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        output = inputs * weights
        ctx.save_for_backward(inputs, weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_inputs, grad_weights = None, None
        inputs, weights = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output * weights
        if ctx.needs_input_grad[1]:
            if torch.min(inputs) < -1e-12 and torch.max(inputs) > 1e-12:
                inputs = torch.abs(inputs) / 2.
            else:
                inputs = torch.abs(inputs)
            grad_weights = -inputs.mean()
        return grad_inputs, grad_weights


# =============================================================================
# 混合操作
# =============================================================================

class MixedOp(nn.Module):
    """每条边上多个候选操作的加权组合"""
    def __init__(self, C, stride, act_fun, edge_idx, stdp_optimizer=None):
        super().__init__()
        self._ops = nn.ModuleList()
        self._op_names = []
        for primitive in PRIMITIVES:
            if primitive in OPS:
                op = OPS[primitive](C, stride, False, act_fun)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                self._ops.append(op)
                self._op_names.append(primitive)
        self.multiply = darts_fun.apply
        self.edge_idx = edge_idx
        self.stdp_optimizer = stdp_optimizer

    def forward(self, x, weights, time_step=0, record_spikes=False):
        feature_map = []
        for i, op in enumerate(self._ops):
            res = op(x)
            feature_map.append(res)
            if record_spikes and self.stdp_optimizer is not None:
                with torch.no_grad():
                    fire_rate = (res > 0).float().mean().item()
                    self.stdp_optimizer.record_spike_trace(
                        self.edge_idx, i, time_step, fire_rate)
        return sum(self.multiply(mp, w) for w, mp in zip(weights, feature_map))


# =============================================================================
# Cell
# =============================================================================

# 9条边到节点的静态映射
EDGE_TO_NODES = {
    0: (0, 2), 1: (1, 2),             # 簇2: input0→2, input1→2
    2: (0, 3), 3: (1, 3), 4: (2, 3),  # 簇3: input0→3, input1→3, 簇2→3
    5: (0, 4), 6: (1, 4), 7: (2, 4), 8: (3, 4),  # 簇4
}

# 每个节点的输入边数
NODE_EDGES = [2, 3, 4]  # 簇2, 簇3, 簇4


class SearchCell(nn.Module):
    """
    搜索Cell: 3个计算节点, 9条边 (2+3+4)
    
    节点: 0=input0, 1=input1, 2=簇2, 3=簇3, 4=簇4
    输出: cat(簇2, 簇3, 簇4)
    """
    def __init__(self, C_prev_prev, C_prev, C, reduction, reduction_prev,
                 act_fun, stdp_optimizer=None):
        super().__init__()
        self.reduction = reduction
        if reduction:
            self.fun = FactorizedReduce(C_prev, C * 3, affine=True, act_fun=act_fun, positive=1)
        else:
            if reduction_prev:
                self.preprocess0 = FactorizedReduce(
                    C_prev_prev, C, affine=False, act_fun=act_fun, positive=1)
            else:
                self.preprocess0 = ReLUConvBN(
                    C_prev_prev, C, 1, 1, 0, affine=False, act_fun=act_fun, positive=1)
            self.preprocess1 = ReLUConvBN(
                C_prev, C, 1, 1, 0, affine=False, act_fun=act_fun, positive=1)

            self._ops = nn.ModuleList()
            for edge_idx in range(9):
                self._ops.append(MixedOp(C, stride=1, act_fun=act_fun,
                                         edge_idx=edge_idx,
                                         stdp_optimizer=stdp_optimizer))

    def forward(self, s0, s1, weights, time_step=0, record_spikes=False):
        if self.reduction:
            return self.fun(s1)

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        if record_spikes:
            stdp = self._ops[0].stdp_optimizer
            if stdp is not None and hasattr(stdp, 'record_node_spike'):
                stdp.record_node_spike(0, time_step, s0)
                stdp.record_node_spike(1, time_step, s1)

        offset = 0
        for node_id, num_edges in enumerate(NODE_EDGES):
            node_weights = [weights[offset + j] for j in range(num_edges)]
            node_inputs = [states[j] for j in range(len(states))][:num_edges]
            s = sum(self._ops[offset + j](node_inputs[j], node_weights[j],
                                          time_step, record_spikes)
                    for j in range(num_edges))
            states.append(s)

            if record_spikes:
                stdp = self._ops[0].stdp_optimizer
                if stdp is not None and hasattr(stdp, 'record_node_spike'):
                    stdp.record_node_spike(node_id + 2, time_step, s)

            offset += num_edges

        return torch.cat(states[2:], dim=1)


# =============================================================================
# 搜索网络
# =============================================================================

class NetworkWithSTDP(BaseModule):
    """支持STDP的神经架构搜索网络"""
    def __init__(self, C, num_classes, layers, criterion, stem_multiplier=3,
                 parse_method='bio_darts', op_threshold=None, step=1,
                 node_type='BiasPLIFNode', use_stdp=True, stdp_type='full',
                 **kwargs):
        super().__init__(step=step, encode_type='direct', **kwargs)

        self.act_fun = eval(node_type)
        self.act_fun = partial(self.act_fun, **kwargs)

        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self.parse_method = parse_method
        self.op_threshold = op_threshold
        self.fire_rate_per_step = [0.] * self.step
        self.forward_step = 0
        self.record_fire_rate = False
        self.dataset = kwargs.get('dataset', 'cifar10')
        self.spike_output = kwargs.get('spike_output', True)

        C_curr = stem_multiplier * C

        if self.dataset in ['dvsg', 'dvsc10', 'NCALTECH101']:
            self.stem = nn.Sequential(
                nn.Conv2d(2 * self.init_channel_mul, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr))
            self.reduce_idx = [layers // 3, 2 * layers // 3]
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3 * self.init_channel_mul, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr))
            self.reduce_idx = [1, 3, 5]

        self._stdp_type = stdp_type
        self._initialize_alphas()
        self._init_stdp_optimizer(stdp_type, use_stdp)

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in self.reduce_idx:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = SearchCell(C_prev_prev, C_prev, C_curr,
                              reduction, reduction_prev,
                              self.act_fun, stdp_optimizer=self.stdp_optimizer)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, 3 * C_curr

        self.global_pooling = nn.Sequential(self.act_fun(), nn.AdaptiveAvgPool2d(1))
        if self.spike_output:
            self.classifier = nn.Sequential(
                nn.Linear(C_prev, 10 * num_classes), self.act_fun())
            self.vote = VotingLayer(10)
        else:
            self.classifier = nn.Linear(C_prev, num_classes)
            self.vote = nn.Identity()

    def _init_stdp_optimizer(self, stdp_type, use_stdp):
        """初始化STDP优化器"""
        self.stdp_optimizer = None
        if not use_stdp:
            return
        if stdp_type == 'full' and STDPArchOptimizer is not None:
            self.stdp_optimizer = STDPArchOptimizer(
                self.alphas_normal,
                tau_plus=20.0, tau_minus=20.0,
                A_plus=0.01, A_minus=0.012, mu=0.1)
        elif stdp_type == 'hybrid' and HybridArchOptimizer is not None:
            self.stdp_optimizer = HybridArchOptimizer(
                self.alphas_normal, stdp_weight=0.5, gradient_lr=0.01)

    def _initialize_alphas(self):
        """初始化架构参数 (2, 9, num_ops)"""
        k = 9
        temp_op = MixedOp(C=self._C, stride=1, act_fun=self.act_fun,
                          edge_idx=0, stdp_optimizer=None)
        num_ops = len(temp_op._ops)
        self._op_names = temp_op._op_names

        # Hybrid 模式需要梯度（计算 ∇_α L_val），纯 STDP 不需要
        need_grad = (self._stdp_type == 'hybrid')
        self.alphas_normal = Variable(
            0.5 * torch.randn(2, k, num_ops).cuda(), requires_grad=need_grad)

        self.alphas_normal_history = {}
        for i in range(k):
            for j in range(num_ops):
                op_name = self._op_names[j] if j < len(self._op_names) else f'op_{j}'
                self.alphas_normal_history[f'edge_{i}_{op_name}'] = []

    def forward(self, inputs, record_spikes=False):
        inputs = self.encoder(inputs)
        self.reset()
        if not self.training:
            self.fire_rate.clear()

        outputs = []
        for t in range(self.step):
            x = inputs[t]
            s0 = s1 = self.stem(x)
            for cell in self.cells:
                if cell.reduction:
                    s0, s1 = s1, cell(s0, s1, None, time_step=t)
                else:
                    weights = calc_weight(self.alphas_normal)
                    s0, s1 = s1, cell(s0, s1, weights, time_step=t,
                                      record_spikes=record_spikes)
            out = self.global_pooling(s1)
            out = self.classifier(out.view(out.size(0), -1))
            outputs.append(self.vote(out))

            if self.record_fire_rate:
                with torch.no_grad():
                    self.fire_rate_per_step[t] += (s1 > 0).float().mean().item()

        if self.record_fire_rate:
            self.forward_step += 1
        return sum(outputs) / len(outputs)

    def update_arch_with_stdp(self, prune_threshold=0.01, dominant_threshold=0.6,
                              enable_exploration=True, exploration_rate=5e-4):
        """使用STDP更新架构参数"""
        if self.stdp_optimizer is None:
            return False
        self.stdp_optimizer.step(
            edge_to_nodes=EDGE_TO_NODES,
            prune_threshold=prune_threshold,
            dominant_threshold=dominant_threshold,
            enable_exploration=enable_exploration,
            exploration_rate=exploration_rate)
        return True

    def _loss(self, input1, target1, input2=None):
        logits = self(input1)
        if input2 is not None:
            return self._criterion(logits, target1, input2)
        return self._criterion(logits, target1)

    def arch_parameters(self):
        return [self.alphas_normal]

    def genotype(self):
        """
        生成基因型: 每个节点选 top-2 边, 每边选 top-1 操作
        操作的 _p/_n 后缀决定节点的 E/I 类型
        """
        weights = calc_weight(self.alphas_normal).data.cpu().numpy()
        op_names = self._op_names

        gene_normal = []
        node_configs = [
            (0, 2, [0, 1]),
            (2, 5, [0, 1, 2]),
            (5, 9, [0, 1, 2, 3]),
        ]

        for node_id, (edge_start, edge_end, available_inputs) in enumerate(node_configs):
            all_edges = []
            for edge_idx in range(edge_start, edge_end):
                best_op_idx = np.argmax(weights[edge_idx])
                best_weight = weights[edge_idx][best_op_idx]
                best_op = op_names[best_op_idx]
                source_node = available_inputs[edge_idx - edge_start]
                all_edges.append((best_weight, best_op, source_node))

            all_edges.sort(reverse=True, key=lambda x: x[0])
            for _, op, source in all_edges[:2]:
                gene_normal.append((op, source))

            top2_ops = [op for _, op, _ in all_edges[:2]]
            has_p = any('_p' in op for op in top2_ops)
            has_n = any('_n' in op for op in top2_ops)
            if has_p and has_n:
                ntype = 'E-I'
            elif has_n:
                ntype = 'I-I'
            elif has_p:
                ntype = 'E-E'
            else:
                ntype = 'unknown'
            logging.info(f"  node{node_id + 2}: {ntype} ({top2_ops})")

        return Genotype(normal=gene_normal, normal_concat=range(2, 5))

    def states(self):
        state = {
            'alphas_normal': self.alphas_normal,
            'alphas_normal_history': self.alphas_normal_history,
            'criterion': self._criterion,
        }
        if self.stdp_optimizer is not None and hasattr(self.stdp_optimizer, 'state_dict'):
            state['stdp_optimizer'] = self.stdp_optimizer.state_dict()
        return state

    def restore(self, states):
        self.alphas_normal = states['alphas_normal']
        self.alphas_normal_history = states['alphas_normal_history']
        if 'stdp_optimizer' in states and states['stdp_optimizer'] is not None:
            if self.stdp_optimizer is not None and hasattr(self.stdp_optimizer, 'load_state_dict'):
                self.stdp_optimizer.load_state_dict(states['stdp_optimizer'])

    def update_history(self):
        weights = calc_weight(self.alphas_normal).data.cpu().numpy()
        k, num_ops = weights.shape
        for i in range(k):
            for j in range(num_ops):
                op_name = self._op_names[j] if j < len(self._op_names) else f'op_{j}'
                key = f'edge_{i}_{op_name}'
                if key not in self.alphas_normal_history:
                    self.alphas_normal_history[key] = []
                self.alphas_normal_history[key].append(float(weights[i][j]))

    def get_stdp_statistics(self):
        if self.stdp_optimizer is not None and hasattr(self.stdp_optimizer, 'get_statistics'):
            return self.stdp_optimizer.get_statistics()
        return {}

    def reset_fire_rate_record(self):
        self.fire_rate_per_step = [0.] * self.step
        self.forward_step = 0

    def get_fire_per_step(self):
        if self.forward_step == 0:
            return [0.] * self.step
        return [x / self.forward_step for x in self.fire_rate_per_step]


Network = NetworkWithSTDP
