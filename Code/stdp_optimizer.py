"""
STDP架构参数优化器
用于在SNN神经架构搜索中使用STDP规则更新架构参数α

提供两种优化器：
  STDPArchOptimizer      - 纯STDP（基于脉冲时序的LTP/LTD规则）
  HybridArchOptimizer    - 混合优化（STDP + 梯度下降）

所有优化器共享统一接口：
  record_spike_trace(edge_idx, op_idx, time_step, spike_rate)
  record_node_spike(node_id, time_step, spike_tensor)
  step(edge_to_nodes=None, prune_threshold=0.01, dominant_threshold=0.6,
       enable_exploration=True, exploration_rate=5e-4)
  reset_traces()
  get_statistics() -> dict
  state_dict() -> dict
  load_state_dict(state_dict)
"""

import torch
import numpy as np
from collections import defaultdict


# =============================================================================
# 纯STDP优化器
# =============================================================================

class STDPArchOptimizer:
    """
    使用STDP规则优化架构参数α
    
    STDP规则：
    - 前突触先于后突触发放 (Δt > 0) → 增强连接 (LTP)
    - 后突触先于前突触发放 (Δt < 0) → 削弱连接 (LTD)
    
    更新公式：
    Δα = A_+ * exp(-Δt/τ_+) * μ * (α_max - α)  if Δt > 0
    Δα = -A_- * exp(Δt/τ_-) * μ * α             if Δt < 0
    """
    
    def __init__(self, alphas, tau_plus=20.0, tau_minus=20.0, 
                 A_plus=0.01, A_minus=0.012, mu=0.1, alpha_max=1.0,
                 use_weight_dependent=True, use_node_based_stdp=True):
        self.alphas = alphas
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.mu = mu
        self.alpha_max = alpha_max
        self.use_weight_dependent = use_weight_dependent
        self.use_node_based_stdp = use_node_based_stdp
        
        self.spike_traces = defaultdict(list)
        self.node_spike_times = defaultdict(list)
        self.op_spike_times = defaultdict(list)
        
        self.update_count = 0
        self.ltp_count = 0
        self.ltd_count = 0
        self.pruned_ops = set()
        self.dominant_edges = set()
    
    def reset_traces(self):
        self.spike_traces.clear()
        self.node_spike_times.clear()
        self.op_spike_times.clear()
    
    def record_node_spike(self, node_id, time_step, spike_tensor):
        """记录节点的脉冲发放时刻"""
        has_spike = (spike_tensor > 0).any().item()
        if has_spike:
            self.node_spike_times[node_id].append(time_step)
    
    def record_spike_trace(self, edge_idx, op_idx, time_step, spike_rate):
        """记录边上操作的脉冲活动"""
        key = (edge_idx, op_idx)
        self.spike_traces[key].append({'time': time_step, 'rate': spike_rate})
        if spike_rate > 0:
            self.op_spike_times[key].append(time_step)
    
    def _compute_update(self, edge_idx, op_idx, source_node=None,
                        enable_exploration=True, exploration_rate=1e-4):
        """计算单个架构参数的STDP更新量"""
        current_alpha = self.alphas[0, edge_idx, op_idx].item()
        
        if self.use_node_based_stdp and source_node is not None:
            pre_times = self.node_spike_times.get(source_node, [])
            post_times = self.op_spike_times.get((edge_idx, op_idx), [])
            
            if not pre_times or not post_times:
                if enable_exploration:
                    return exploration_rate * (1.0 - current_alpha / 0.1) if current_alpha < 0.1 \
                        else -exploration_rate * 0.1
                return 0.0
            
            delta = 0.0
            for t_pre in pre_times:
                for t_post in post_times:
                    dt = t_post - t_pre
                    if self.use_weight_dependent:
                        w_ltp = self.mu * (self.alpha_max - current_alpha)
                        w_ltd = self.mu * current_alpha
                    else:
                        w_ltp = w_ltd = 1.0
                    
                    if dt > 0:
                        delta += self.A_plus * np.exp(-dt / self.tau_plus) * w_ltp
                        self.ltp_count += 1
                    elif dt < 0:
                        delta -= self.A_minus * np.exp(dt / self.tau_minus) * w_ltd
                        self.ltd_count += 1
            
            if enable_exploration and abs(delta) < exploration_rate:
                if current_alpha < 0.1:
                    delta += exploration_rate * 0.5 * (1.0 - current_alpha / 0.1)
                elif current_alpha > 0.9:
                    delta -= exploration_rate * 0.5
            return delta
        
        # fallback: 基于发放率
        traces = self.spike_traces.get((edge_idx, op_idx), [])
        total_rate = sum(t['rate'] for t in traces)
        if len(traces) < 2 or total_rate < 1e-6:
            if enable_exploration:
                return exploration_rate * (1.0 - current_alpha / 0.1) if current_alpha < 0.1 \
                    else -exploration_rate * 0.1
            return 0.0
        
        delta = 0.0
        pre_times = self.node_spike_times.get(source_node, []) if source_node is not None else []
        if pre_times:
            for t_pre in pre_times:
                for trace in traces:
                    dt = trace['time'] - t_pre
                    rate = trace['rate']
                    w_ltp = self.mu * (self.alpha_max - current_alpha) if self.use_weight_dependent else 1.0
                    w_ltd = self.mu * current_alpha if self.use_weight_dependent else 1.0
                    if dt > 0 and rate > 0:
                        delta += self.A_plus * np.exp(-dt / self.tau_plus) * w_ltp * rate
                        self.ltp_count += 1
                    elif dt < 0 and rate > 0:
                        delta -= self.A_minus * np.exp(dt / self.tau_minus) * w_ltd * rate
                        self.ltd_count += 1
        else:
            avg_rate = total_rate / len(traces)
            for trace in traces:
                rate = trace['rate']
                if rate > avg_rate:
                    delta += self.A_plus * (rate - avg_rate) * (self.alpha_max - current_alpha)
                elif rate < avg_rate * 0.5:
                    delta -= self.A_minus * (avg_rate - rate) * current_alpha
        
        if enable_exploration and abs(delta) < exploration_rate:
            if current_alpha < 0.1:
                delta += exploration_rate * 0.5 * (1.0 - current_alpha / 0.1)
            elif current_alpha > 0.9:
                delta -= exploration_rate * 0.5
        return delta

    def step(self, edge_to_nodes=None, prune_threshold=0.01, dominant_threshold=0.6,
             enable_exploration=True, exploration_rate=5e-4, min_update_threshold=1e-10):
        """执行一步STDP更新并剪枝"""
        k, num_ops = self.alphas[0].shape
        
        with torch.no_grad():
            for edge_idx in range(k):
                for op_idx in range(num_ops):
                    if (edge_idx, op_idx) in self.pruned_ops:
                        if enable_exploration and self.alphas[0, edge_idx, op_idx].item() < -50:
                            delta = exploration_rate * 0.5
                        else:
                            continue
                    else:
                        src, _ = (edge_to_nodes or {}).get(edge_idx, (None, None))
                        delta = self._compute_update(
                            edge_idx, op_idx, source_node=src,
                            enable_exploration=enable_exploration,
                            exploration_rate=exploration_rate)
                    
                    if abs(delta) > min_update_threshold:
                        self.alphas[0, edge_idx, op_idx] += delta
                        self.alphas[1, edge_idx, op_idx] += delta
                        self.alphas[0, edge_idx, op_idx].clamp_(-100, self.alpha_max)
                        self.alphas[1, edge_idx, op_idx].clamp_(-100, self.alpha_max)
                        self.update_count += 1
            
            # 剪枝
            softmax_w = torch.softmax(self.alphas[0], dim=-1)
            for edge_idx in range(k):
                for op_idx in range(num_ops):
                    if softmax_w[edge_idx, op_idx] < prune_threshold:
                        self.alphas[0, edge_idx, op_idx] = -100.0
                        self.alphas[1, edge_idx, op_idx] = -100.0
                        self.pruned_ops.add((edge_idx, op_idx))
                
                edge_w = torch.softmax(self.alphas[0, edge_idx], dim=-1)
                max_idx = torch.argmax(edge_w).item()
                max_w = edge_w[max_idx].item()
                if max_w > (edge_w.sum().item() - max_w) and max_w > dominant_threshold:
                    for op_idx in range(num_ops):
                        if op_idx != max_idx:
                            self.alphas[0, edge_idx, op_idx] = -100.0
                            self.alphas[1, edge_idx, op_idx] = -100.0
                            self.pruned_ops.add((edge_idx, op_idx))
                    self.dominant_edges.add(edge_idx)
        
        self.reset_traces()
    
    def get_statistics(self):
        return {
            'total_updates': self.update_count,
            'ltp_count': self.ltp_count,
            'ltd_count': self.ltd_count,
            'ltp_ratio': self.ltp_count / max(1, self.ltp_count + self.ltd_count),
            'pruned_ops_count': len(self.pruned_ops),
            'dominant_edges_count': len(self.dominant_edges),
        }
    
    def state_dict(self):
        return {
            'tau_plus': self.tau_plus, 'tau_minus': self.tau_minus,
            'A_plus': self.A_plus, 'A_minus': self.A_minus,
            'mu': self.mu, 'alpha_max': self.alpha_max,
            'use_weight_dependent': self.use_weight_dependent,
            'update_count': self.update_count,
            'ltp_count': self.ltp_count, 'ltd_count': self.ltd_count,
            'pruned_ops': list(self.pruned_ops),
            'dominant_edges': list(self.dominant_edges),
        }
    
    def load_state_dict(self, state_dict):
        for key in ['tau_plus', 'tau_minus', 'A_plus', 'A_minus', 'mu',
                     'alpha_max', 'use_weight_dependent']:
            if key in state_dict:
                setattr(self, key, state_dict[key])
        self.update_count = state_dict.get('update_count', 0)
        self.ltp_count = state_dict.get('ltp_count', 0)
        self.ltd_count = state_dict.get('ltd_count', 0)
        self.pruned_ops = set(state_dict.get('pruned_ops', []))
        self.dominant_edges = set(state_dict.get('dominant_edges', []))


# =============================================================================
# 混合优化器（STDP + 梯度下降）
# =============================================================================

class HybridArchOptimizer:
    """
    混合优化器: α' = (1-λ) * α_gradient + λ * α_STDP
    
    内部维护一个 STDPArchOptimizer，step() 先做 STDP 更新，
    再叠加梯度更新（通过 record_gradient 提供）。
    """
    
    def __init__(self, alphas, stdp_weight=0.5, gradient_lr=0.01, **stdp_kwargs):
        self.alphas = alphas
        self.stdp_weight = stdp_weight
        self.gradient_lr = gradient_lr
        self.stdp_optimizer = STDPArchOptimizer(alphas, **stdp_kwargs)
        self.gradient_buffer = None
    
    def record_spike_trace(self, edge_idx, op_idx, time_step, spike_rate):
        self.stdp_optimizer.record_spike_trace(edge_idx, op_idx, time_step, spike_rate)
    
    def record_node_spike(self, node_id, time_step, spike_tensor):
        self.stdp_optimizer.record_node_spike(node_id, time_step, spike_tensor)
    
    def record_gradient(self, grad_alpha):
        """记录架构梯度（由 Hybrid 训练循环调用）"""
        self.gradient_buffer = grad_alpha.clone()
    
    def reset_traces(self):
        self.stdp_optimizer.reset_traces()
    
    def step(self, **kwargs):
        """先 STDP 更新，再叠加梯度更新"""
        self.stdp_optimizer.step(**kwargs)
        
        if self.gradient_buffer is not None:
            k, num_ops = self.alphas[0].shape
            with torch.no_grad():
                for edge_idx in range(k):
                    for op_idx in range(num_ops):
                        delta = -self.gradient_lr * self.gradient_buffer[0, edge_idx, op_idx].item()
                        contrib = (1 - self.stdp_weight) * delta
                        self.alphas[0, edge_idx, op_idx] += contrib
                        self.alphas[1, edge_idx, op_idx] += contrib
                        self.alphas[0, edge_idx, op_idx].clamp_(0, 1)
                        self.alphas[1, edge_idx, op_idx].clamp_(0, 1)
            self.gradient_buffer = None
    
    def get_statistics(self):
        stats = self.stdp_optimizer.get_statistics()
        stats['stdp_weight'] = self.stdp_weight
        stats['gradient_lr'] = self.gradient_lr
        return stats
    
    def state_dict(self):
        return {
            'stdp_weight': self.stdp_weight,
            'gradient_lr': self.gradient_lr,
            'stdp_optimizer': self.stdp_optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        self.stdp_weight = state_dict.get('stdp_weight', self.stdp_weight)
        self.gradient_lr = state_dict.get('gradient_lr', self.gradient_lr)
        if 'stdp_optimizer' in state_dict and state_dict['stdp_optimizer'] is not None:
            self.stdp_optimizer.load_state_dict(state_dict['stdp_optimizer'])




