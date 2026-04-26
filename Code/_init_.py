"""
- parse: 架构权重解析函数
- edge_num: 每个节点的边数
- forward_edge_num: 总前向边数
"""

import numpy as np

# =============================================================================
# 边数量定义
# =============================================================================

# 每个节点的边数
# Node 1: 2个输入 × 2个神经元 = 4条边
# Node 2: 3个输入 × 2个神经元 = 6条边
# Node 3: 4个输入 × 2个神经元 = 8条边
edge_num = [2, 3, 4]

# 总前向边数
forward_edge_num = sum(edge_num)  # 2 + 3 + 4 = 9


# =============================================================================
# 架构解析函数
# =============================================================================

def parse(weights, primitives, op_threshold=None, parse_method='bio_darts', 
          steps=3, reduction=False, back_connection=False):
    """
    解析架构权重，生成基因型
    
    Args:
        weights: 架构权重矩阵 [k, num_ops]，numpy array
        primitives: 操作列表，如 ['conv_3x3_p', 'conv_5x5_p', ...]
        op_threshold: 操作阈值，只选择权重大于此值的操作（如0.85）
        parse_method: 解析方法 ('bio_darts', 'darts', 'threshold'等)
        steps: 节点数（默认3）
        reduction: 是否为reduction cell
        back_connection: 是否有反向连接
    
    Returns:
        gene: 基因型列表 [(operation_name, edge_index), ...]
        
    Example:
        weights = np.array([
            [0.05, 0.02, 0.85, 0.08],  # Edge 0
            [0.10, 0.70, 0.15, 0.05],  # Edge 1
        ])
        primitives = ['conv_3x3_p', 'conv_5x5_p', 'sep_conv_p', 'skip_p']
        
        gene = parse(weights, primitives, op_threshold=0.5)
        # 返回: [('sep_conv_p', 0), ('conv_5x5_p', 1)]
    """
    gene = []
    
    if parse_method == 'bio_darts' or parse_method == 'threshold':
        # 基于阈值的解析方法
        for edge_idx in range(weights.shape[0]):
            edge_weights = weights[edge_idx]
            
            # 找到权重最大的操作
            best_op_idx = np.argmax(edge_weights)
            best_weight = edge_weights[best_op_idx]
            best_op = primitives[best_op_idx]
            
            # 如果设置了阈值，只选择超过阈值的操作
            if op_threshold is None or best_weight >= op_threshold:
                gene.append((best_op, edge_idx))
            
    elif parse_method == 'darts':
        # 标准DARTS解析方法（每个节点选择top-k操作）
        offset = 0
        
        for node_idx in range(steps):
            # 计算该节点的输入边数
            num_inputs = 2 + node_idx  # Node 0: 2输入, Node 1: 3输入, ...
            
            # 获取该节点的所有边的权重
            node_edges = []
            for i in range(num_inputs):
                edge_weights = weights[offset + i]
                best_op_idx = np.argmax(edge_weights)
                best_weight = edge_weights[best_op_idx]
                node_edges.append((best_weight, best_op_idx, offset + i))
            
            # 对该节点的边按权重排序，选择top-2
            node_edges.sort(reverse=True, key=lambda x: x[0])
            
            for weight, op_idx, edge_idx in node_edges[:2]:  # top-2
                if op_threshold is None or weight >= op_threshold:
                    gene.append((primitives[op_idx], edge_idx))
            
            offset += num_inputs
    
    elif parse_method == 'all':
        # 选择所有边的最优操作（不考虑阈值）
        for edge_idx in range(weights.shape[0]):
            edge_weights = weights[edge_idx]
            best_op_idx = np.argmax(edge_weights)
            best_op = primitives[best_op_idx]
            gene.append((best_op, edge_idx))
    
    else:
        # 默认方法：与bio_darts相同
        for edge_idx in range(weights.shape[0]):
            edge_weights = weights[edge_idx]
            best_op_idx = np.argmax(edge_weights)
            best_weight = edge_weights[best_op_idx]
            best_op = primitives[best_op_idx]
            
            if op_threshold is None or best_weight >= op_threshold:
                gene.append((best_op, edge_idx))
    
    return gene


# =============================================================================
# 辅助函数
# =============================================================================

def format_genotype(genotype, primitives=None):
    """
    格式化基因型，便于可读性
    
    Args:
        genotype: Genotype对象
        primitives: 操作列表（可选）
    
    Returns:
        str: 格式化的字符串
    """
    lines = []
    lines.append("Genotype(")
    lines.append("  normal=[")
    
    for op, edge in genotype.normal:
        lines.append(f"    ('{op}', {edge}),")
    
    lines.append("  ],")
    lines.append(f"  normal_concat={list(genotype.normal_concat)}")
    lines.append(")")
    
    return '\n'.join(lines)


def count_operations(genotype):
    """
    统计基因型中每种操作的使用次数
    
    Args:
        genotype: Genotype对象
    
    Returns:
        dict: {operation_name: count}
    """
    from collections import Counter
    
    op_counts = Counter()
    for op, edge in genotype.normal:
        op_counts[op] += 1
    
    return dict(op_counts)


def analyze_node_types(genotype):
    """
    分析基因型中的节点类型
    
    根据选中的操作（_p或_n），推断每个节点的类型
    
    Args:
        genotype: Genotype对象
    
    Returns:
        dict: 节点类型分析结果
    """
    # 边分配：
    # Node 1: edge 0-3 (neuron1: 0-1, neuron2: 2-3)
    # Node 2: edge 4-9 (neuron1: 4-6, neuron2: 7-9)
    # Node 3: edge 10-17 (neuron1: 10-13, neuron2: 14-17)
    
    node_configs = [
        ('Node1', 0, 2, 2, 4),
        ('Node2', 4, 7, 7, 10),
        ('Node3', 10, 14, 14, 18)
    ]
    
    node_types = {}
    
    for node_name, n1_start, n1_end, n2_start, n2_end in node_configs:
        # 分析神经元1的操作
        n1_ops = [op for op, edge in genotype.normal if n1_start <= edge < n1_end]
        n1_type = 'E' if any('_p' in op for op in n1_ops) else 'I'
        
        # 分析神经元2的操作
        n2_ops = [op for op, edge in genotype.normal if n2_start <= edge < n2_end]
        n2_type = 'E' if any('_p' in op for op in n2_ops) else 'I'
        
        node_types[node_name] = f"{n1_type}-{n2_type}"
    
    return node_types


# =============================================================================
# 模块导出
# =============================================================================

__all__ = [
    'parse',
    'edge_num',
    'forward_edge_num',
    'format_genotype',
    'count_operations',
    'analyze_node_types'
]
