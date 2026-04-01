import numpy as np
from graphviz import Digraph

file_path = 'exps\\best_gen_2.npy'
genotype = np.load(file_path)
print("数组内容:\n", genotype)


# 定义操作集
PRIMITIVES = [
    'none', 'skip_connect', 'conv_1x1', 'conv_3x3',
    'conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'
]

PRIMITIVES_up = [
    'nearest', 'bilinear', 'ConvTranspose'
]

# 创建绘图函数
def draw_graph_G(genotype, save=True, file_path='generator_arch'):
    num_cell = genotype.shape[0]
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', 
                       fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot'
    )
    g.body.extend(['rankdir=LR'])
    
    # 创建节点 (共13个节点)
    g.node('0', fillcolor='darkseagreen2')  # 输入节点
    for i in range(1, 12):
        g.node(str(i), fillcolor='lightblue')  # 中间节点
    g.node('12', fillcolor='palegoldenrod')  # 输出节点

    # 绘制每个细胞的边
    for cell_i in range(num_cell):
        ops = []
        for edge_i in range(7):
            if edge_i < 2:  # 前2条边使用上采样操作
                ops.append(PRIMITIVES_up[genotype[cell_i][edge_i]])
            else:  # 后5条边使用常规操作
                ops.append(PRIMITIVES[genotype[cell_i][edge_i]])
        
        # 添加细胞内部的7条边
        base = 4 * cell_i
        connections = [
            (f"{base}", f"{base+1}", ops[0]),
            (f"{base}", f"{base+2}", ops[1]),
            (f"{base+1}", f"{base+3}", ops[2]),
            (f"{base+2}", f"{base+3}", ops[3]),
            (f"{base+1}", f"{base+4}", ops[4]),
            (f"{base+2}", f"{base+4}", ops[5]),
            (f"{base+3}", f"{base+4}", ops[6])
        ]
        
        for src, dst, label in connections:
            g.edge(src, dst, label=label, fillcolor='gray')

    # 添加特殊连接（跨细胞连接）
    g.edge('3', '7', label='bilinear', fillcolor='gray')
    g.edge('3', '11', label='nearest', fillcolor='gray')
    g.edge('7', '11', label='nearest', fillcolor='gray')

    # 保存并显示结果
    if save:
        g.render(file_path, view=True)
        print(f"架构图已保存至: {file_path}.pdf")
    return g

# 生成架构图
graph = draw_graph_G(genotype)