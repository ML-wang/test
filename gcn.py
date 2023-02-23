from torch_geometric.datasets import Planetoid
import torch


# 第一次需要翻墙下载、不然会报错，只需要下载一次
data = Planetoid(root='/data/CiteSeer', name='CiteSeer')
'''
Citeseer网络是一个引文网络，节点为论文，一共3327篇论文。
论文一共分为六类：Agents、AI（人工智能）、DB（数据库）、IR（信息检索）、ML（机器语言）和HCI。
如果两篇论文间存在引用关系，那么它们之间就存在链接关系
'''
print(len(data))  # 该数据集中只有一个网络,输出结果为1
data = data[0]    # 取出数据集
print(data)
'''
Data(x=[3327, 3703], edge_index=[2, 9104], y=[3327], train_mask=[3327], val_mask=[3327], test_mask=[3327])
x=[3327, 3703] 表示一共有3327个节点，然后节点的特征维度为3703
edge_index=[2, 9104]，表示一共9104条edge。数据一共两行，每一行都表示节点编号。
y=[3327] 表示


'''
print(data.x)
'''
节点特征矩阵，shape为[num_nodes, num_node_features]，Tensor类型
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
'''
print(data.y)
'''
图级标签或节点级标签，Tensor类型
tensor([3, 1, 5,  ..., 3, 1, 5])
'''

print(data.edge_index)
'''
tensor([[   0,    1,    1,  ..., 3324, 3325, 3326],
        [ 628,  158,  486,  ..., 2820, 1643,   33]])
'''
print(data.edge_attr)
print(data.pos)
# print(data.num_nodes)  # 输出结果为节点数量3327
# print(data.num_edges)  # 输出结果为边的个数9104
# print(data.num_node_features)  # 输出结果为每个节点的特征维度，每个节点表示为3703维向量
# print(data.has_isolated_nodes())  # 该图中是否存在独立节点，存在
# print(data.has_self_loops())  # 该节点是否存在闭环
# print(data.is_directed())  # 判断该数据集是否是无向图





















