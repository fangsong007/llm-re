import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 初始化画布
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_facecolor('#f9f9f9')

# 输入层
input_layer = plt.Rectangle((0.1, 0.7), 1.2, 0.5, color='#e0f0ff', edgecolor='black', linewidth=2)
ax.add_patch(input_layer)
ax.text(0.7, 0.95, "输入层", fontsize=12, ha='center')
ax.text(0.7, 0.8, "文本标记（[E1]/[E1]等）", fontsize=10, ha='center')

# BERT 编码器
encoder = plt.Rectangle((1.4, 0.7), 2.0, 0.5, color='#c0e0ff', edgecolor='black', linewidth=2)
ax.add_patch(encoder)
ax.text(2.4, 0.95, "BERT 编码器", fontsize=12, ha='center')
ax.text(2.4, 0.8, "多层 Transformer", fontsize=10, ha='center')

# 分类头
classifier = plt.Rectangle((3.6, 0.7), 1.2, 0.5, color='#a0d0ff', edgecolor='black', linewidth=2)
ax.add_patch(classifier)
ax.text(4.2, 0.95, "分类头", fontsize=12, ha='center')
ax.text(4.2, 0.8, "线性层 + 激活函数", fontsize=10, ha='center')

# 输出标签
output = plt.Rectangle((4.9, 0.7), 0.8, 0.5, color='#80c0ff', edgecolor='black', linewidth=2)
ax.add_patch(output)
ax.text(5.3, 0.95, "输出", fontsize=12, ha='center')
ax.text(5.3, 0.8, "关系类型标签", fontsize=10, ha='center')

# 箭头连接
ax.annotate('', xy=(1.3, 0.95), xytext=(1.4, 0.95),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center', va='center')

ax.annotate('', xy=(3.5, 0.95), xytext=(3.6, 0.95),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center', va='center')

ax.annotate('', xy=(4.8, 0.95), xytext=(4.9, 0.95),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center', va='center')

# 隐藏坐标轴
ax.axis('off')
plt.show()