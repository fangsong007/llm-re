from graphviz import Digraph

dot = Digraph(format="svg")
dot.attr(rankdir="TB", fontname="SimHei")  # 设置字体
dot.node("Input", "输入层\n(Token IDs, Attention Mask)", fontname="SimHei")
dot.node("Embedding", "Embedding\n(词向量 + 位置编码)", fontname="SimHei")
dot.node("Encoder", "BERT Encoder\n(Transformer 层)", fontname="SimHei")
dot.node("Classifier", "分类器\n(最终分类)", fontname="SimHei")

# 连接
dot.edge("Input", "Embedding")
dot.edge("Embedding", "Encoder")
dot.edge("Encoder", "Classifier")
dot.edge("Classifier", "Output", label="关系预测", fontname="SimHei")

# 渲染
dot.render("简单结构", view=True)
