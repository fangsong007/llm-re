import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 解决 matplotlib 显示中文乱码问题
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 适用于 Windows（黑体）
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 定义数据集文件路径
ground_truth_path = r"E:\python_datas\new_paper\Dissertation\result\root\autodl-tmp\roberta_zh_L-6-H-768_A-12\模型结构可视化\output_relations.txt"
prediction_path = r"E:\python_datas\new_paper\Dissertation\result\root\autodl-tmp\roberta_zh_L-6-H-768_A-12\predictions2.txt"


def parse_relations(file_path):
    relations = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(", ")
                if len(parts) < 4:
                    continue
                try:
                    doc_id = parts[0].split(": ")[1]
                    event1_id = parts[1].split(": ")[1]
                    event2_id = parts[2].split(": ")[1]
                    relation = parts[3].split(": ")[1] if len(parts[3].split(": ")) > 1 else "其它"

                    if doc_id not in relations:
                        relations[doc_id] = {}
                    relations[doc_id][(event1_id, event2_id)] = relation
                except IndexError:
                    continue
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查文件路径是否正确。")
    return relations

# 解析人工标注数据
true_relations = parse_relations(ground_truth_path)

# 解析模型预测数据
predicted_relations = parse_relations(prediction_path)

# 统计人工标注的事件关系和模型预测的事件关系
true_labels = []
pred_labels = []

all_doc_ids = set(true_relations.keys()) | set(predicted_relations.keys())
for doc_id in all_doc_ids:
    true_doc_relations = true_relations.get(doc_id, {})
    pred_doc_relations = predicted_relations.get(doc_id, {})
    all_event_pairs = set(true_doc_relations.keys()) | set(pred_doc_relations.keys())
    for event_pair in all_event_pairs:
        true_relation = true_doc_relations.get(event_pair, "其它")
        predicted_relation = pred_doc_relations.get(event_pair, "其它")
        true_labels.append(true_relation)
        pred_labels.append(predicted_relation)

# 计算混淆矩阵
labels = ["时序", "因果", "其它"]
cm = confusion_matrix(true_labels, pred_labels, labels=labels)

# 绘制混淆矩阵
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("预测关系", fontsize=12)
plt.ylabel("真实关系", fontsize=12)
plt.title("混淆矩阵", fontsize=14)
plt.show()

# 计算分类报告
report = classification_report(true_labels, pred_labels, target_names=labels, output_dict=True)
overall_accuracy = accuracy_score(true_labels, pred_labels)  # 计算整体 accuracy

# 绘制柱状图
metrics = ["accuracy", "precision", "recall", "f1-score"]
values = {
    "accuracy": [overall_accuracy] * len(labels),  # 让 accuracy 在每个类别上显示
    "precision": [report[label]["precision"] for label in labels],
    "recall": [report[label]["recall"] for label in labels],
    "f1-score": [report[label]["f1-score"] for label in labels],
}

x = np.arange(len(labels))
width = 0.2

# 自定义颜色
colors = [(115 / 255, 186 / 255, 214 / 255),  # 蓝色
          (2 / 255, 38 / 255, 62 / 255),  # 绿色
          (239 / 255, 65 / 255, 67 / 255),  # 红色
          (191 / 255, 30 / 255, 46 / 255)]  # 紫色

fig, ax = plt.subplots(figsize=(8, 5))
for i, metric in enumerate(metrics):
    bars = ax.bar(x + i * width, values[metric], width, label=metric.capitalize(), color=colors[i])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.3f}',
                ha='center', va='bottom')

ax.set_xlabel("关系类型")
ax.set_ylabel("分数")
ax.set_title("评估指标对比")
ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.legend()  # 调整图例位置，避免遮挡数据

# 保存柱状图
# plt.savefig("柱状图1.png", dpi=300, bbox_inches="tight")
plt.show()

# 统计人工标注的关系数量
print("人工标注的事件关系总数量:", len(true_labels))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 解决 matplotlib 显示中文乱码问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def parse_relations(file_path, is_ground_truth=False):
    relations = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 规范化解码流程
            parts = [p.strip() for p in line.strip().split(", ")]
            if len(parts) < 4:
                continue

            # 统一提取字段
            doc_id = parts[0].split(": ")[1]
            e1 = parts[1].split(": ")[1]
            e2 = parts[2].split(": ")[1]

            # 处理关系字段
            rel_field = parts[3].split(": ", 1)
            relation = rel_field[1] if len(rel_field) > 1 else ""

            # 特殊处理标注数据中的空关系
            if is_ground_truth:
                relation = "其它" if relation == "" else relation
            else:
                relation = relation if relation else "其它"

            # 规范事件对存储顺序（小ID在前）
            sorted_pair = tuple(sorted((e1, e2), key=lambda x: int(x.split('_')[1])))
            if doc_id not in relations:
                relations[doc_id] = {}
            relations[doc_id][sorted_pair] = relation
    return relations


# 解析数据（注意标注数据需标记 is_ground_truth=True）
true_relations = parse_relations(ground_truth_path, is_ground_truth=True)
pred_relations = parse_relations(prediction_path)

# 精准对齐统计逻辑
true_labels, pred_labels = [], []
for doc_id in true_relations:  # 仅遍历标注数据中的文档
    true_doc = true_relations[doc_id]
    pred_doc = pred_relations.get(doc_id, {})

    # 仅处理标注数据中存在的事件对
    for event_pair, true_rel in true_doc.items():
        pred_rel = pred_doc.get(event_pair, "其它")
        true_labels.append(true_rel)
        pred_labels.append(pred_rel)

# 后续统计代码保持不变...
print("修正后真实事件关系数量:", len(true_labels))  # 应等于实际数量