import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 解决 matplotlib 显示中文乱码问题
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 适用于 Windows（黑体）
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 读取人工标注数据
with open("test_split.json", "r", encoding="utf-8") as f:
    ground_truth_data = json.load(f)

# 读取模型预测数据
with open("predictions.txt", "r", encoding="utf-8") as f:
    predictions = f.readlines()

# 解析人工标注数据
true_relations = {}
for item in ground_truth_data:
    doc_id = item["news-ID"]
    true_relations[doc_id] = {}

    # 记录 ground truth 关系
    for relation in item.get("relations", []):
        event_pair = (relation["one_event"]["id"], relation["other_event"]["id"])
        true_relations[doc_id][event_pair] = relation["relation_type"]

    # 如果 relations 为空，则所有事件对的关系均为 "其它"
    if not item.get("relations", []):
        event_ids = [event["id"] for event in item["events"]]
        for i in range(len(event_ids)):
            for j in range(i + 1, len(event_ids)):
                true_relations[doc_id][(event_ids[i], event_ids[j])] = "其它"

# 解析模型预测数据
predicted_relations = {}
for line in predictions:
    parts = line.strip().split(", ")

    if len(parts) < 4:
        continue

    try:
        doc_id = parts[0].split(": ")[1]
        event1_id = parts[1].split(": ")[1]
        event2_id = parts[2].split(": ")[1]
        relation = parts[3].split(": ")[1] if ": " in parts[3] else "其它"

        if doc_id not in predicted_relations:
            predicted_relations[doc_id] = {}
        predicted_relations[doc_id][(event1_id, event2_id)] = relation

    except IndexError:
        continue

# 统计人工标注的事件关系和模型预测的事件关系
true_labels = []
pred_labels = []

for doc_id, relations in true_relations.items():
    for event_pair, true_relation in relations.items():
        predicted_relation = predicted_relations.get(doc_id, {}).get(event_pair, "其它")
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
colors = [(115/255, 186/255, 214/255),  # 蓝色
    (2/255, 38/255, 62/255),  # 绿色
    (239/255, 65/255, 67/255),   # 红色
    (191/255, 30/255, 46/255)  ]  # 蓝色, 绿色, 红色, 紫色

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
#plt.savefig("柱状图1.png", dpi=300, bbox_inches="tight")
plt.show()

# 统计人工标注的关系数量
print("人工标注的事件关系总数量:", len(true_labels))
