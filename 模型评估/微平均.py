import json
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize

# 设置 Matplotlib 中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 适用于 Windows，Linux 可能需要改为 "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

# 文件路径
GROUND_TRUTH_FILE = r"E:\python_datas\new_paper\Dissertation\sparkMax-32\模型评估\格式转换\output_relations.txt"
PREDICTIONS_FILE = r"E:\python_datas\new_paper\Dissertation\result\root\autodl-tmp\roberta_zh_L-6-H-768_A-12\predictions1.txt"


# 读取人工标注数据集
def load_ground_truth(filepath):
    ground_truth = defaultdict(set)
    pattern = re.compile(r"Doc ID:\s*(\d+),\s*Event 1 ID:\s*(\d+_\d+),\s*Event 2 ID:\s*(\d+_\d+),\s*Relation:\s*(.*)")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                doc_id, event1, event2, relation = m.groups()
                relation = relation.strip()
                if relation == "" or relation.lower() == "none":
                    relation = "其它关系"
                elif relation == "因果":
                    relation = "因果关系"
                elif relation == "时序":
                    relation = "时序关系"
                ground_truth[doc_id].add((event1, event2, relation))
    return ground_truth

# 读取预测数据集
def load_predictions(filepath):
    return load_ground_truth(filepath)  # 预测集格式相同，复用解析函数

# 加载数据
ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
predictions = load_predictions(PREDICTIONS_FILE)

# 统一标签
def choose_label(label_set):
    if "因果关系" in label_set:
        return "因果关系"
    elif "时序关系" in label_set:
        return "时序关系"
    else:
        return "其它关系"

# 生成评估数据
all_y_true, all_y_pred = [], []
all_doc_ids = set(ground_truth.keys()) | set(predictions.keys())

for doc_id in all_doc_ids:
    gt_dict, pred_dict = defaultdict(set), defaultdict(set)
    for (e1, e2, rel) in ground_truth.get(doc_id, set()):
        gt_dict[(e1, e2)].add(rel)
    for (e1, e2, rel) in predictions.get(doc_id, set()):
        pred_dict[(e1, e2)].add(rel)
    all_pairs = set(gt_dict.keys()) | set(pred_dict.keys())
    for pair in all_pairs:
        all_y_true.append(choose_label(gt_dict.get(pair, {"其它关系"})))
        all_y_pred.append(choose_label(pred_dict.get(pair, {"其它关系"})))

# 获取类别标签
classes = sorted(list(set(all_y_true) | set(all_y_pred)))

# 计算指标
accuracy = accuracy_score(all_y_true, all_y_pred)
precision = precision_score(all_y_true, all_y_pred, average="macro", zero_division=0)
recall = recall_score(all_y_true, all_y_pred, average="macro", zero_division=0)
f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
cm = confusion_matrix(all_y_true, all_y_pred, labels=classes)

# 绘制混淆矩阵
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("混淆矩阵")
plt.show()

# 计算 ROC-AUC
label_to_index = {label: i for i, label in enumerate(classes)}
y_true_num = np.array([label_to_index[label] for label in all_y_true])
y_pred_num = np.array([label_to_index[label] for label in all_y_pred])
y_true_bin = label_binarize(y_true_num, classes=range(len(classes)))
y_pred_bin = label_binarize(y_pred_num, classes=range(len(classes)))

plt.figure(figsize=(8, 6))
fpr, tpr, roc_auc = {}, {}, {}
for i, cls in enumerate(classes):
    fpr[cls], tpr[cls], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[cls] = auc(fpr[cls], tpr[cls])
    plt.plot(fpr[cls], tpr[cls], lw=2, label=f"{cls} (AUC = {roc_auc[cls]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2, label="随机分类 (AUC = 0.5)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("假阳性率")
plt.ylabel("真阳性率")
plt.title("多分类 AUC-ROC 曲线")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 计算宏平均 AUC 和微平均 AUC
try:
    macro_auc = roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovr")
    micro_auc = roc_auc_score(y_true_bin, y_pred_bin, average="micro", multi_class="ovr")
except Exception as e:
    macro_auc, micro_auc = None, None

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
if macro_auc is not None:
    print(f"宏平均 AUC: {macro_auc:.4f}")
    print(f"微平均 AUC: {micro_auc:.4f}")
else:
    print("无法计算 AUC")
