import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def read_predicted_data(file_path):
    """
    从文本文件中读取新格式的预测数据集
    :param file_path: 预测数据集文件路径
    :return: 预测数据集列表
    """
    predicted_data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split(', ')
                doc_id = parts[0].split(': ')[1]
                event1_id = parts[1].split(': ')[1]
                event2_id = parts[2].split(': ')[1]
                relation = parts[3].split(': ')[1]
                if relation == "":
                    relation = "无关系"
                elif relation == "因果":
                    relation = "因果关系"
                elif relation == "时序":
                    relation = "时序关系"
                if doc_id not in predicted_data:
                    predicted_data[doc_id] = []
                predicted_data[doc_id].append((event1_id, event2_id, relation))
            except IndexError as e:
                print(f"第 {line_num} 行数据格式有误: {line.strip()}，错误信息: {e}")
    return [{"news-ID": k, "output": "\n".join([f"（{e1}，{e2}，{r}）" for e1, e2, r in v])} for k, v in predicted_data.items()]


def read_annotated_data(file_path):
    """
    从 JSON 文件中读取人工标注数据集
    :param file_path: 人工标注数据集文件路径
    :return: 人工标注数据集列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}，请检查 {file_path} 文件内容是否为有效的 JSON 格式。")
        return []
    except FileNotFoundError:
        print(f"文件未找到: {file_path}，请检查文件路径是否正确。")
        return []


def parse_predicted_relations(predicted_data):
    """
    解析预测数据中的事件关系
    :param predicted_data: 预测数据集
    :return: 以新闻 ID 为键的事件关系字典
    """
    predicted_relations = {}
    for item in predicted_data:
        news_id = item["news-ID"]
        outputs = item["output"].split('\n')
        relations = []
        for output in outputs:
            if output:
                try:
                    parts = output.strip('（）').split('，')
                    if len(parts) != 3:
                        print(f"数据格式错误，新闻 ID: {news_id}，输出内容: {output}")
                        continue
                    event1 = parts[0]
                    event2 = parts[1]
                    relation_type = parts[2]
                    relations.append((event1, event2, relation_type))
                except Exception as e:
                    print(f"解析数据时出错，新闻 ID: {news_id}，输出内容: {output}，错误信息: {e}")
        if not relations:
            relations = [("all", "all", "无关系")]
        predicted_relations[news_id] = relations
    return predicted_relations


def parse_annotated_relations(annotated_data):
    """
    解析人工标注数据中的事件关系
    :param annotated_data: 人工标注数据集
    :return: 以新闻 ID 为键的事件关系字典
    """
    annotated_relations = {}
    for item in annotated_data:
        news_id = item["news-ID"]
        relations = []
        for relation in item["relations"]:
            event1 = relation["one_event"]["id"]
            event2 = relation["other_event"]["id"]
            if relation["relation_type"] == "时序":
                relation_type = "时序关系"
            elif relation["relation_type"] == "因果":
                relation_type = "因果关系"
            relations.append((event1, event2, relation_type))
        if not relations:
            relations = [("all", "all", "无关系")]
        annotated_relations[news_id] = relations
    return annotated_relations


def extract_labels(predicted_relations, annotated_relations):
    """
    提取真实标签和预测标签
    :param predicted_relations: 预测的事件关系字典
    :param annotated_relations: 人工标注的事件关系字典
    :return: 真实标签列表和预测标签列表
    """
    all_news_ids = set(predicted_relations.keys()) | set(annotated_relations.keys())
    true_labels = []
    pred_labels = []
    for news_id in all_news_ids:
        pred = predicted_relations.get(news_id, [("all", "all", "无关系")])
        true = annotated_relations.get(news_id, [("all", "all", "无关系")])
        max_len = max(len(pred), len(true))
        for i in range(max_len):
            pred_label = pred[i][2] if i < len(pred) else "无关系"
            true_label = true[i][2] if i < len(true) else "无关系"
            true_labels.append(true_label)
            pred_labels.append(pred_label)
    return true_labels, pred_labels


def evaluate_model(true_labels, pred_labels):
    """
    评估模型性能
    :param true_labels: 真实标签列表
    :param pred_labels: 预测标签列表
    :return: 每个类别的准确率、召回率、F1 值、混淆矩阵以及总体的准确率、召回率和 F1 值
    """
    labels = ["时序关系", "因果关系", "无关系"]
    precision = precision_score(true_labels, pred_labels, average=None, labels=labels)
    recall = recall_score(true_labels, pred_labels, average=None, labels=labels)
    f1 = f1_score(true_labels, pred_labels, average=None, labels=labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=labels)
    overall_precision = precision_score(true_labels, pred_labels, average='weighted')
    overall_recall = recall_score(true_labels, pred_labels, average='weighted')
    overall_f1 = f1_score(true_labels, pred_labels, average='weighted')
    return precision, recall, f1, conf_matrix, overall_precision, overall_recall, overall_f1


def print_evaluation_results(labels, precision, recall, f1, conf_matrix, overall_precision, overall_recall, overall_f1):
    """
    打印评估结果
    :param labels: 类别标签列表
    :param precision: 准确率数组
    :param recall: 召回率数组
    :param f1: F1 值数组
    :param conf_matrix: 混淆矩阵
    :param overall_precision: 总体准确率
    :param overall_recall: 总体召回率
    :param overall_f1: 总体 F1 值
    """
    print("每个类别的评估结果：")
    for i, label in enumerate(labels):
        print(f"{label}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-score: {f1[i]:.4f}")

    print("\n混淆矩阵：")
    print(" " * 10 + " ".join([f"{label: <10}" for label in labels]))
    for i, row in enumerate(conf_matrix):
        print(f"{labels[i]: <10}" + " ".join([f"{val: <10}" for val in row]))

    print("\n模型总体评估结果：")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1-score: {overall_f1:.4f}")


if __name__ == "__main__":
    # 替换为实际的文件路径
    predicted_file_path = r'E:\python_datas\new_paper\Dissertation\result\root\autodl-tmp\roberta_zh_L-6-H-768_A-12\predictions.txt'
    annotated_file_path = r'E:\python_datas\new_paper\Dissertation\result\root\autodl-tmp\roberta_zh_L-6-H-768_A-12\test_split.json'

    # 读取数据
    predicted_data = read_predicted_data(predicted_file_path)
    annotated_data = read_annotated_data(annotated_file_path)

    if not predicted_data or not annotated_data:
        print("数据读取失败，请检查文件内容和路径。")
    else:
        # 解析数据
        predicted_relations = parse_predicted_relations(predicted_data)
        annotated_relations = parse_annotated_relations(annotated_data)

        # 提取标签
        true_labels, pred_labels = extract_labels(predicted_relations, annotated_relations)

        # 评估模型
        precision, recall, f1, conf_matrix, overall_precision, overall_recall, overall_f1 = evaluate_model(true_labels,
                                                                                                          pred_labels)

        # 计算事件关系总数量
        predicted_relation_count = sum([len(rels) for rels in predicted_relations.values()])
        annotated_relation_count = sum([len(rels) for rels in annotated_relations.values()])

        print(f"预测数据集中识别出的事件关系总数量: {predicted_relation_count}")
        print(f"人工标注数据集中的事件关系总数量: {annotated_relation_count}")

        # 打印结果
        labels = ["时序关系", "因果关系", "无关系"]
        print_evaluation_results(labels, precision, recall, f1, conf_matrix, overall_precision, overall_recall,
                                 overall_f1)