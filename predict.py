import json
import random
import os
import torch
from torch.optim import AdamW
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt

# 定义关系类型到标签的映射
relation_type_mapping = {
    "因果": 0,
    "时序": 1,
    "": 2  # 新增无关系的映射
}

# 反转关系类型到标签的映射，用于将预测标签转换为关系类型
label_to_relation_mapping = {v: k for k, v in relation_type_mapping.items()}

# ========================== 设置随机种子 ==========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ========================== 数据集定义 ==========================
class EventRelationDataset(Dataset):
    """
    数据集文件为JSON格式，用于构建事件关系相关的数据集。
    会对每篇文章中所有事件两两组合生成样本，
    并在文本中将两个事件触发词分别用 [E1] [/E1] 和 [E2] [/E2] 标记。
    """

    def __init__(self, file_path, tokenizer, max_length, device):
        self.samples = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件 {file_path} 不存在，请检查路径是否正确。")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            print("无法确定具体错误位置对应的事件 ID，请手动检查文件中可能存在问题的部分。")
            return

        for doc in data:
            doc_id = doc.get("news-ID", "未知文档 ID")
            try:
                text = doc["doc"]
                events = doc["events"]
                relations = doc.get("relations", [])
                relation_dict = {}
                for relation in relations:
                    event1_id = relation["one_event"]["id"]
                    event2_id = relation["other_event"]["id"]
                    relation_type = relation["relation_type"]
                    relation_dict[(event1_id, event2_id)] = relation_type_mapping.get(relation_type, -1)

                for i in range(len(events)):
                    for j in range(i + 1, len(events)):
                        event1 = events[i]
                        event2 = events[j]
                        e1_trigger = event1["event-information"]["trigger"][0]["text"]
                        e2_trigger = event2["event-information"]["trigger"][0]["text"]
                        e1_start = event1["event-information"]["trigger"][0]["start"]
                        e1_end = event1["event-information"]["trigger"][0]["end"]
                        e2_start = event2["event-information"]["trigger"][0]["start"]
                        e2_end = event2["event-information"]["trigger"][0]["end"]
                        marked_text = text[:e1_start] + f"[E1]{e1_trigger}[/E1]" + text[e1_end:]
                        # 重新计算 e2 的位置
                        e2_start += len(f"[E1]{e1_trigger}[/E1]") - (e1_end - e1_start)
                        e2_end += len(f"[E1]{e1_trigger}[/E1]") - (e1_end - e1_start)
                        marked_text = marked_text[:e2_start] + f"[E2]{e2_trigger}[/E2]" + marked_text[e2_end:]

                        event1_id = event1["id"]
                        event2_id = event2["id"]
                        label = relation_dict.get((event1_id, event2_id), -1)
                        if label == -1:
                            # 如果没有对应的关系，默认设为无关系的标签
                            label = relation_type_mapping[""]

                        self.samples.append({
                            "doc_id": doc_id,
                            "event1_id": event1_id,
                            "event2_id": event2_id,
                            "input_text": marked_text,
                            "label": label
                        })
            except KeyError as e:
                print(f"文档 ID {doc_id} 中存在键错误: {e}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["input_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0).to(self.device) for key, val in encoding.items()}
        # 将标签的数据类型转换为 torch.long
        item["label"] = torch.tensor(sample["label"], dtype=torch.long).to(self.device)
        item["doc_id"] = sample["doc_id"]
        item["event1_id"] = sample["event1_id"]
        item["event2_id"] = sample["event2_id"]
        return item


# ========================== 训练函数 ==========================
def train(model, train_dataset, device, epochs=3, batch_size=2, learning_rate=2e-5):
    """
    训练模型的函数，执行多个轮次的训练过程，包括前向传播、计算损失、反向传播和参数更新等操作。
    """
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    epoch_losses = []  # 用于存储每个 epoch 的平均损失
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1} Training', unit='batch')
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Average Loss: {avg_loss:.4f}")
    return model, epoch_losses

# ========================== 预测函数 ==========================
def predict(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().tolist())
    return predictions

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "train_split.json")
    test_file = os.path.join(script_dir, "test_split.json")
    epochs = 3

    # 加载训练好的模型和分词器
    model_dir = "event_relation_model"
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用模型配置中的最大位置嵌入数作为 max_length，以防输入超过模型支持范围
    max_length = model.config.max_position_embeddings
    print(f"Using max_length = {max_length}")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试数据集文件 {test_file} 不存在，请检查路径是否正确。")
    test_dataset = EventRelationDataset(test_file, tokenizer, max_length, device)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # 进行预测
    predictions = predict(model, test_loader, device)

    # 将预测标签转换为关系类型
    relation_predictions = [label_to_relation_mapping[pred] for pred in predictions]

    # 保存预测结果到文件
    with open('predictions.txt', 'w') as f:
        for i, sample in enumerate(test_dataset.samples):
            doc_id = sample["doc_id"]
            event1_id = sample["event1_id"]
            event2_id = sample["event2_id"]
            relation = relation_predictions[i]
            f.write(f"Doc ID: {doc_id}, Event 1 ID: {event1_id}, Event 2 ID: {event2_id}, Relation: {relation}\n")

    print("预测完成，结果已保存到 predictions.txt。")