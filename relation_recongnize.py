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
    示例数据结构如下：
    [
        {
            "news-ID": "15964",
            "doc": "文章内容……",
            "events": [
                {"id": "15964_1", "event-information": {"trigger": [{"text": "出售", "start": 341, "end":343}], "event_type": "财经/交易_出售"}},
                {"id": "15964_2", "event-information": {"trigger": [{"text": "拘留", "start": 171, "end":173}], "event_type": "司法行为_拘捕"}}
            ]
        },
   ...
    ]
    会对每篇文章中所有事件两两组合生成样本，
    并在文本中将两个事件触发词分别用 [E1] [/E1] 和 [E2] [/E2] 标记。
    """

    def __init__(self, file_path, tokenizer, max_length):
        self.samples = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件 {file_path} 不存在，请检查路径是否正确。")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc in data:
            doc_id = doc["news-ID"]
            text = doc["doc"]
            events = doc["events"]
            for i in range(len(events)):
                for j in range(i + 1, len(events)):
                    event1 = events[i]
                    event2 = events[j]
                    e1_trigger = event1["event-information"]["trigger"][0]["text"]
                    e2_trigger = event2["event-information"]["trigger"][0]["text"]
                    # 改进触发词标记方法，通过位置信息准确替换，避免重复词替换问题
                    e1_start = event1["event-information"]["trigger"][0]["start"]
                    e1_end = event1["event-information"]["trigger"][0]["end"]
                    e2_start = event2["event-information"]["trigger"][0]["start"]
                    e2_end = event2["event-information"]["trigger"][0]["end"]
                    marked_text = text[:e1_start] + f"[E1]{e1_trigger}[/E1]" + text[e1_end:]
                    marked_text = marked_text[:e2_start] + f"[E2]{e2_trigger}[/E2]" + marked_text[e2_end:]
                    # 这里目前默认关系标签设为0 (None)，实际项目中需使用正确标注
                    # 可以考虑后续添加从数据中读取真实标签的逻辑
                    label = 0
                    self.samples.append({
                        "doc_id": doc_id,
                        "event1_id": event1["id"],
                        "event2_id": event2["id"],
                        "input_text": marked_text,
                        "label": label
                    })
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        item = {key: val.squeeze(0).to(device) for key, val in encoding.items()}  # 将编码后的张量都移动到device设备上
        item["label"] = torch.tensor(sample["label"], dtype=torch.long).to(device)  # 把标签张量也移动到device设备上
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
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for batch in train_loader:
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
        avg_loss = total_loss / len(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")
    return model


# ========================== 预测函数 ==========================
def predict(model, test_dataset, device, batch_size=2):
    """
    使用训练好的模型进行预测的函数，在不计算梯度的情况下，对测试数据集进行预测并返回预测结果。
    """
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            doc_ids = batch["doc_id"]
            event1_ids = batch["event1_id"]
            event2_ids = batch["event2_id"]
            for i, pred in enumerate(preds):
                predictions.append({
                    "doc_id": doc_ids[i],
                    "event1_id": event1_ids[i],
                    "event2_id": event2_ids[i],
                    "relation": pred
                })
    return predictions


# ========================== 主函数 ==========================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "train_split.json")
    output_file = os.path.join(script_dir, "predictions.txt")
    epochs = 3
    mode = "train"  # train ，若要预测，将 mode 改为 "predict"

    # 定义模型目录的绝对路径（确保该目录下有相关模型文件），这里明确指向vocab.txt所在的正确目录
    model_dir = "/root/autodl-tmp/roberta_zh_L-6-H-768_A-12"

    # 检查vocab.txt文件是否存在，使用正确的绝对路径
    vocab_path = os.path.join(model_dir, "vocab.txt")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab.txt文件不存在于 {model_dir} 目录下，请检查模型文件完整性。")

    # 检查config.json文件是否存在，同样使用正确的绝对路径
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json文件不存在于 {model_dir} 目录下，请检查模型文件完整性。")

    # 使用BertTokenizer从本地加载词表，用config.json初始化
    tokenizer = BertTokenizer(vocab_path, config=config_path)
    # 添加事件特殊标记
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用BertForSequenceClassification加载模型权重
    model = BertForSequenceClassification.from_pretrained(
        model_dir, 
        num_labels=3
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # 使用模型配置中的最大位置嵌入数作为 max_length，以防输入超过模型支持范围
    max_length = model.config.max_position_embeddings
    print(f"Using max_length = {max_length}")

    if mode == "train":
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"训练数据集文件 {data_file} 不存在，请检查路径是否正确。")
        train_dataset = EventRelationDataset(data_file, tokenizer, max_length=max_length)
        model = train(model, train_dataset, device, epochs=epochs, batch_size=2)
        model.save_pretrained("event_relation_model")
        tokenizer.save_pretrained("event_relation_model")
        print("训练完成，模型已保存。")
    elif mode == "predict":
        pred_model_dir = os.path.join(script_dir, "event_relation_model")
        pred_model_dir = os.path.abspath(pred_model_dir)
        if os.path.exists(pred_model_dir):
            model = BertForSequenceClassification.from_pretrained(
                pred_model_dir, 
                num_labels=3
            )
            tokenizer = BertTokenizer.from_pretrained(
                pred_model_dir, 
                config=os.path.join(pred_model_dir, "config.json"),
                vocab_file=os.path.join(pred_model_dir, "vocab.txt")
            )
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"测试数据集文件 {data_file} 不存在，请检查路径是否正确。")
            test_dataset = EventRelationDataset(data_file, tokenizer, max_length=max_length)
            predictions = predict(model, test_dataset, device, batch_size=2)
            label_map = {0: "None", 1: "因果关系", 2: "时序关系"}
            output_lines = [f"({pred['event1_id']},{pred['event2_id']},{label_map[pred['relation']]})" for pred in predictions]
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(",".join(output_lines))
            print(f"预测结果已保存到 {output_file}")


if __name__ == "__main__":
    main()