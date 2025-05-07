import json
from itertools import combinations


def convert_relations(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for doc in data:
        doc_id = doc["news-ID"]
        events = doc["events"]

        # 提取并排序事件ID（按数字后缀排序）
        event_ids = sorted(
            [event["id"] for event in events],
            key=lambda x: int(x.split('_')[1])
        )

        # 生成所有可能的事件对组合（不考虑顺序）
        for event1, event2 in combinations(event_ids, 2):
            relations = [
                rel for rel in doc["relations"]
                if {rel["one_event"]["id"], rel["other_event"]["id"]} == {event1, event2}
            ]

            if relations:
                # 存在关系时，每个关系生成一行
                for rel in relations:
                    results.append(
                        f"Doc ID: {doc_id}, "
                        f"Event 1 ID: {rel['one_event']['id']}, "
                        f"Event 2 ID: {rel['other_event']['id']}, "
                        f"Relation: {rel['relation_type']}"
                    )
            else:
                # 不存在关系时生成空关系行
                results.append(
                    f"Doc ID: {doc_id}, "
                    f"Event 1 ID: {event1}, "
                    f"Event 2 ID: {event2}, "
                    f"Relation: "
                )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))


# 使用示例
convert_relations(
    r"E:\python_datas\new_paper\Dissertation\result\root\autodl-tmp\roberta_zh_L-6-H-768_A-12\test_split.json",
    "output_relations.txt"
)