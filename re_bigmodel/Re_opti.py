import json
import websocket
import _thread as thread
from tqdm import tqdm
import time
import re
from urllib.parse import urlparse, urlencode
from datetime import datetime
from wsgiref.handlers import format_date_time
import base64
import hmac
import hashlib
import jieba.posseg as pseg  # 新增jieba词性分析[8](@ref)

# ========== 配置参数 ==========
# 去讯飞星火官网注册账号，对应位置输入信息
APPID =     "xxxxxxxxxxx"
APIKey =    "xxxxxxxxxxx"
APISecret = "xxxxxxxxxxx"
Spark_url = "xxxxxxxxxxx"
DOMAIN =    "xxxxxxxxxxx"

DATA_PATH = r"E:\python_datas\new_paper\Dissertation\result\root\autodl-tmp\roberta_zh_L-6-H-768_A-12\test_split.json"
SAVE_FILE = r"E:\python_datas\new_paper\Dissertation\sparkMax-32\关系识别模型\output\规则_output优化49.json"
ERROR_FILE = "errors.json"

# ========== 增强后的领域知识配置 ==========

CAUSAL_WORDS = {
    '导致', '致使', '引发', '造成', '使得', '因此', '所以', '故而',
    "时许", "通过", "随后", "已经", "至少", "立即", "进一步","分别",
    "根据", "经过", "同时", "由于", "最终", "以及",  "再次",
    "因为", "另外", "此外", "一直", "于是",  "初步", "分许",
    "可以", "为了", "特别", "随即", "及时", "针对", "很快", "终身",
    "就是", "及其", "故意",  "然后", "但是", "已致", "并处", "只有",
    "按照", "如果", "不断", "最新",  "对于", "不过", "陆续", "此时"
}

TEMPORAL_WORDS = {
    '之后', '随后', '然后', '次日', '第二天', '紧接着', "时许", "通过",
    "已经", "至少", "立即", "进一步", "分别", "根据", "经过", "同时",
    "由于", "最终", "以及" , "再次", "因为", "另外", "此外", "一直",
    "于是",  "初步", "分许", "可以", "为了", "特别", "随即", "及时",
    "针对", "很快", "终身", "就是", "及其", "因此", "故意", "但是", "已致",
    "并处", "只有", "按照", "如果", "不断", "最新",  "对于", "不过", "陆续", "此时", "正常"
}

CAUSAL_WEIGHTS = {
    '强因果': {'造成', '导致', '引发', '致使', '因此', '使得', '所以', '故而', '通过', '随后', '已经', '至少', '立即',
               '分别', '根据', '经过'},
    '中等因果': {'同时', '由于', '最终', '以及', '再次', '因为', '另外', '此外', '一直', '于是', '初步', '分许', '可以',
                 '为了', '特别', '随即', '及时', '针对'},
    '弱因果': {'很快', '终身', '就是', '及其', '故意', '然后', '但是', '已致', '并处', '只有', '按照', '如果', '不断',
               '最新', '对于', '不过', '陆续', '此时'}
}

TEMPORAL_WEIGHTS = {
    '强时序': {'之后', '随后', '然后', '次日', '第二天', '紧接着', '进一步', '时许', '通过', '已经', '至少', '立即',
               '分别', '根据', '经过'},
    '中等时序': {'最终', '以及', '再次', '因为', '另外', '此外', '一直', '于是', '初步', '分许', '可以', '为了', '特别',
                 '随即', '及时', '针对', '很快', '终身'},
    '弱时序': {'就是', '及其', '因此', '故意', '但是', '已致', '并处', '只有', '按照', '如果', '不断', '最新', '对于',
               '不过', '陆续', '此时', '正常'}
}

# 权重系数
WEIGHT_MAP = {
    '强因果': 1.8, '中等因果': 1.5, '弱因果': 1.2,
    '强时序': 1.8, '中等时序': 1.5, '弱时序': 1.2
}


class SparkAI:
    def __init__(self, appid, apikey, apisecret, gpt_url, domain):
        self.appid = appid
        self.apikey = apikey
        self.apisecret = apisecret
        self.gpt_url = gpt_url
        self.domain = domain
        self.response_content = ""

    def create_url(self):
        """生成WebSocket连接URL"""
        host = urlparse(self.gpt_url).netloc
        path = urlparse(self.gpt_url).path
        now = datetime.now()
        date = format_date_time(time.mktime(now.timetuple()))
        signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
        signature_sha = hmac.new(
            self.apisecret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()
        signature_base64 = base64.b64encode(signature_sha).decode()
        authorization_origin = (
            f'api_key="{self.apikey}", '
            f'algorithm="hmac-sha256", '
            f'headers="host date request-line", '
            f'signature="{signature_base64}"'
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode()
        return f"{self.gpt_url}?{urlencode({'authorization': authorization, 'date': date, 'host': host})}"

    def on_message(self, ws, message):
        data = json.loads(message)
        if data["header"]["code"] != 0:
            print(f"Error: {data['header']['code']}")
            ws.close()
        else:
            content = data["payload"]["choices"]["text"][0]["content"]
            self.response_content += content
            if data["payload"]["choices"]["status"] == 2:
                ws.close()

    def on_error(self, ws, error):
        print("WebSocket Error:", error)

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket Closed")

    def on_open(self, ws, query):
        """建立连接后发送请求"""
        def run(*args):
            data = json.dumps({
                "header": {"app_id": self.appid, "uid": "1234"},
                "parameter": {
                    "chat": {
                        "domain": self.domain,
                        "temperature": 0.2,
                        "max_tokens": 4096,
                        "top_k": 3,
                    }
                },
                "payload": {"message": {"text": [{"role": "user", "content": query}]}}
            })
            ws.send(data)
        thread.start_new_thread(run, ())

    def invoke(self, prompt):
        self.response_content = ""
        ws_url = self.create_url()
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        ws.on_open = lambda ws: self.on_open(ws, prompt)
        ws.run_forever()
        time.sleep(1)
        return self.response_content


class RelationAnalyzer:
    """综合关系分析器（增强版）"""

    def __init__(self, events):
        self.events = {e["id"]: e for e in events}
        self.ordered_events = [e["id"] for e in events]
        self.sentence_positions = self._build_sentence_index(events)  # 新增句子索引[4](@ref)
    def extract_relations(self, raw_response):
        """修复缺失的核心方法"""
        pattern = r"[$（]([\w_]+)\s*[,，]\s*([\w_]+)\s*[,，]\s*(因果关系|时序关系|None)[$）]"
        candidates = re.findall(pattern, raw_response)

        relations = []
        seen_pairs = set()

        for e1, e2, rel in candidates:
            if not self._validate_ids(e1, e2) or e1 == e2:
                continue

            final_rel = self._apply_rules(e1, e2, rel)
            if not final_rel:
                continue

            pair_key = tuple(sorted((e1, e2)) + [final_rel])
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                relations.append({
                    "source": e1,
                    "target": e2,
                    "relation": final_rel,
                    "confidence": self._calculate_confidence(e1, e2, final_rel)
                })
        return relations

    def _validate_ids(self, e1, e2):
        return e1 in self.events and e2 in self.events
    def _build_sentence_index(self, events):
        """构建事件所在句子的位置索引"""
        return {e['id']: idx for idx, e in enumerate(events)}

    def _calculate_confidence(self, e1, e2, rel):
        """动态权重置信度计算（根据网页1知识增强策略和网页5语义依存理论）"""
        score = 0.0
        content = self.events[e1]["content"] + self.events[e2]["content"]

        # 因果/时序词权重计算
        if rel == "因果关系":
            for category, words in CAUSAL_WEIGHTS.items():
                matched = sum(1 for w in words if w in content)
                score += matched * 0.1 * WEIGHT_MAP[category]  # 调整权重系数

        elif rel == "时序关系":
            for category, words in TEMPORAL_WEIGHTS.items():
                matched = sum(1 for w in words if w in content)
                score += matched * 0.1 * WEIGHT_MAP[category]

        # 位置惩罚机制（根据网页4事件演化图谱理论）
        pos_diff = abs(self.sentence_positions[e1] - self.sentence_positions[e2])
        position_penalty = min(pos_diff * 0.05, 0.3)  # 每句差0.05，最大扣0.3[2](@ref)
        score -= position_penalty

        # 置信度边界控制（根据网页3机器学习优化）
        return round(max(0.2, min(0.95, score)), 2)  # 调整置信度范围[0.2-0.95]

    def _apply_rules(self, e1, e2, rel):
        """增强的规则验证（整合网页1知识增强和网页5语义依存）"""
        content = self.events[e1]["content"] + self.events[e2]["content"]

        # 使用jieba进行词性分析（新增部分）[7,8](@ref)
        words = pseg.cut(content)
        verb_count = sum(1 for w, flag in words if flag.startswith('v'))

        # 增强的否定检测（根据网页3模式匹配方法）
        negation_words = {'未', '没有', '未能', '无法', '不', '非'}
        if any(w in negation_words for w in content.split()):
            return None

        # 增强的因果验证（根据网页1知识增强策略）
        if rel == "因果关系":
            # 必须包含动词（根据网页7词性分析）
            if verb_count < 1:
                return None
            # 时序验证强化（根据网页5事件演化理论）
            if not self._check_temporal_order(e1, e2):
                return None

        return rel

    def _check_temporal_order(self, e1, e2):
        """检查事件时序"""
        idx1 = self.ordered_events.index(e1)
        idx2 = self.ordered_events.index(e2)
        return idx1 < idx2

    def _calculate_confidence(self, e1, e2, rel):
        """动态权重置信度计算"""
        score = 0.0
        content = self.events[e1]["content"] + self.events[e2]["content"]

        # 因果词匹配（带权重）
        for word in CAUSAL_WORDS:
            if word in content:
                score += 0.15 * self.causal_weights.get(word, 1.0)

        # 时序词匹配
        temporal_count = sum(1 for w in TEMPORAL_WORDS if w in content)
        score += min(temporal_count * 0.1, 0.3)

        # 位置距离惩罚
        pos_diff = abs(self.ordered_events.index(e1) - self.ordered_events.index(e2))
        score -= min(pos_diff * 0.05, 0.3)

        return round(max(0.3, min(0.95, score)), 2)



def build_enhanced_prompt(doc, events):
    """增强的提示词生成（整合网页1知识增强策略）"""
    event_items = "\n".join(
        f"{idx + 1}. {e['id']} [{e.get('trigger', '')}]：{e['content']}"
        for idx, e in enumerate(events)
    )

    # 动态生成知识描述（根据网页1方法论）
    causal_samples = "、".join(sorted(CAUSAL_WEIGHTS['强因果'])[:3]) + "等分级因果词"
    temporal_samples = "、".join(sorted(TEMPORAL_WEIGHTS['强时序'])[:3]) + "等分级时序词"

    return f"""你是一个新闻领域事件关系分析专家，请根据以下规则分析事件关系：
1. 因果关系需满足：
   - 包含分级因果词（强：{causal_samples}）
   - 原因事件早于结果事件
   - 包含至少1个动词
2. 时序关系需满足：
   - 包含分级时序词（强：{temporal_samples}）
   - 符合事件发展逻辑
文章内容：{doc}
事件列表：{event_items}
请输出严格格式：(事件A_id,事件B_id,关系类型)
示例：（示例保持事件ID对应）
"""


def _extract_trigger(content):
    """使用jieba提取触发词（新增方法）[7,8]"""
    words = pseg.cut(content)
    for word, flag in words:
        if flag.startswith('v') or flag.startswith('n'):
            return word
    return ''

def safe_invoke(ai_client, prompt, max_retries=3):
    """智能重试机制"""
    for attempt in range(max_retries):
        try:
            response = ai_client.invoke(prompt)
            if response.strip():
                return response
            time.sleep(1.5 ** attempt)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return ""

def save_results(filename, data):
    """安全保存结果，保存格式为列表，每个元素包含 news-ID 和 output"""
    try:
        with open(filename, 'r+', encoding='utf-8') as f:
            try:
                existing = json.load(f)
            except:
                existing = []
            existing.append(data)
            f.seek(0)
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except FileNotFoundError:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([data], f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    spark_ai = SparkAI(APPID, APIKey, APISecret, Spark_url, DOMAIN)

    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"数据加载失败: {e}")
        exit(1)

    from tqdm import tqdm
    with tqdm(dataset, desc="分析进度") as pbar:
        for item in pbar:
            try:
                doc = item["doc"]
                events = item["events"]

                prompt = build_enhanced_prompt(doc, events)
                raw_response = safe_invoke(spark_ai, prompt)

                analyzer = RelationAnalyzer(events)
                relations = analyzer.extract_relations(raw_response)

                # 构造输出字符串，每个关系格式为：(事件A_id,事件B_id,关系类型)
                output_str = ", ".join([f"({r['source']},{r['target']},{r['relation']})" for r in relations])
                result = {
                    "news-ID": item["news-ID"],
                    "output": output_str
                }
                save_results(SAVE_FILE, result)

            except Exception as e:
                error_info = {
                    "news-ID": item.get("news-ID", "unknown"),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                save_results(ERROR_FILE, error_info)
                pbar.write(f"处理失败：{item.get('news-ID')} - {str(e)}")

            pbar.update(1)
