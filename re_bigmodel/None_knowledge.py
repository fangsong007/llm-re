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


# ========== 配置参数 ==========
# 去讯飞星火官网注册账号，对应位置输入信息
APPID =     "xxxxxxxxxxx"
APIKey =    "xxxxxxxxxxx"
APISecret = "xxxxxxxxxxx"
Spark_url = "xxxxxxxxxxx"
DOMAIN =    "xxxxxxxxxxx"

DATA_PATH = r"E:\python_datas\new_paper\Dissertation\result\root\autodl-tmp\roberta_zh_L-6-H-768_A-12\test_split.json"
SAVE_FILE = r"E:\python_datas\new_paper\Dissertation\sparkMax-32\关系识别模型\output\对比实验_output.json"
ERROR_FILE = "errors.json"

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
    """
    对比实验版本的关系分析器：
    移除了基于因果、时序领域知识词汇的规则修正，
    直接使用模型输出的关系，不做额外校正。
    """

    def __init__(self, events):
        self.events = {e["id"]: e for e in events}
        self.ordered_events = [e["id"] for e in events]

    def extract_relations(self, raw_response):
        """提取关系，不进行额外的规则修正"""
        pattern = r"[\(（]([\w_]+)\s*[,，]\s*([\w_]+)\s*[,，]\s*(因果关系|时序关系|None)[\)）]"
        candidates = re.findall(pattern, raw_response)

        relations = []
        seen_pairs = set()

        for e1, e2, rel in candidates:
            if not self._validate_ids(e1, e2) or e1 == e2:
                continue

            final_rel = rel
            pair_key = tuple(sorted((e1, e2)) + [final_rel])
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                relations.append({
                    "source": e1,
                    "target": e2,
                    "relation": final_rel
                })
        return relations

    def _validate_ids(self, e1, e2):
        return e1 in self.events and e2 in self.events


def build_enhanced_prompt(doc, events):
    """
    构造提示词：
    仅给出文章内容和事件列表，不提供因果、时序词汇示例。
    """
    event_items = "\n".join(
        f"{idx + 1}. {e['id']} [{e.get('trigger', '')}]：{e['content']}"
        for idx, e in enumerate(events)
    )
    return f"""你是一个优秀的新闻领域事件关系分析专家，现在需要你从给定的一篇文章中找出各个事件之间是否存在因果关系或时序关系:
文章内容：{doc}
文中涉及事件列表：{event_items}
注意：两个事件之间可能即存在因果关系又存在时序关系，
## 请你找出所有事件之间的关系并直接输出答案，按事件ID输出关系，格式严格为：(事件A_id, 事件B_id, 关系类型)
-关系类型：因果关系/时序关系/None
输出示例：
(990_1,990_2,时序关系)，(990_2,990_3,因果关系)，(990_1,990_3,None)
请严格按步骤分析每个事件对。
"""

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
