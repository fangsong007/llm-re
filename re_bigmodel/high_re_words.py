import json
import jieba
import jieba.posseg as pseg
from collections import defaultdict
from tqdm import tqdm

# ========== 配置参数 ==========
DATA_PATH = r"E:\python_datas\new_paper\Dissertation\result\root\autodl-tmp\roberta_zh_L-6-H-768_A-12\train_split.json"
OUTPUT_FILE = r"E:\python_datas\new_paper\Dissertation\sparklite\prepare\new_word_analysis_binary2.json"

# 使用jieba内置停用词（需要先下载）
try:
    from jieba.analyse import STOP_WORDS as jieba_stopwords
except ImportError:
    jieba.initialize()
    jieba_stopwords = set()

# ========== 初始化因果词和时序词 ==========
INITIAL_CAUSAL_WORDS = {'导致', '致使', '引发', '造成', '使得', '因此', '所以', '故而'}
INITIAL_TEMPORAL_WORDS = {'之后', '随后', '然后', '次日', '第二天', '紧接着'}


# ========== 核心功能函数 ==========
def load_data(file_path):
    """加载数据集并提取所有文本内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载数据失败：{e}")
        return []

    all_text = []
    for item in data:
        all_text.append(item["doc"])
        for event in item.get("events", []):
            all_text.append(event.get("content", ""))
    return all_text


def analyze_words(texts):
    """执行词频和词性分析"""
    word_freq = defaultdict(int)
    pos_counter = defaultdict(int)
    candidate_words = defaultdict(int)

    for text in tqdm(texts, desc="分析文本"):
        words = pseg.cut(text)
        for word, flag in words:
            word = word.strip()
            if len(word) < 2 or word in jieba_stopwords:
                continue

            word_freq[word] += 1
            pos_counter[flag] += 1

            if flag in ['c', 'd', 'p']:  # 连词(c), 副词(d), 介词(p)
                candidate_words[word] += 1

    return word_freq, pos_counter, candidate_words

def find_new_keywords(initial_set, candidate_words, word_freq, top_n=50):
    """发现新的关键词（返回字典格式：{词: 出现次数}）"""
    candidates = []
    for word, c_count in candidate_words.items():
        if word in initial_set:
            continue
        f_count = word_freq.get(word, 0)
        if f_count > 5:  # 频率阈值
            candidates.append((word, c_count, f_count))

    # 按候选分数(c_count)和频率(f_count)排序
    candidates.sort(key=lambda x: (-x[1], -x[2]))
    return {word: f_count for word, _, f_count in candidates[:top_n]}


def save_results(output, file_path):
    """保存分析结果"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"分析结果已保存至 {file_path}")
    except Exception as e:
        print(f"保存结果失败：{e}")


# ========== 主程序 ==========
def main():
    # 初始化jieba（确保停用词加载）
    jieba.initialize()

    # 1. 加载数据
    print("正在加载数据...")
    texts = load_data(DATA_PATH)
    if not texts:
        print("未加载到有效数据，请检查数据路径和格式")
        return

    print(f"已加载 {len(texts)} 条文本")

    # 2. 分析词频和词性
    print("开始分析文本...")
    word_freq, pos_counter, candidate_words = analyze_words(texts)

    # 3. 分析因果词
    causal_results = {
        "initial_words": list(INITIAL_CAUSAL_WORDS),
        "existing_counts": {w: word_freq.get(w, 0) for w in INITIAL_CAUSAL_WORDS},
        "new_candidates": find_new_keywords(INITIAL_CAUSAL_WORDS, candidate_words, word_freq)
    }

    # 4. 分析时序词
    temporal_results = {
        "initial_words": list(INITIAL_TEMPORAL_WORDS),
        "existing_counts": {w: word_freq.get(w, 0) for w in INITIAL_TEMPORAL_WORDS},
        "new_candidates": find_new_keywords(INITIAL_TEMPORAL_WORDS, candidate_words, word_freq)
    }

    # 5. 准备输出结果
    output = {
        "causal_analysis": causal_results,
        "temporal_analysis": temporal_results,
        "top_50_words": sorted(word_freq.items(), key=lambda x: -x[1])[:50],
        "pos_distribution": dict(pos_counter)
    }

    # 6. 保存结果
    save_results(output, OUTPUT_FILE)


if __name__ == "__main__":
    main()