# 多线程 DeepSeek 对抗文本生成与 BERT 分类器对比实验
# 1. 多线程调用 DeepSeek API 生成对抗文本
# 2. 使用 BERT 诈骗检测模型对原文和改写文本分别预测
# 3. 统计原始数据集准确率 vs 改写后准确率变化

import os
import csv
import json
import time
import random
import threading
from typing import Dict, List, Optional
from collections import defaultdict
import concurrent.futures

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("错误: 未安装 openai 库，请运行: pip install openai")
    exit(1)


# ============================================================================
# 配置参数
# ============================================================================
API_KEYS = [
    "sk-468914dbbf4d48b8924c715e1d9bf140",
    "sk-b98d2f9aaabe41d3ad644c0346173e1d",
    "sk-1b987bec6c944bc392f448836aa78952",
    "sk-970132229daa434d9cad2a901495997f",
    "sk-c207fdb4d6954d15b26ff0b16ef343d2",
]
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

INPUT_CSV = "测试集结果.csv"
OUTPUT_DIR = "deepseek_multithreaded_results"
MODEL_PATH = "bert-model/fraud_detection_model.pt"
TOKENIZER_PATH = "bert-base-chinese"

MAX_SAMPLES = 10
MAX_WORKERS = min(len(API_KEYS), 8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

thread_local = threading.local()


# ============================================================================
# BERT 诈骗检测模型定义
# ============================================================================
class FraudDetectionModel(nn.Module):
    def __init__(self, dropout=0.3):
        super(FraudDetectionModel, self).__init__()
        model_path = os.path.join('bert-model', 'bert-fraud')
        if os.path.exists(model_path):
            self.bert = BertModel.from_pretrained(model_path)
        else:
            self.bert = BertModel.from_pretrained('bert-base-chinese', cache_dir='bert-model')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# ============================================================================
# DeepSeek API 调用
# ============================================================================
def get_deepseek_client() -> OpenAI:
    if not hasattr(thread_local, "client"):
        api_key = random.choice(API_KEYS)
        thread_local.client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return thread_local.client


def generate_adversarial_text(original_text: str) -> Dict[str, Optional[str]]:
    client = get_deepseek_client()
    
    system_prompt = (
        "你是一名资深的中文诈骗话术改写专家。你的任务是在保持诈骗语义的前提下，"
        "将原始对话改写得更加自然、隐蔽且难以被自动检测模型识别。"
    )
    
    user_prompt = f"""请阅读以下诈骗对话，并输出 JSON 格式的改写结果：

原始文本:
{original_text}

要求：
1. 生成一个新的对抗文本，必须保持诈骗意图与语义一致
2. 用词更自然、更隐蔽，避免明显的诈骗特征词
3. 简要说明改写策略

JSON 输出格式：
{{
  "adversarial_text": "改写后的对抗文本",
  "reason": "改写策略说明"
}}
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.5,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            
            start = content.find('{')
            end = content.rfind('}')
            if start == -1 or end == -1:
                raise ValueError("未找到 JSON 片段")
            
            data = json.loads(content[start:end + 1])
            return {
                "adversarial_text": data.get("adversarial_text"),
                "reason": data.get("reason", ""),
                "success": True
            }
            
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return {
                    "adversarial_text": None,
                    "reason": f"DeepSeek 调用失败: {exc}",
                    "success": False
                }
    
    return {"adversarial_text": None, "reason": "未知错误", "success": False}


# ============================================================================
# 数据加载
# ============================================================================
def load_dialogues_from_csv(csv_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    dialogues = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            text_key = 'specific_dialogue_content'
            if text_key not in row:
                text_key = '\ufeffspecific_dialogue_content'
            
            fraud_key = 'is_fraud'
            if fraud_key not in row:
                fraud_key = '\ufeffis_fraud'
            
            text = row.get(text_key, '').strip()
            is_fraud_str = row.get(fraud_key, '0').strip()
            
            if not text:
                continue
            
            is_fraud = 1 if is_fraud_str in ['1', 'True', 'true', '是'] else 0
            
            dialogues.append({
                'id': idx + 1,
                'text': text,
                'is_fraud': is_fraud
            })
            
            if max_samples and len(dialogues) >= max_samples:
                break
    
    return dialogues


# ============================================================================
# BERT 预测器
# ============================================================================
class BERTPredictor:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = FraudDetectionModel()
        self.model.to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
    
    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            prob = probs[0, pred].item()
        
        return {'pred': pred, 'prob': prob}


# ============================================================================
# 多线程处理任务
# ============================================================================
def process_single_dialogue(dialogue: Dict, predictor: BERTPredictor) -> Dict:
    original_text = dialogue['text']
    true_label = dialogue['is_fraud']
    
    original_pred = predictor.predict(original_text)
    
    gen_result = generate_adversarial_text(original_text)
    adversarial_text = gen_result.get('adversarial_text') or original_text
    
    adversarial_pred = predictor.predict(adversarial_text)
    
    return {
        'id': dialogue['id'],
        'true_label': true_label,
        'original_text': original_text,
        'adversarial_text': adversarial_text,
        'deepseek_reason': gen_result.get('reason', ''),
        'deepseek_success': gen_result.get('success', False),
        'original_pred': original_pred['pred'],
        'original_prob': original_pred['prob'],
        'adversarial_pred': adversarial_pred['pred'],
        'adversarial_prob': adversarial_pred['prob'],
        'prediction_changed': original_pred['pred'] != adversarial_pred['pred'],
        'attack_success': (original_pred['pred'] == true_label) and (adversarial_pred['pred'] != true_label)
    }


def run_multithreaded_attack(dialogues: List[Dict], predictor: BERTPredictor) -> List[Dict]:
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_dialogue = {
            executor.submit(process_single_dialogue, dialogue, predictor): dialogue
            for dialogue in dialogues
        }
        
        completed = 0
        total = len(dialogues)
        
        for future in concurrent.futures.as_completed(future_to_dialogue):
            result = future.result()
            results.append(result)
            
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"已完成 {completed}/{total} 个样本 ({completed/total*100:.1f}%)")
    
    results.sort(key=lambda x: x['id'])
    return results


# ============================================================================
# 统计与报告生成
# ============================================================================
def compute_statistics(results: List[Dict]) -> Dict:
    total = len(results)
    
    original_correct = sum(1 for r in results if r['original_pred'] == r['true_label'])
    adversarial_correct = sum(1 for r in results if r['adversarial_pred'] == r['true_label'])
    
    prediction_changed = sum(1 for r in results if r['prediction_changed'])
    attack_success = sum(1 for r in results if r['attack_success'])
    deepseek_success = sum(1 for r in results if r['deepseek_success'])
    
    original_acc = original_correct / total * 100 if total > 0 else 0
    adversarial_acc = adversarial_correct / total * 100 if total > 0 else 0
    acc_drop = original_acc - adversarial_acc
    
    fraud_samples = [r for r in results if r['true_label'] == 1]
    non_fraud_samples = [r for r in results if r['true_label'] == 0]
    
    fraud_original_correct = sum(1 for r in fraud_samples if r['original_pred'] == 1)
    fraud_adversarial_correct = sum(1 for r in fraud_samples if r['adversarial_pred'] == 1)
    
    non_fraud_original_correct = sum(1 for r in non_fraud_samples if r['original_pred'] == 0)
    non_fraud_adversarial_correct = sum(1 for r in non_fraud_samples if r['adversarial_pred'] == 0)
    
    fraud_original_acc = fraud_original_correct / len(fraud_samples) * 100 if fraud_samples else 0
    fraud_adversarial_acc = fraud_adversarial_correct / len(fraud_samples) * 100 if fraud_samples else 0
    
    non_fraud_original_acc = non_fraud_original_correct / len(non_fraud_samples) * 100 if non_fraud_samples else 0
    non_fraud_adversarial_acc = non_fraud_adversarial_correct / len(non_fraud_samples) * 100 if non_fraud_samples else 0
    
    return {
        'total_samples': total,
        'fraud_samples': len(fraud_samples),
        'non_fraud_samples': len(non_fraud_samples),
        'original_accuracy': original_acc,
        'adversarial_accuracy': adversarial_acc,
        'accuracy_drop': acc_drop,
        'prediction_changed_count': prediction_changed,
        'prediction_changed_rate': prediction_changed / total * 100 if total > 0 else 0,
        'attack_success_count': attack_success,
        'attack_success_rate': attack_success / total * 100 if total > 0 else 0,
        'deepseek_success_count': deepseek_success,
        'deepseek_success_rate': deepseek_success / total * 100 if total > 0 else 0,
        'fraud_original_acc': fraud_original_acc,
        'fraud_adversarial_acc': fraud_adversarial_acc,
        'fraud_acc_drop': fraud_original_acc - fraud_adversarial_acc,
        'non_fraud_original_acc': non_fraud_original_acc,
        'non_fraud_adversarial_acc': non_fraud_adversarial_acc,
        'non_fraud_acc_drop': non_fraud_original_acc - non_fraud_adversarial_acc,
    }


def generate_report(stats: Dict, output_dir: str):
    report = f"""
{'=' * 80}
DeepSeek 对抗文本生成实验报告
{'=' * 80}

一、实验设置
{'=' * 80}
总样本数: {stats['total_samples']}
  - 诈骗样本: {stats['fraud_samples']}
  - 非诈骗样本: {stats['non_fraud_samples']}
DeepSeek 成功生成对抗文本: {stats['deepseek_success_count']} ({stats['deepseek_success_rate']:.2f}%)

二、原始数据集实验结果
{'=' * 80}
总体准确率: {stats['original_accuracy']:.2f}%
  - 诈骗样本准确率: {stats['fraud_original_acc']:.2f}%
  - 非诈骗样本准确率: {stats['non_fraud_original_acc']:.2f}%

三、改写后数据集实验结果
{'=' * 80}
总体准确率: {stats['adversarial_accuracy']:.2f}%
  - 诈骗样本准确率: {stats['fraud_adversarial_acc']:.2f}%
  - 非诈骗样本准确率: {stats['non_fraud_adversarial_acc']:.2f}%

四、效果变化分析
{'=' * 80}
准确率变化: {stats['accuracy_drop']:.2f}% {'(下降)' if stats['accuracy_drop'] > 0 else '(上升)'}
  - 诈骗样本准确率变化: {stats['fraud_acc_drop']:.2f}%
  - 非诈骗样本准确率变化: {stats['non_fraud_acc_drop']:.2f}%

预测翻转样本数: {stats['prediction_changed_count']} ({stats['prediction_changed_rate']:.2f}%)
攻击成功样本数: {stats['attack_success_count']} ({stats['attack_success_rate']:.2f}%)
  (攻击成功 = 原本预测正确，改写后预测错误)

{'=' * 80}
"""
    
    report_file = os.path.join(output_dir, 'experiment_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n报告已保存至: {report_file}")


def save_results(results: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'adversarial_results.csv')
    
    fieldnames = [
        'id', 'true_label', 'original_text', 'adversarial_text',
        'deepseek_reason', 'deepseek_success',
        'original_pred', 'original_prob',
        'adversarial_pred', 'adversarial_prob',
        'prediction_changed', 'attack_success'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"详细结果已保存至: {output_file}")
    
    success_results = [r for r in results if r['attack_success']]
    if success_results:
        success_file = os.path.join(output_dir, 'attack_success_samples.csv')
        with open(success_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(success_results)
        print(f"攻击成功样本已保存至: {success_file}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 80)
    print("DeepSeek 多线程对抗文本生成实验")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  输入文件: {INPUT_CSV}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  最大样本数: {MAX_SAMPLES}")
    print(f"  线程数: {MAX_WORKERS}")
    print(f"  API Keys: {len(API_KEYS)} 个")
    
    print("\n加载数据...")
    dialogues = load_dialogues_from_csv(INPUT_CSV, MAX_SAMPLES)
    print(f"成功加载 {len(dialogues)} 个对话样本")
    
    print("\n加载 BERT 模型...")
    predictor = BERTPredictor(MODEL_PATH, TOKENIZER_PATH)
    print("模型加载完成")
    
    print("\n开始多线程生成对抗文本并预测...")
    results = run_multithreaded_attack(dialogues, predictor)
    
    print("\n计算统计数据...")
    stats = compute_statistics(results)
    
    print("\n保存结果...")
    save_results(results, OUTPUT_DIR)
    
    print("\n生成实验报告...")
    generate_report(stats, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("实验完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
