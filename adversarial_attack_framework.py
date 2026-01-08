"""
1. 词级别改写（TextFooler）
2. 语义一致性评估与过滤
3. 欺诈对话数据集全流程攻击
4. 高质量改写样本生成
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM
from tqdm import tqdm
import os
import json
from typing import List, Dict, Tuple, Optional
import jieba
import jieba.posseg as pseg
from collections import defaultdict
import re

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class WordImportanceCalculator:
    # 计算词重要性的类
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def compute_importance_by_deletion(self, text: str, true_label: int) -> List[Tuple[int, str, float]]:

        # 分词
        words = list(jieba.cut(text))
        if len(words) <= 1:
            return []
        
        # 获取原始预测概率
        original_prob = self._get_prediction_prob(text, true_label)
        
        importance_scores = []
        
        # 逐个删除词并计算重要性
        for i, word in enumerate(words):
            # 删除第i个词
            modified_words = words[:i] + words[i+1:]
            modified_text = ''.join(modified_words)
            
            # 获取删除后的预测概率
            modified_prob = self._get_prediction_prob(modified_text, true_label)
            
            # 重要性 = 原始概率 - 删除后概率
            importance = original_prob - modified_prob
            importance_scores.append((i, word, importance))
        
        # 按重要性排序
        importance_scores.sort(key=lambda x: x[2], reverse=True)
        
        return importance_scores
    
    def _get_prediction_prob(self, text: str, label: int) -> float:
        # 获取模型对指定标签的预测概率
        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            token_type_ids = encoding.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                probs = torch.softmax(logits, dim=1)
                return probs[0][label].item()
        except Exception as e:
            print(f"预测错误: {e}")
            return 0.0


class SynonymGenerator:
    #同义词生成器
    
    def __init__(self, method='bert', model_path='bert-base-chinese'):
        self.method = method
        if method == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForMaskedLM.from_pretrained(model_path)
            self.model.to(device)
            self.model.eval()
    
    def get_synonyms(self, word: str, context: str, top_k: int = 10) -> List[str]:

        if self.method == 'bert':
            return self._get_synonyms_bert(word, context, top_k)
        else:
            return self._get_synonyms_simple(word, top_k)
    
    def _get_synonyms_bert(self, word: str, context: str, top_k: int) -> List[str]:
        try:
            # 将目标词替换为[MASK]
            masked_context = context.replace(word, '[MASK]', 1)
            
            # 编码
            encoding = self.tokenizer(
                masked_context,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # 找到[MASK]的位置
            mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[1]
            
            if len(mask_token_index) == 0:
                return []
            
            # 预测
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = outputs.logits
            
            # 获取[MASK]位置的预测
            mask_token_logits = predictions[0, mask_token_index[0], :]
            top_tokens = torch.topk(mask_token_logits, top_k + 10, dim=0).indices.tolist()
            
            # 解码并过滤
            synonyms = []
            for token_id in top_tokens:
                candidate = self.tokenizer.decode([token_id]).strip()
                # 过滤特殊字符和原词
                if candidate and candidate != word and len(candidate) > 0 and not candidate.startswith('['):
                    synonyms.append(candidate)
                if len(synonyms) >= top_k:
                    break
            
            return synonyms
        except Exception as e:
            print(f"BERT同义词生成错误: {e}")
            return []
    
    def _get_synonyms_simple(self, word: str, top_k: int) -> List[str]:
        """简单的同义词字典(可扩展)"""
        synonym_dict = {
            '银行': ['金融机构', '银行机构', '金融单位'],
            '客服': ['工作人员', '服务人员', '客户经理'],
            '验证': ['核实', '确认', '检查', '审核'],
            '账户': ['账号', '帐户', '帐号'],
            '信息': ['资料', '数据', '情况'],
            '需要': ['要', '得', '必须'],
            '您': ['你', '您'],
            '好': ['行', '可以', '好的'],
        }
        return synonym_dict.get(word, [])[:top_k]


class SemanticSimilarityChecker:
    """语义相似度检查器"""
    
    def __init__(self, method='cosine', model_path='bert-base-chinese'):
        self.method = method
        if method in ['cosine', 'bert']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(device)
            self.model.eval()
    
    def check_similarity(self, text1: str, text2: str, threshold: float = 0.8) -> Tuple[bool, float]:

        if self.method == 'cosine':
            return self._check_similarity_cosine(text1, text2, threshold)
        else:
            return self._check_similarity_simple(text1, text2, threshold)
    
    def _check_similarity_cosine(self, text1: str, text2: str, threshold: float) -> Tuple[bool, float]:

        try:
            # 获取文本嵌入
            emb1 = self._get_embedding(text1)
            emb2 = self._get_embedding(text2)
            
            # 计算余弦相似度
            similarity = torch.cosine_similarity(emb1, emb2, dim=0).item()
            
            return similarity >= threshold, similarity
        except Exception as e:
            print(f"相似度计算错误: {e}")
            return False, 0.0
    
    def _get_embedding(self, text: str) -> torch.Tensor:

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
            # 使用[CLS]的嵌入
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        return embedding
    
    def _check_similarity_simple(self, text1: str, text2: str, threshold: float) -> Tuple[bool, float]:
        # 简单的字符重叠相似度
        words1 = set(jieba.cut(text1))
        words2 = set(jieba.cut(text2))
        
        if len(words1) == 0 or len(words2) == 0:
            return False, 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity >= threshold, similarity


class TextFoolerAttack:

    def __init__(self, model, tokenizer, synonym_generator, similarity_checker):
        self.model = model
        self.tokenizer = tokenizer
        self.synonym_generator = synonym_generator
        self.similarity_checker = similarity_checker
        self.importance_calculator = WordImportanceCalculator(model, tokenizer)
        
        # A100 GPU优化参数
        self.batch_size = 32  # A100可以处理大批量
        self.max_candidates = 20  # 增加候选同义词数（提高攻击成功率）
        self.use_amp = True  # 启用混合精度
        
        # 调试选项：输出前N个成功样本
        self.show_samples = 3  # 输出前3个成功样本
        self.sample_count = 0  # 已输出样本计数
    
    def attack(self, text: str, true_label: int, max_changes: int = 10, 
               similarity_threshold: float = 0.7) -> Dict:

        # 获取原始预测
        original_pred, original_prob = self._get_prediction(text)
        
        # 如果原始预测就是错的,不需要攻击
        if original_pred != true_label:
            # 调试输出：显示前3个预测错误的样本
            if self.sample_count < 3:
                self.sample_count += 1
                print(f"\n[跳过] 样本#{self.sample_count} - 原始预测错误")
                print(f"文本: {text[:100]}...")
                print(f"真实标签: {true_label}, 预测: {original_pred}\n")
            
            return {
                'success': False,
                'original_text': text,
                'adversarial_text': text,
                'original_label': true_label,
                'original_pred': original_pred,
                'adversarial_pred': original_pred,
                'num_changes': 0,
                'similarity': 1.0,
                'reason': 'original_prediction_wrong'
            }
        
        # 计算词重要性
        importance_scores = self.importance_calculator.compute_importance_by_deletion(text, true_label)
        
        if len(importance_scores) == 0:
            # 调试输出
            if self.sample_count < 3:
                print(f"[失败] 无可替换的词")
            
            return {
                'success': False,
                'original_text': text,
                'adversarial_text': text,
                'original_label': true_label,
                'original_pred': original_pred,
                'adversarial_pred': original_pred,
                'num_changes': 0,
                'similarity': 1.0,
                'reason': 'no_words_to_replace'
            }
        
        # 按重要性替换词
        words = list(jieba.cut(text))
        adversarial_words = words.copy()
        num_changes = 0
        changed_positions = []
        
        for word_idx, word, importance in importance_scores:
            if num_changes >= max_changes:
                break
            
            # 只跳过空白和标点
            if len(word.strip()) == 0 or word in ['，', '。', '！', '？', '、', '；', '：', '"', '"', ''', ''', '（', '）', '《', '》']:
                continue
            
            # 获取同义词（A100优化：增加候选数）
            context = text
            synonyms = self.synonym_generator.get_synonyms(word, context, top_k=self.max_candidates)
            
            if len(synonyms) == 0:
                continue
            
            # 尝试每个同义词
            best_synonym = None
            best_prob_drop = 0
            
            for synonym in synonyms:
                # 替换词
                candidate_words = adversarial_words.copy()
                candidate_words[word_idx] = synonym
                candidate_text = ''.join(candidate_words)
                
                # 检查语义相似度
                is_similar, sim_score = self.similarity_checker.check_similarity(text, candidate_text, similarity_threshold)
                
                if not is_similar:
                    continue
                
                # 获取预测
                pred, prob = self._get_prediction(candidate_text)
                
                # 如果攻击成功,立即返回（输出在最后统一处理）
                if pred != true_label:
                    return {
                        'success': True,
                        'original_text': text,
                        'adversarial_text': candidate_text,
                        'original_label': true_label,
                        'original_pred': original_pred,
                        'adversarial_pred': pred,
                        'num_changes': num_changes + 1,
                        'similarity': sim_score,
                        'changed_words': changed_positions + [(word, synonym)]
                    }
                
                # 记录最佳替换(概率下降最多)
                prob_drop = original_prob - prob
                if prob_drop > best_prob_drop:
                    best_prob_drop = prob_drop
                    best_synonym = synonym
            
            # 应用最佳替换
            if best_synonym is not None:
                adversarial_words[word_idx] = best_synonym
                num_changes += 1
                changed_positions.append((word, best_synonym))
        
        # 生成最终对抗文本
        adversarial_text = ''.join(adversarial_words)
        adversarial_pred, _ = self._get_prediction(adversarial_text)
        _, final_similarity = self.similarity_checker.check_similarity(text, adversarial_text, 0.0)
        
        # 无论成功与否，都输出前3个替换产物
        if self.sample_count < self.show_samples:
            self.sample_count += 1
            success_status = "成功" if adversarial_pred != true_label else "失败"
            print(f"\n{'='*60}")
            print(f"样本 #{self.sample_count} - 攻击{success_status}")
            print(f"{'='*60}")
            print(f"原文: {text}")
            print(f"改写: {adversarial_text}")
            print(f"真实标签: {'诈骗' if true_label == 1 else '非诈骗'}")
            print(f"原始预测: {'诈骗' if original_pred == 1 else '非诈骗'}")
            print(f"攻击后预测: {'诈骗' if adversarial_pred == 1 else '非诈骗'}")
            print(f"替换词数: {num_changes}")
            print(f"语义相似度: {final_similarity:.3f}")
            print(f"替换详情: {changed_positions}")
            print(f"{'='*60}\n")
        
        return {
            'success': adversarial_pred != true_label,
            'original_text': text,
            'adversarial_text': adversarial_text,
            'original_label': true_label,
            'original_pred': original_pred,
            'adversarial_pred': adversarial_pred,
            'num_changes': num_changes,
            'similarity': final_similarity,
            'changed_words': changed_positions
        }
    
    def _get_prediction(self, text: str) -> Tuple[int, float]:
        """获取模型预测（A100优化：支持混合精度）"""
        try:
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # A100优化：使用混合精度加速（修复API）
            if self.use_amp and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        outputs = self.model(**inputs)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            prob = probs[0][pred].item()
            
            return pred, prob
        except Exception as e:
            print(f"预测错误: {e}")
            return 0, 0.0
    
    def _get_predictions_batch(self, texts: List[str]) -> List[Tuple[int, float]]:
        """批量获取模型预测（A100优化：批量推理）"""
        try:
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # A100优化：批量推理 + 混合精度（修复API）
            if self.use_amp and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        outputs = self.model(**inputs)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            results = []
            for i in range(len(texts)):
                pred = preds[i].item()
                prob = probs[i][pred].item()
                results.append((pred, prob))
            
            return results
        except Exception as e:
            print(f"批量预测错误: {e}")
            return [(0, 0.0)] * len(texts)
    
    def _is_stopword(self, word: str) -> bool:
        # 判断是否为停用词
        stopwords = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
        return word in stopwords


class LLMEvaluator:
    # LLM评估器 - 评估对抗样本的语义一致性
    
    def __init__(self, method='rule_based', api_key=None, api_keys=None, base_url=None):
        self.method = method
        self.api_key = api_key
        self.api_keys = api_keys or []
        self.base_url = base_url or "https://api.deepseek.com"
        
        # 如果提供了单个api_key，添加到列表中
        if api_key and not api_keys:
            self.api_keys = [api_key]
        
        # 初始化客户端（如果使用DeepSeek）
        self.client = None
        if self.method == 'deepseek' and self.api_keys:
            try:
                from openai import OpenAI
                import random
                # 随机选择一个API key
                selected_key = random.choice(self.api_keys)
                self.client = OpenAI(api_key=selected_key, base_url=self.base_url)
            except ImportError:
                print("警告: 未安装openai库，DeepSeek API调用将失败")
                print("请运行: pip install openai")
                self.method = 'rule_based'
    
    def evaluate_semantic_consistency(self, original_text: str, adversarial_text: str) -> Tuple[bool, float, str]:
        # 评估对抗样本的语义一致性，返回(是否一致, 置信度, 原因)
        if self.method == 'rule_based':
            return self._evaluate_rule_based(original_text, adversarial_text)
        elif self.method == 'deepseek':
            return self._evaluate_deepseek(original_text, adversarial_text)
        elif self.method == 'openai':
            return self._evaluate_openai(original_text, adversarial_text)
        else:
            return self._evaluate_rule_based(original_text, adversarial_text)
    
    def evaluate_with_score(
        self,
        original_text: str,
        adversarial_text: str
    ) -> Dict[str, Optional[str]]:
        # 使用LLM给出1-5分评分，若<3则返回改写文本
        if self.method == 'deepseek':
            return self._evaluate_with_rating_deepseek(original_text, adversarial_text)
        else:
            return self._evaluate_with_rating_rule_based(original_text, adversarial_text)
    
    def _evaluate_rule_based(self, original_text: str, adversarial_text: str) -> Tuple[bool, float, str]:
        # 基于规则的评估
        # 规则1: 检查关键词保留
        original_words = set(jieba.cut(original_text))
        adversarial_words = set(jieba.cut(adversarial_text))
        
        # 计算词重叠率
        overlap = len(original_words.intersection(adversarial_words))
        overlap_rate = overlap / len(original_words) if len(original_words) > 0 else 0
        
        # 规则2: 检查句子长度变化
        len_diff = abs(len(original_text) - len(adversarial_text))
        len_ratio = len_diff / len(original_text) if len(original_text) > 0 else 0
        
        # 移除关键实体保护，允许所有词被替换
        
        # 综合判断（只考虑词重叠率和长度变化）
        is_consistent = (overlap_rate >= 0.5 and len_ratio <= 0.4)
        confidence = (overlap_rate * 0.7 + (1 - len_ratio) * 0.3)
        
        reason = f"词重叠率={overlap_rate:.2f}, 长度变化率={len_ratio:.2f}"
        
        return is_consistent, confidence, reason
    
    def _evaluate_with_rating_rule_based(self, original_text: str, adversarial_text: str) -> Dict[str, Optional[str]]:
        # 无LLM时的评分备用，实现简单的1-5分打分
        is_consistent, confidence, reason = self._evaluate_rule_based(original_text, adversarial_text)
        # 将置信度映射到1-5分
        score = max(1, min(5, int(round(confidence * 5))))
        
        # 如果评分<3，生成简单的规则改写（兜底逻辑）
        suggested_text = None
        if score < 3:
            # 简单策略：返回当前对抗文本作为建议（表示无法改进）
            suggested_text = adversarial_text
            reason += " | 规则评分<3，建议文本即为当前对抗文本"
        
        return {
            'score': score,
            'is_consistent': is_consistent,
            'reason': reason,
            'suggested_text': suggested_text
        }
    
    def _evaluate_deepseek(self, original_text: str, adversarial_text: str) -> Tuple[bool, float, str]:
        # 使用DeepSeek API评估语义一致性
        if not self.client:
            print("警告: DeepSeek客户端未初始化，使用规则评估")
            return self._evaluate_rule_based(original_text, adversarial_text)
        
        try:
            import time
            
            # 构建提示词
            system_prompt = """你是一个语义一致性评估专家。你的任务是判断两个句子是否表达相同的意思。
请严格按照以下格式回答：
判断: 是/否
置信度: 0.0-1.0之间的数字
理由: 简短说明"""

            user_prompt = f"""请判断以下两个句子是否表达相同的意思：

句子1: {original_text}
句子2: {adversarial_text}

注意：
1. 如果两句话的核心意思相同，只是表达方式不同，应判断为"是"
2. 如果关键信息发生改变（如银行→公司、转账→发送），应判断为"否"
3. 请给出你的置信度（0-1之间的小数）

请严格按照格式回答。"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 调用DeepSeek API（带重试机制）
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        temperature=0.2,
                        max_tokens=200,
                        stream=False
                    )
                    
                    # 解析响应
                    content = response.choices[0].message.content.strip()
                    
                    # 提取判断结果
                    is_consistent = False
                    confidence = 0.5
                    reason = content
                    
                    # 解析响应内容
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if '判断' in line or 'judgment' in line.lower():
                            is_consistent = '是' in line or 'yes' in line.lower() or '相同' in line
                        elif '置信度' in line or 'confidence' in line.lower():
                            # 提取数字
                            import re
                            numbers = re.findall(r'0\.\d+|\d+\.\d+', line)
                            if numbers:
                                confidence = float(numbers[0])
                        elif '理由' in line or 'reason' in line.lower():
                            reason = line.split(':', 1)[-1].strip() if ':' in line else line
                    
                    return is_consistent, confidence, reason
                    
                except Exception as exc:
                    print(f"DeepSeek API调用失败，第 {attempt} 次尝试: {exc}")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        print("DeepSeek API调用失败，回退到规则评估")
                        return self._evaluate_rule_based(original_text, adversarial_text)
            
        except Exception as e:
            print(f"DeepSeek评估出错: {e}")
            return self._evaluate_rule_based(original_text, adversarial_text)
    
    def _evaluate_with_rating_deepseek(self, original_text: str, adversarial_text: str) -> Dict[str, Optional[str]]:
        # 调用DeepSeek，返回1-5分评分，分数<3时给出改写文本
        if not self.client:
            print("警告: DeepSeek客户端未初始化，使用规则评分")
            return self._evaluate_with_rating_rule_based(original_text, adversarial_text)
        
        system_prompt = (
            "你是一个对抗样本评审专家，请对生成的中文对抗文本进行质量评分并在必要时给出改写建议。"
        )
        user_prompt = f"""请比较以下两段文本，并按照要求输出JSON：

原始文本:
{original_text}

对抗文本:
{adversarial_text}

要求：
1. 参照语义一致性、流畅度、欺诈语境保持度，给对抗文本打1-5的分数（5为最好）。
2. 如果分数小于3，请给出你修改后的对抗文本，需保持诈骗语义但更自然、隐蔽。
3. 请严格输出如下JSON（若无需改写，suggested_text 设为""）：
{{
  "score": 1-5的整数,
  "is_consistent": true/false,
  "reason": "评分依据",
  "suggested_text": "当score<3时给出的改写，否则留空字符串"
}}
"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError(f"未解析到JSON: {content}")
            result = json.loads(match.group())
            
            score = int(result.get("score", 3))
            score = max(1, min(5, score))
            is_consistent = bool(result.get("is_consistent", False))
            reason = result.get("reason", "")
            suggested_text = result.get("suggested_text", "")
            
            # 如果评分<3但没有改写文本，主动请求DeepSeek改写
            if score < 3 and not suggested_text:
                suggested_text = self._request_rewrite(original_text, adversarial_text)
            
            # 最终兜底：如果评分<3但仍无改写，返回当前对抗文本
            if score < 3 and not suggested_text:
                suggested_text = adversarial_text
                reason += " | DeepSeek未返回改写，使用当前对抗文本"
            
            return {
                "score": score,
                "is_consistent": is_consistent,
                "reason": reason,
                "suggested_text": suggested_text or None
            }
        except Exception as e:
            print(f"DeepSeek评分失败，回退规则评分: {e}")
            return self._evaluate_with_rating_rule_based(original_text, adversarial_text)
    
    def _request_rewrite(self, original_text: str, adversarial_text: str) -> Optional[str]:
        # 当评分过低时，请求DeepSeek给出新的对抗文本
        if not self.client:
            return None
        
        prompt = f"""当前对抗文本评分不足，请你在保留诈骗语义的前提下，给出更自然、更具迷惑性的改写。

原始文本:
{original_text}

需要改写的对抗文本:
{adversarial_text}

输出一段新的对抗文本即可，不要添加额外说明。"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个擅长撰写中文诈骗对话的对抗样本生成器。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"DeepSeek改写失败: {e}")
            return None
    
    def _evaluate_openai(self, original_text: str, adversarial_text: str) -> Tuple[bool, float, str]:
        # 使用OpenAI API评估(需要API密钥)
        # 这里是示例代码,实际使用需要配置API
        # import openai
        # openai.api_key = self.api_key
        # 
        # prompt = f"""请判断以下两个句子是否表达相同的意思:
        # 
        # 句子1: {original_text}
        # 句子2: {adversarial_text}
        # 
        # 请回答"是"或"否",并给出置信度(0-1)和原因。
        # """
        # 
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # 
        # # 解析响应
        # ...
        
        # 暂时使用规则方法
        return self._evaluate_rule_based(original_text, adversarial_text)


def main():
    # 主函数 - 示例用法
    print("=" * 80)
    print("对抗攻击框架")
    print("=" * 80)
    
    # 示例: 加载模型
    model_path = 'bert-model/fraud_detection_model.pt'
    if not os.path.exists(model_path):
        print(f"\n错误: 模型文件不存在: {model_path}")
        print("请先训练模型或指定正确的模型路径")
        return
    
    print("\n正在加载模型...")
    # 这里需要根据实际情况加载模型
    # model = ...
    # tokenizer = ...
    
    print("\n框架已准备就绪!")
    print("\n使用说明:")
    print("1. 导入此模块: from adversarial_attack_framework import *")
    print("2. 初始化攻击器: attacker = TextFoolerAttack(model, tokenizer, synonym_gen, sim_checker)")
    print("3. 执行攻击: result = attacker.attack(text, label)")
    print("4. 使用LLM评估: evaluator = LLMEvaluator()")
    print("5. 评估结果: is_valid, conf, reason = evaluator.evaluate_semantic_consistency(orig, adv)")


if __name__ == "__main__":
    main()
