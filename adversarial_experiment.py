# 对抗攻击实验主程序 - 在欺诈对话数据集上执行对抗攻击并评估效果
# 使用方法: python adversarial_experiment.py --mode attack --num_samples 500

import torch
import pandas as pd
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# 导入BERT模型
from bert2 import FraudDetectionModel, load_dataset
from transformers import BertTokenizer

# 导入对抗攻击框架
from adversarial_attack_framework import (
    TextFoolerAttack,
    SynonymGenerator,
    SemanticSimilarityChecker,
    LLMEvaluator
)

# 设置中文字体（兼容Linux和Windows）
import platform
if platform.system() == 'Linux':
    # Linux系统使用DejaVu Sans或不设置中文（使用英文标签）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    USE_CHINESE = False  # Linux下使用英文标签
else:
    # Windows系统使用SimHei
    plt.rcParams['font.sans-serif'] = ['SimHei']
    USE_CHINESE = True
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class AdversarialExperiment:
    # 对抗攻击实验类
    
    def __init__(self, model_path, tokenizer_path='bert-base-chinese', 
                 use_llm=False, deepseek_api_keys=None):
        print("\n初始化对抗攻击实验...")
        
        # 加载tokenizer
        print(f"加载tokenizer: {tokenizer_path}")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = FraudDetectionModel()
        self.model.to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # 初始化攻击组件
        print("初始化TextFooler攻击组件...")
        self.synonym_generator = SynonymGenerator(method='bert', model_path='bert-base-chinese')
        self.similarity_checker = SemanticSimilarityChecker(method='cosine', model_path='bert-base-chinese')
        self.attacker = TextFoolerAttack(
            self.model, 
            self.tokenizer, 
            self.synonym_generator, 
            self.similarity_checker
        )
        
        # 初始化LLM评估器（可选）
        self.use_llm = use_llm
        self.llm_evaluator = None
        if use_llm and deepseek_api_keys:
            print(f"初始化DeepSeek LLM评估器（{len(deepseek_api_keys)}个API key）")
            self.llm_evaluator = LLMEvaluator(
                method='deepseek',
                api_keys=deepseek_api_keys,
                base_url="https://api.deepseek.com"
            )
        else:
            print("不使用LLM评估（仅TextFooler攻击）")
        
        print("初始化完成!\n")
    
    def evaluate_baseline(self, data_path, num_samples=500, output_dir='adversarial_results'):
        # 基线实验：评估模型在原始测试集上的性能
        print("=" * 80)
        print("基线实验：评估原始测试集性能")
        print("=" * 80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        print(f"\n加载数据: {data_path}")
        dataset = load_dataset(data_path)
        if len(dataset) > num_samples:
            dataset = dataset[:num_samples]
        print(f"数据集大小: {len(dataset)}")
        
        # 评估模型
        print("\n开始评估...")
        predictions = []
        true_labels = []
        
        for item in tqdm(dataset, desc="评估进度"):
            text = item['text']
            label = 1 if item['is_fraud'] else 0
            
            # 预测
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, 
                                   truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs, dim=1).item()
            
            predictions.append(pred)
            true_labels.append(label)
        
        # 计算指标
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
        cm = confusion_matrix(true_labels, predictions)
        
        # 分别计算诈骗和非诈骗样本的准确率
        fraud_mask = [l == 1 for l in true_labels]
        nonfraud_mask = [l == 0 for l in true_labels]
        
        fraud_acc = accuracy_score(
            [l for l, m in zip(true_labels, fraud_mask) if m],
            [p for p, m in zip(predictions, fraud_mask) if m]
        ) if sum(fraud_mask) > 0 else 0
        
        nonfraud_acc = accuracy_score(
            [l for l, m in zip(true_labels, nonfraud_mask) if m],
            [p for p, m in zip(predictions, nonfraud_mask) if m]
        ) if sum(nonfraud_mask) > 0 else 0
        
        metrics = {
            'total': len(dataset),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fraud_accuracy': fraud_acc,
            'nonfraud_accuracy': nonfraud_acc,
            'confusion_matrix': cm.tolist()
        }
        
        # 输出结果
        print("\n" + "=" * 60)
        print("基线实验结果")
        print("=" * 60)
        print(f"总样本数: {metrics['total']}")
        print(f"总体准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  - 诈骗样本准确率: {metrics['fraud_accuracy']:.4f} ({metrics['fraud_accuracy']*100:.2f}%)")
        print(f"  - 非诈骗样本准确率: {metrics['nonfraud_accuracy']:.4f} ({metrics['nonfraud_accuracy']*100:.2f}%)")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        print(f"\n混淆矩阵:")
        print(f"                预测非诈骗  预测诈骗")
        print(f"实际非诈骗        {cm[0][0]:6d}    {cm[0][1]:6d}")
        print(f"实际诈骗          {cm[1][0]:6d}    {cm[1][1]:6d}")
        print("=" * 60)
        
        # 保存结果
        results_file = os.path.join(output_dir, 'baseline_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {results_file}")
        
        return metrics
    
    def run_attack_experiment(self, data_path, num_samples=500, output_dir='adversarial_results'):
        # 执行对抗攻击实验
        print("=" * 80)
        print("开始对抗攻击实验")
        print("=" * 80)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        print(f"\n加载数据: {data_path}")
        dataset = load_dataset(data_path)
        
        # 限制样本数
        if len(dataset) > num_samples:
            dataset = dataset[:num_samples]
        
        print(f"数据集大小: {len(dataset)}")
        
        # 执行攻击
        results = []
        success_count = 0
        
        print("\n执行对抗攻击...")
        for i, item in enumerate(tqdm(dataset, desc="攻击进度")):
            text = item['text']
            label = 1 if item['is_fraud'] else 0
            
            # 执行攻击
            result = self.attacker.attack(
                text=text,
                true_label=label,
                max_changes=10,
                similarity_threshold=0.8
            )
            
            # 使用LLM评估（无论攻击成功与否都评分）
            if self.use_llm and self.llm_evaluator:
                llm_result = self.llm_evaluator.evaluate_with_score(
                    result['original_text'],
                    result['adversarial_text']
                )
                result['llm_score'] = llm_result['score']
                result['llm_is_consistent'] = llm_result['is_consistent']
                result['llm_reason'] = llm_result['reason']
                result['llm_suggested_text'] = llm_result['suggested_text']
                
                # 打印前10条样本的LLM评分与改写
                if i < 10:
                    print(f"LLM评分: {result['llm_score']} (一致性: {result['llm_is_consistent']})")
                    if result['llm_suggested_text']:
                        print("LLM改写文本:")
                        print(result['llm_suggested_text'])
                        print("-" * 60)
                
                # 统计有效样本（攻击成功且LLM评分>=3）
                if result['success'] and llm_result['score'] >= 3:
                    success_count += 1
            else:
                result['llm_score'] = None
                result['llm_is_consistent'] = None
                result['llm_reason'] = 'no_llm_evaluator'
                result['llm_suggested_text'] = None
            
            results.append(result)
            
            # 每100个样本保存一次
            if (i + 1) % 100 == 0:
                self._save_intermediate_results(results, output_dir, i + 1)
        
        # 保存最终结果
        print("\n保存实验结果...")
        self._save_results(results, output_dir)
        
        # 生成统计报告
        print("\n生成统计报告...")
        self._generate_report(results, output_dir)
        
        print("\n" + "=" * 80)
        print("实验完成!")
        print(f"结果保存在: {output_dir}")
        print("=" * 80)
    
    def _save_intermediate_results(self, results, output_dir, count):
        # 保存中间结果
        df = pd.DataFrame(results)
        output_file = os.path.join(output_dir, f'intermediate_results_{count}.csv')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    def _save_results(self, results, output_dir):
        # 保存实验结果
        # 保存完整结果
        df = pd.DataFrame(results)
        output_file = os.path.join(output_dir, 'adversarial_results_full.csv')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 保存成功的对抗样本
        success_df = df[df['success'] == True]
        success_file = os.path.join(output_dir, 'adversarial_results_success.csv')
        success_df.to_csv(success_file, index=False, encoding='utf-8-sig')
        
        # 保存LLM评分>=3的高质量样本
        if 'llm_score' in df.columns:
            high_quality_df = df[df['llm_score'] >= 3]
            high_quality_file = os.path.join(output_dir, 'adversarial_results_high_quality.csv')
            high_quality_df.to_csv(high_quality_file, index=False, encoding='utf-8-sig')
            
            # 保存评分<3且有改写建议的样本
            low_score_df = df[(df['llm_score'] < 3) & (df['llm_suggested_text'].notna())]
            low_score_file = os.path.join(output_dir, 'adversarial_results_low_score_with_rewrite.csv')
            low_score_df.to_csv(low_score_file, index=False, encoding='utf-8-sig')
            
            print(f"完整结果: {output_file}")
            print(f"成功样本: {success_file}")
            print(f"LLM高质量样本(评分>=3): {high_quality_file}")
            print(f"LLM低分样本(含改写): {low_score_file}")
        else:
            print(f"完整结果: {output_file}")
            print(f"成功样本: {success_file}")
    
    def _generate_report(self, results, output_dir):
        # 生成统计报告
        total = len(results)
        success = sum(1 for r in results if r['success'])
        
        # LLM评分统计
        llm_scored = sum(1 for r in results if r.get('llm_score') is not None)
        llm_high_quality = sum(1 for r in results if r.get('llm_score', 0) >= 3)
        llm_low_score = sum(1 for r in results if r.get('llm_score', 0) < 3)
        llm_with_rewrite = sum(1 for r in results if r.get('llm_score', 0) < 3 and r.get('llm_suggested_text'))
        
        success_results = [r for r in results if r['success']]
        avg_changes = np.mean([r['num_changes'] for r in success_results]) if success_results else 0
        avg_similarity = np.mean([r['similarity'] for r in success_results]) if success_results else 0
        
        # 计算各种比率（避免除零错误）
        attack_success_rate = (success/total*100) if total > 0 else 0.0
        llm_high_quality_rate = (llm_high_quality/llm_scored*100) if llm_scored > 0 else 0.0
        llm_rewrite_rate = (llm_with_rewrite/llm_low_score*100) if llm_low_score > 0 else 0.0
        
        report = f"""
{'=' * 80}
对抗攻击实验报告
{'=' * 80}

实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

一、基本统计
{'=' * 80}
总样本数: {total}
攻击成功数: {success}
攻击成功率: {attack_success_rate:.2f}%

LLM评分统计:
  已评分样本数: {llm_scored}
  高质量样本(评分>=3): {llm_high_quality} ({llm_high_quality_rate:.2f}%)
  低分样本(评分<3): {llm_low_score}
  低分样本含改写建议: {llm_with_rewrite} ({llm_rewrite_rate:.2f}%)

二、攻击效果
{'=' * 80}
平均改变词数: {avg_changes:.2f}
平均语义相似度: {avg_similarity:.3f}

三、模型准确率变化
{'=' * 80}
"""
        
        # 计算准确率变化
        original_correct = sum(1 for r in results if r['original_pred'] == r['original_label'])
        adversarial_correct = sum(1 for r in results if r['adversarial_pred'] == r['original_label'])
        
        original_acc = original_correct / total * 100
        adversarial_acc = adversarial_correct / total * 100
        acc_drop = original_acc - adversarial_acc
        
        report += f"""原始准确率: {original_acc:.2f}%
攻击后准确率: {adversarial_acc:.2f}%
准确率下降: {acc_drop:.2f}%

四、详细分析
{'=' * 80}
"""
        
        # 按标签分析（避免除零错误）
        fraud_results = [r for r in results if r['original_label'] == 1]
        non_fraud_results = [r for r in results if r['original_label'] == 0]
        
        fraud_success = sum(1 for r in fraud_results if r['success'])
        non_fraud_success = sum(1 for r in non_fraud_results if r['success'])
        
        # 计算成功率（避免除零）
        fraud_success_rate = (fraud_success/len(fraud_results)*100) if len(fraud_results) > 0 else 0.0
        non_fraud_success_rate = (non_fraud_success/len(non_fraud_results)*100) if len(non_fraud_results) > 0 else 0.0
        
        report += f"""
欺诈样本:
  总数: {len(fraud_results)}
  攻击成功: {fraud_success}
  成功率: {fraud_success_rate:.2f}%

非欺诈样本:
  总数: {len(non_fraud_results)}
  攻击成功: {non_fraud_success}
  成功率: {non_fraud_success_rate:.2f}%

{'=' * 80}
"""
        
        # 保存报告
        report_file = os.path.join(output_dir, 'experiment_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"\n报告已保存: {report_file}")
    
    def analyze_cases(self, results_file, num_cases=5):
        # 分析典型案例
        print("=" * 80)
        print("典型案例分析")
        print("=" * 80)
        
        df = pd.read_csv(results_file)
        
        # 成功案例
        success_df = df[df['success'] == True].head(num_cases)
        print("\n【攻击成功案例】")
        for i, row in success_df.iterrows():
            print(f"\n案例 {i+1}:")
            print(f"原文: {row['original_text'][:100]}...")
            print(f"改写: {row['adversarial_text'][:100]}...")
            print(f"真实标签: {'欺诈' if row['original_label'] == 1 else '非欺诈'}")
            print(f"原始预测: {'欺诈' if row['original_pred'] == 1 else '非欺诈'}")
            print(f"攻击后预测: {'欺诈' if row['adversarial_pred'] == 1 else '非欺诈'}")
            print(f"改变词数: {row['num_changes']}")
            print(f"语义相似度: {row['similarity']:.3f}")
            if 'llm_valid' in row and row['llm_valid']:
                print(f"LLM验证: 通过 ✓")
        
        # 失败案例
        fail_df = df[df['success'] == False].head(num_cases)
        print("\n【攻击失败案例】")
        for i, row in fail_df.iterrows():
            print(f"\n案例 {i+1}:")
            print(f"原文: {row['original_text'][:100]}...")
            print(f"真实标签: {'欺诈' if row['original_label'] == 1 else '非欺诈'}")
            print(f"失败原因: {row.get('reason', '未知')}")
    
    def analyze_attack_mechanism(self, results_file, output_dir, num_cases=10):
        # 深入分析攻击机制：为什么某些改写能骗过模型
        print("\n" + "=" * 80)
        print("深入分析：为什么某些改写能骗过模型")
        print("=" * 80)
        
        df = pd.read_csv(results_file)
        success_df = df[df['success'] == True]
        
        if len(success_df) == 0:
            print("没有成功的攻击样本")
            return
        
        # 分析1：词替换模式分析
        print("\n【分析1：词替换模式】")
        print("-" * 60)
        
        # 统计改变词数分布
        num_changes_dist = success_df['num_changes'].value_counts().sort_index()
        print("\n改变词数分布：")
        for num, count in num_changes_dist.items():
            print(f"  {num}个词: {count}个样本 ({count/len(success_df)*100:.1f}%)")
        
        # 分析2：语义相似度与攻击成功的关系
        print("\n【分析2：语义相似度分析】")
        print("-" * 60)
        
        similarity_bins = pd.cut(success_df['similarity'], bins=[0, 0.7, 0.8, 0.9, 1.0])
        similarity_dist = similarity_bins.value_counts().sort_index()
        print("\n语义相似度分布：")
        for bin_range, count in similarity_dist.items():
            print(f"  {bin_range}: {count}个样本 ({count/len(success_df)*100:.1f}%)")
        
        # 分析3：典型成功案例详细分析
        print("\n【分析3：典型成功案例详细分析】")
        print("-" * 60)
        
        # 选择不同改变词数的代表性案例
        analysis_report = []
        
        for num_change in [1, 2, 3, 4, 5]:
            cases = success_df[success_df['num_changes'] == num_change]
            if len(cases) > 0:
                case = cases.iloc[0]
                
                print(f"\n案例（改变{num_change}个词）：")
                print(f"原文: {case['original_text'][:150]}")
                print(f"改写: {case['adversarial_text'][:150]}")
                print(f"预测变化: {'欺诈' if case['original_pred'] == 1 else '非欺诈'} → {'欺诈' if case['adversarial_pred'] == 1 else '非欺诈'}")
                print(f"语义相似度: {case['similarity']:.3f}")
                
                # 分析词替换
                original_words = set(case['original_text'].split())
                adversarial_words = set(case['adversarial_text'].split())
                
                print(f"\n分析：")
                print(f"  - 通过替换{num_change}个词成功改变了模型预测")
                print(f"  - 语义相似度保持在{case['similarity']:.1%}")
                print(f"  - 说明模型对某些关键词敏感，同义词替换可以绕过检测")
                
                analysis_report.append({
                    'num_changes': num_change,
                    'original': case['original_text'],
                    'adversarial': case['adversarial_text'],
                    'similarity': case['similarity'],
                    'pred_change': f"{case['original_pred']} → {case['adversarial_pred']}"
                })
        
        # 分析4：攻击成功的关键因素
        print("\n【分析4：攻击成功的关键因素总结】")
        print("-" * 60)
        
        avg_changes = success_df['num_changes'].mean()
        avg_similarity = success_df['similarity'].mean()
        
        print(f"\n关键发现：")
        print(f"1. 平均只需替换 {avg_changes:.1f} 个词就能成功攻击")
        print(f"2. 攻击后语义相似度平均保持在 {avg_similarity:.1%}")
        print(f"3. 说明模型存在明显的脆弱性：")
        print(f"   - 对同义词替换敏感")
        print(f"   - 可能过度依赖某些关键词特征")
        print(f"   - 缺乏对语义整体理解的鲁棒性")
        
        # 保存详细分析报告
        report_file = os.path.join(output_dir, 'attack_mechanism_analysis.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("深入分析：为什么某些改写能骗过模型\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("一、词替换模式分析\n")
            f.write("-" * 60 + "\n")
            for num, count in num_changes_dist.items():
                f.write(f"改变{num}个词: {count}个样本 ({count/len(success_df)*100:.1f}%)\n")
            
            f.write("\n二、关键发现\n")
            f.write("-" * 60 + "\n")
            f.write(f"平均替换词数: {avg_changes:.2f}\n")
            f.write(f"平均语义相似度: {avg_similarity:.3f}\n")
            f.write(f"\n三、模型脆弱性分析\n")
            f.write("1. 模型对同义词替换高度敏感\n")
            f.write("2. 可能过度依赖表层词汇特征\n")
            f.write("3. 缺乏深层语义理解能力\n")
        
        print(f"\n详细分析报告已保存: {report_file}")
        
        return analysis_report
    
    def run_ablation_study(self, data_path, num_samples=200, output_dir='ablation_results'):
        # 消融实验：对比不同替换策略的影响
        print("\n" + "=" * 80)
        print("消融实验：对比不同替换策略")
        print("=" * 80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        dataset = load_dataset(data_path)
        if len(dataset) > num_samples:
            dataset = dataset[:num_samples]
        
        ablation_results = {}
        
        # 实验1：只替换1个词
        print("\n【实验A：最多替换1个词】")
        self.attacker.max_candidates = 1
        results_1 = self._run_attack_batch(dataset, max_changes=1)
        ablation_results['max_1_word'] = self._compute_metrics(results_1)
        self._save_ablation_results(results_1, os.path.join(output_dir, 'ablation_1word.csv'))
        
        # 实验2：替换2-3个词
        print("\n【实验B：最多替换3个词】")
        self.attacker.max_candidates = 5
        results_3 = self._run_attack_batch(dataset, max_changes=3)
        ablation_results['max_3_words'] = self._compute_metrics(results_3)
        self._save_ablation_results(results_3, os.path.join(output_dir, 'ablation_3words.csv'))
        
        # 实验3：替换4-5个词
        print("\n【实验C：最多替换5个词】")
        results_5 = self._run_attack_batch(dataset, max_changes=5)
        ablation_results['max_5_words'] = self._compute_metrics(results_5)
        self._save_ablation_results(results_5, os.path.join(output_dir, 'ablation_5words.csv'))
        
        # 生成对比报告
        self._generate_ablation_report(ablation_results, output_dir)
        
        return ablation_results
    
    def _run_attack_batch(self, dataset, max_changes=5):
        # 执行批量攻击
        results = []
        for item in tqdm(dataset, desc=f"攻击(max={max_changes})"):
            result = self.attacker.attack(
                item['text'],
                item['is_fraud'],
                max_changes=max_changes
            )
            results.append(result)
        return results
    
    def _compute_metrics(self, results):
        # 计算评估指标
        total = len(results)
        success = sum(1 for r in results if r['success'])
        
        success_results = [r for r in results if r['success']]
        avg_changes = np.mean([r['num_changes'] for r in success_results]) if success_results else 0
        avg_similarity = np.mean([r['similarity'] for r in success_results]) if success_results else 0
        
        original_correct = sum(1 for r in results if r['original_pred'] == r['original_label'])
        adversarial_correct = sum(1 for r in results if r['adversarial_pred'] == r['original_label'])
        
        return {
            'total': total,
            'success': success,
            'success_rate': success / total if total > 0 else 0,
            'avg_changes': avg_changes,
            'avg_similarity': avg_similarity,
            'original_acc': original_correct / total if total > 0 else 0,
            'adversarial_acc': adversarial_correct / total if total > 0 else 0,
            'acc_drop': (original_correct - adversarial_correct) / total if total > 0 else 0
        }
    
    def _save_ablation_results(self, results, filepath):
        # 保存消融实验结果
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    def _generate_ablation_report(self, ablation_results, output_dir):
        # 生成消融实验对比报告
        print("\n" + "=" * 80)
        print("消融实验对比结果")
        print("=" * 80)
        
        print(f"\n{'实验组':<20} {'攻击成功率':<15} {'平均替换词数':<15} {'准确率下降':<15}")
        print("-" * 65)
        
        for name, metrics in ablation_results.items():
            display_name = {
                'max_1_word': '最多1个词',
                'max_3_words': '最多3个词',
                'max_5_words': '最多5个词'
            }.get(name, name)
            
            print(f"{display_name:<18} {metrics['success_rate']:>6.2%}         {metrics['avg_changes']:>6.2f}           {metrics['acc_drop']:>6.2%}")
        
        print("=" * 80)
        
        # 保存JSON报告
        report_file = os.path.join(output_dir, 'ablation_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(ablation_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n消融实验报告已保存: {report_file}")


def main():
    # 主函数
    parser = argparse.ArgumentParser(description='对抗攻击实验')
    parser.add_argument('--mode', type=str, default='compare', 
                       choices=['baseline', 'attack', 'analyze', 'compare', 'ablation', 'deep_analyze'],
                       help='运行模式: baseline(基线), attack(攻击), analyze(分析), compare(完整对比), ablation(消融实验), deep_analyze(深入分析)')
    parser.add_argument('--model_path', type=str, default='bert-model/fraud_detection_model.pt',
                       help='模型路径')
    parser.add_argument('--data_path', type=str, default='测试集结果.csv',
                       help='数据集路径')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='攻击样本数')
    parser.add_argument('--output_dir', type=str, default='adversarial_results',
                       help='输出目录')
    parser.add_argument('--results_file', type=str, default='adversarial_results/adversarial_results_full.csv',
                       help='结果文件路径(用于分析模式)')
    parser.add_argument('--use_llm', action='store_true',
                       help='是否使用DeepSeek LLM评估（默认不使用）')
    
    args = parser.parse_args()
    
    # 检查模型和数据文件
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        print("请先训练模型: python bert2.py")
        return
    
    if args.mode != 'analyze' and not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        return
    
    # 加载DeepSeek配置（如果需要）
    deepseek_api_keys = None
    if args.use_llm or args.mode == 'compare':
        try:
            from deepseek_config import DEEPSEEK_API_KEYS
            deepseek_api_keys = DEEPSEEK_API_KEYS
            print(f"已加载DeepSeek配置（{len(deepseek_api_keys)}个API key）")
        except ImportError:
            if args.use_llm:
                print("错误: 无法加载deepseek_config.py")
                print("请确保deepseek_config.py存在并包含DEEPSEEK_API_KEYS")
                return
            else:
                print("警告: 未配置DeepSeek API，compare模式将跳过实验二")
    
    if args.mode == 'baseline':
        # 基线实验
        experiment = AdversarialExperiment(args.model_path)
        experiment.evaluate_baseline(
            data_path=args.data_path,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
    
    elif args.mode == 'attack':
        # 单次攻击实验
        experiment = AdversarialExperiment(
            args.model_path,
            use_llm=args.use_llm,
            deepseek_api_keys=deepseek_api_keys
        )
        experiment.run_attack_experiment(
            data_path=args.data_path,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
    
    elif args.mode == 'compare':
        # 完整对比实验：基线 + 实验一 + 实验二
        print("\n" + "=" * 80)
        print("开始完整对比实验（三组实验）")
        print("=" * 80)
        
        # 实验1：基线评估
        print("\n【实验0：基线评估】")
        exp_baseline = AdversarialExperiment(args.model_path)
        baseline_metrics = exp_baseline.evaluate_baseline(
            data_path=args.data_path,
            num_samples=args.num_samples,
            output_dir=os.path.join(args.output_dir, 'baseline')
        )
        
        # 实验1：TextFooler攻击（不用LLM评估）
        print("\n【实验1：TextFooler对抗攻击】")
        exp1 = AdversarialExperiment(
            args.model_path,
            use_llm=False
        )
        exp1.run_attack_experiment(
            data_path=args.data_path,
            num_samples=args.num_samples,
            output_dir=os.path.join(args.output_dir, 'textfooler')
        )
        
        # 实验2：TextFooler + LLM评估
        print("\n【实验2：TextFooler + LLM评估】")
        if deepseek_api_keys:
            exp2 = AdversarialExperiment(
                args.model_path,
                use_llm=True,
                deepseek_api_keys=deepseek_api_keys
            )
            exp2.run_attack_experiment(
                data_path=args.data_path,
                num_samples=args.num_samples,
                output_dir=os.path.join(args.output_dir, 'textfooler_llm')
            )
        else:
            print("警告: 未配置DeepSeek API，跳过实验2")
            print("请在deepseek_config.py中配置API密钥")
        
        # 生成对比报告
        print("\n" + "=" * 80)
        print("生成对比报告")
        print("=" * 80)
        generate_comparison_report(args.output_dir)
    
    elif args.mode == 'analyze':
        # 分析结果
        if not os.path.exists(args.results_file):
            print(f"错误: 结果文件不存在: {args.results_file}")
            return
        
        experiment = AdversarialExperiment(args.model_path)
        experiment.analyze_cases(args.results_file, num_cases=5)
    
    elif args.mode == 'ablation':
        # 消融实验
        experiment = AdversarialExperiment(args.model_path)
        experiment.run_ablation_study(
            data_path=args.data_path,
            num_samples=args.num_samples,
            output_dir=os.path.join(args.output_dir, 'ablation')
        )
    
    elif args.mode == 'deep_analyze':
        # 深入分析
        if not os.path.exists(args.results_file):
            print(f"错误: 结果文件不存在: {args.results_file}")
            return
        
        experiment = AdversarialExperiment(args.model_path)
        experiment.analyze_attack_mechanism(
            results_file=args.results_file,
            output_dir=os.path.dirname(args.results_file),
            num_cases=10
        )


def generate_comparison_report(output_dir):
    """
    生成三组实验的对比报告
    
    Args:
        output_dir: 输出目录
    """
    print("\n读取实验结果...")
    
    # 读取基线结果
    baseline_file = os.path.join(output_dir, 'baseline', 'baseline_results.json')
    exp1_file = os.path.join(output_dir, 'textfooler', 'attack_summary.json')
    exp2_file = os.path.join(output_dir, 'textfooler_llm', 'attack_summary.json')
    
    results = {}
    
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r', encoding='utf-8') as f:
            results['baseline'] = json.load(f)
    
    if os.path.exists(exp1_file):
        with open(exp1_file, 'r', encoding='utf-8') as f:
            results['textfooler'] = json.load(f)
    
    if os.path.exists(exp2_file):
        with open(exp2_file, 'r', encoding='utf-8') as f:
            results['textfooler_llm'] = json.load(f)
    
    # 生成对比表格
    print("\n" + "=" * 80)
    print("三组实验对比结果")
    print("=" * 80)
    
    print(f"\n{'实验组':<20} {'准确率':<15} {'准确率下降':<15} {'样本数':<10}")
    print("-" * 60)
    
    baseline_acc = results.get('baseline', {}).get('accuracy', 0)
    
    print(f"{'基线（原始数据）':<18} {baseline_acc:>6.2%}         -              {results.get('baseline', {}).get('total', 0):>6}")
    
    if 'textfooler' in results:
        tf_acc = results['textfooler'].get('accuracy_after_attack', 0)
        tf_drop = baseline_acc - tf_acc
        print(f"{'实验一（TextFooler）':<18} {tf_acc:>6.2%}      {tf_drop:>6.2%}        {results['textfooler'].get('total_samples', 0):>6}")
    
    if 'textfooler_llm' in results:
        tfllm_acc = results['textfooler_llm'].get('accuracy_after_attack', 0)
        tfllm_drop = baseline_acc - tfllm_acc
        print(f"{'实验二（TF+LLM）':<18} {tfllm_acc:>6.2%}      {tfllm_drop:>6.2%}        {results['textfooler_llm'].get('total_samples', 0):>6}")
    
    print("=" * 80)
    
    # 保存对比报告
    report_file = os.path.join(output_dir, 'comparison_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n对比报告已保存到: {report_file}")
    print("\n实验完成！")


if __name__ == "__main__":
    main()
