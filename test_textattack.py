# 简化版TextAttack测试脚本 - 测试3条数据
import os
import sys
import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader

# 添加父目录到路径以导入bert2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 从bert2导入所需的类和函数
from bert2 import (
    FraudDetectionModel,
    FraudDetectionDataset,
    load_dataset,
    predict,
    calculate_metrics,
    device,
    MODEL_PATH
)

# 配置
TEST_PATH = '测试集结果.csv'

print("="*80)
print("步骤1: 加载模型（使用bert2.py的方法）")
print("="*80)

# 加载tokenizer
if os.path.exists(os.path.join(MODEL_PATH, 'tokenizer')):
    tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_PATH, 'tokenizer'))
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 创建模型并加载权重（使用bert2的方法）
model = FraudDetectionModel()
model.to(device)
model_file = os.path.join(MODEL_PATH, 'fraud_detection_model.pt')
model.load_trained_model(model_file)  # 使用bert2的加载方法
model.eval()
print(f"✓ 模型加载成功，设备: {device}")

# ============================================================================
# 2. 测试TextAttack导入
# ============================================================================
print("\n" + "="*80)
print("步骤2: 测试TextAttack导入")
print("="*80)

try:
    from textattack.models.wrappers import ModelWrapper
    from textattack.attack_recipes import TextFoolerJin2019
    from textattack.datasets import Dataset
    from textattack import Attacker, AttackArgs
    from textattack.transformations import ChineseWordSwapMaskedLM
    
    # 下载必要的NLTK数据
    import nltk
    print("下载NLTK数据...")
    try:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('universal_tagset', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        pass
    
    print("✓ TextAttack导入成功")
except ImportError as e:
    print(f"✗ TextAttack导入失败: {e}")
    print("请先运行: pip install textattack")
    exit(1)

# ============================================================================
# 3. 创建模型包装器
# ============================================================================
print("\n" + "="*80)
print("步骤3: 创建TextAttack模型包装器")
print("="*80)

class BertFraudModelWrapper(ModelWrapper):
    # TextAttack模型包装器
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def __call__(self, text_input_list):
        # 返回每个类别的概率
        encodings = self.tokenizer(
            text_input_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.model.bert.device)
        attention_mask = encodings['attention_mask'].to(self.model.bert.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu()

model_wrapper = BertFraudModelWrapper(model, tokenizer)
print("✓ 模型包装器创建成功")

# ============================================================================
# 4. 测试模型包装器
# ============================================================================
print("\n" + "="*80)
print("步骤4: 测试模型包装器")
print("="*80)

test_text = "left: 你好，这是客服中心。right: 什么事？"
probs = model_wrapper([test_text])
pred = torch.argmax(probs, dim=1).item()
print(f"测试文本: {test_text}")
print(f"预测概率: {probs[0].tolist()}")
print(f"预测类别: {'诈骗' if pred == 1 else '正常'}")
print("✓ 模型包装器测试成功")

# ============================================================================
# 5. 加载测试数据
# ============================================================================
print("\n" + "="*80)
print("步骤5: 加载测试数据")
print("="*80)

df = pd.read_csv(TEST_PATH)
test_samples = []
for idx, row in df.head(3).iterrows():
    text = row['specific_dialogue_content']
    label = 1 if row['is_fraud'] == 1 else 0
    test_samples.append((text, label))

print(f"✓ 加载了 {len(test_samples)} 条测试样本")
for i, (text, label) in enumerate(test_samples):
    print(f"  样本{i+1}: 标签={'诈骗' if label==1 else '正常'}, 长度={len(text)}字")

# ============================================================================
# 6. 创建TextAttack数据集
# ============================================================================
print("\n" + "="*80)
print("步骤6: 创建TextAttack数据集")
print("="*80)

dataset = Dataset(test_samples)
print(f"✓ TextAttack数据集创建成功，包含 {len(dataset)} 条样本")

# ============================================================================
# 7. 使用TextFooler recipe攻击
# ============================================================================
print("\n" + "="*80)
print("步骤7: 使用TextFooler Recipe攻击")
print("="*80)

try:
    # 使用中文TextFooler攻击（按照官方文档）
    from textattack.transformations import ChineseWordSwapMaskedLM
    from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
    from textattack.constraints.semantics.sentence_encoders import MultilingualUniversalSentenceEncoder
    from textattack.search_methods import GreedyWordSwapWIR
    from textattack.goal_functions import UntargetedClassification
    from textattack import Attack
    import string
    
    # 目标函数
    goal_function = UntargetedClassification(model_wrapper)
    
    # 转换：使用中文BERT MLM（使用默认的xlm-roberta-base）
    transformation = ChineseWordSwapMaskedLM(
        model="bert-base-chinese"
    )
    
    # 约束：中文停用词
    chinese_stopwords = set([
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", 
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
        "看", "好", "自己", "这", "那", "他", "她", "它", "们", "吗", "吧", "呢",
        "啊", "哦", "嗯", "哎", "呀", "、", "。", "，", "！", "？", "；", "：",
        "left", "right", ":", "："
    ])
    chinese_stopwords = chinese_stopwords.union(set(string.punctuation))
    
    # 创建MUSE语义相似度约束（按照官方示例）
    print("加载MUSE语义编码器...")
    muse = MultilingualUniversalSentenceEncoder(
        threshold=0.9,  # 语义相似度阈值（0.9表示90%相似）
        metric="cosine",
        compare_against_original=True,
        window_size=3,  # 缩小窗口以适配中文短文本
        skip_text_shorter_than_window=False,
    )
    
    constraints = [
        RepeatModification(),
        StopwordModification(stopwords=chinese_stopwords),
        muse  # 添加语义相似度约束
    ]
    print(f"✓ 已添加 {len(constraints)} 个约束（包括MUSE语义约束）")
    
    # 搜索方法
    search_method = GreedyWordSwapWIR(wir_method="weighted-saliency")
    
    # 创建攻击
    attack = Attack(goal_function, constraints, transformation, search_method)
    print("✓ 中文TextFooler攻击创建成功")
    
    # 配置攻击参数（优化GPU性能）
    attack_args = AttackArgs(
        num_examples=3,
        log_to_csv="textattack_simple_results.csv",
        disable_stdout=False,

    )
    
    # 执行攻击
    print("\n开始攻击...")
    attacker = Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()
    
    # 统计结果并输出详细信息
    print("\n" + "="*80)
    print("攻击结果统计")
    print("="*80)
    
    # TextAttack的结果对象类型判断
    from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult
    
    successful = sum([isinstance(r, SuccessfulAttackResult) for r in results])
    failed = sum([isinstance(r, FailedAttackResult) for r in results])
    skipped = sum([isinstance(r, SkippedAttackResult) for r in results])
    
    print(f"成功: {successful}, 失败: {failed}, 跳过: {skipped}")
    if successful + failed > 0:
        print(f"攻击成功率: {successful/(successful+failed)*100:.2f}%")
    
    # 输出每个样本的详细对抗样本
    print("\n" + "="*80)
    print("对抗样本详情")
    print("="*80)
    
    for i, result in enumerate(results):
        print(f"\n{'='*80}")
        print(f"【样本 {i+1}】")
        print(f"{'='*80}")
        
        # 获取原始文本
        try:
            original_text = result.original_text()
            print(f"原始文本:\n{original_text}\n")
        except Exception as e:
            print(f"获取原始文本失败: {e}")
            original_text = str(result.original_result.attacked_text.text) if hasattr(result, 'original_result') else "未知"
            print(f"原始文本:\n{original_text}\n")
        
        if isinstance(result, SuccessfulAttackResult):
            print(f"状态: ✓ 攻击成功\n")
            
            # 获取对抗文本
            try:
                perturbed_text = result.perturbed_text()
                print(f"对抗文本:\n{perturbed_text}\n")
            except Exception as e:
                print(f"获取对抗文本失败: {e}")
                perturbed_text = str(result.perturbed_result.attacked_text.text)
                print(f"对抗文本:\n{perturbed_text}\n")
            
            # 预测结果
            print(f"原始预测: {result.original_result.output}")
            print(f"对抗预测: {result.perturbed_result.output}\n")
            
            # 显示具体替换的词
            try:
                original_words = result.original_result.attacked_text.words
                perturbed_words = result.perturbed_result.attacked_text.words
                diff_indices = result.original_result.attacked_text.all_words_diff(result.perturbed_result.attacked_text)
                
                print(f"替换词数: {len(diff_indices)}")
                if len(diff_indices) > 0:
                    print(f"具体替换:")
                    for idx in diff_indices:
                        print(f"  位置{idx}: '{original_words[idx]}' → '{perturbed_words[idx]}'")
            except Exception as e:
                print(f"获取替换详情失败: {e}")
                
        elif isinstance(result, FailedAttackResult):
            print(f"状态: ✗ 攻击失败\n")
            print(f"原始预测: {result.original_result.output}\n")
            
            # 尝试获取最后的对抗样本
            try:
                if hasattr(result, 'perturbed_result') and result.perturbed_result:
                    # 获取最后尝试的文本
                    last_text = str(result.perturbed_result.attacked_text.text)
                    print(f"最后尝试的对抗文本:\n{last_text}\n")
                    print(f"最后尝试的预测: {result.perturbed_result.output}\n")
                    
                    # 显示尝试替换的词
                    original_words = result.original_result.attacked_text.words
                    perturbed_words = result.perturbed_result.attacked_text.words
                    diff_indices = result.original_result.attacked_text.all_words_diff(result.perturbed_result.attacked_text)
                    
                    if len(diff_indices) > 0:
                        print(f"尝试替换的词数: {len(diff_indices)}")
                        print(f"具体替换:")
                        for idx in diff_indices:
                            print(f"  位置{idx}: '{original_words[idx]}' → '{perturbed_words[idx]}'")
                    else:
                        print(f"未进行任何替换（约束太严格，没有词被替换）")
                else:
                    print(f"未能生成对抗样本（可能因为约束太严格或查询预算耗尽）")
                    print(f"Result类型: {type(result)}")
                    print(f"Result属性: {dir(result)}")
            except Exception as e:
                print(f"获取失败攻击详情时出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"状态: - 跳过")
    
    print("\n✓ 测试完成！")
    
except Exception as e:
    print(f"✗ 攻击执行失败: {e}")
    import traceback
    traceback.print_exc()
