"""
独立模型训练程序 - 二分类和多分类使用独立的BERT模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置模型保存路径
MODEL_PATH = 'bert-model'

print(f"模型将保存在: {MODEL_PATH}")

# 设置数据路径
TRAIN_PATH = os.path.join("训练集结果.csv")
TEST_PATH = os.path.join("测试集结果.csv")

# 加载数据集
def load_dataset(file_path):
    """
    加载诈骗对话数据集
    
    Args:
        file_path: 数据文件路径（CSV格式）
        
    Returns:
        数据列表
    """
    print(f"加载诈骗对话数据集: {file_path}")
    dataset = []
    
    try:
        import pandas as pd
        # 读取CSV文件
        df = pd.read_csv(file_path)
        print(f"成功加载CSV数据, 共{len(df)}条")
        
        # 提取specific_dialogue_content和is_fraud字段
        for _, row in df.iterrows():
            dialogue_content = row['specific_dialogue_content']
            # 将is_fraud转换为布尔值
            is_fraud = True if row['is_fraud'] == True else False
            
            dataset.append({
                'text': dialogue_content,
                'is_fraud': is_fraud
            })
            
    except Exception as e:
        print(f"加载数据集失败: {e}")
    
    print(f"加载了{len(dataset)}条数据")
    return dataset


# 自定义数据集类
class FraudDetectionDataset(Dataset):
    def __init__(self, data, tokenizer=None, max_length=512):
        """
        初始化数据集
        
        Args:
            data: 数据列表
            tokenizer: 可选的预加载tokenizer
            max_length: 最大序列长度
        """
        self.data = data
        self.max_length = max_length
        
        # 如果提供了tokenizer，直接使用；否则再加载
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                # 尝试从本地加载tokenizer
                if os.path.exists(os.path.join(MODEL_PATH, 'tokenizer')):
                    print("从本地加载tokenizer...")
                    self.tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_PATH, 'tokenizer'))
                else:
                    print("下载tokenizer并保存到本地...")
                    self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=MODEL_PATH)
                    self.tokenizer.save_pretrained(os.path.join(MODEL_PATH, 'tokenizer'))
            except Exception as e:
                print(f"加载tokenizer失败: {e}")
                # 出现错误时尝试重新下载
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=MODEL_PATH)
                self.tokenizer.save_pretrained(os.path.join(MODEL_PATH, 'tokenizer'))

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取单个数据项"""
        item = self.data[idx]
        text = item['text']
        
        # 对文本进行分词和编码
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 将张量转换为所需的形状
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        token_type_ids = encoding['token_type_ids'].squeeze(0)
        
        # 诈骗检测二分类模式
        label = 1 if item['is_fraud'] else 0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': torch.tensor(label, dtype=torch.long)
        }


# 准备数据加载器
def prepare_dataloaders(train_path, test_path, batch_size=40, train_ratio=0.8, test_only=False):
    """
    准备训练集和验证集的数据加载器
    
    Args:
        train_path: 训练集数据路径
        test_path: 测试集数据路径
        batch_size: 批量大小
        train_ratio: 训练集比例
        test_only: 如果为True，只返回测试集加载器
    
    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    # 默认加载器值
    train_loader = None
    val_loader = None
    test_loader = None
    
    # 初始化tokenizer
    if os.path.exists(os.path.join(MODEL_PATH, 'tokenizer')):
        # 如果有保存的tokenizer，使用已保存的
        tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_PATH, 'tokenizer'))
    else:
        # 否则创建新的
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 如果不只是测试模式，加载训练和验证数据
    if not test_only:
        # 加载训练数据
        train_data = load_dataset(train_path)
        
        # 创建数据集实例
        dataset = FraudDetectionDataset(train_data, tokenizer=tokenizer)
        
        # 分割训练集和验证集
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 加载测试数据
    if os.path.exists(test_path):
        test_data = load_dataset(test_path)
        test_dataset = FraudDetectionDataset(test_data, tokenizer=tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, tokenizer


# 诈骗检测模型
class FraudDetectionModel(nn.Module):
    def __init__(self, dropout=0.3):
        super(FraudDetectionModel, self).__init__()
        # 从本地加载BERT模型，如果不存在则下载并保存
        if os.path.exists(os.path.join(MODEL_PATH, 'bert-fraud')):
            print("从本地加载诈骗检测BERT模型...")
            self.bert = BertModel.from_pretrained(os.path.join(MODEL_PATH, 'bert-fraud'))
        else:
            print("下载BERT模型并保存到本地...")
            self.bert = BertModel.from_pretrained('bert-base-chinese', cache_dir=MODEL_PATH)
            self.bert.save_pretrained(os.path.join(MODEL_PATH, 'bert-fraud'))
            
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 诈骗/非诈骗二分类
    
    # 加载已训练模型
    def load_trained_model(self, model_path):
        if os.path.exists(model_path):
            print(f"从{model_path}加载已训练模型...")
            self.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"模型文件{model_path}不存在，将使用新初始化的模型")
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output  # 获取[CLS]标记的表示，用于分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# 训练函数
def train_model(model, train_loader, val_loader, epochs=3, 
                    learning_rate=2e-5, model_save_path=None):
    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate, 
                        eps=1e-8, weight_decay=0.08)
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 记录训练过程
    training_stats = []
    total_start_time = time.time()
    
    # 最佳模型跟踪
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        print(f'\n======== Epoch {epoch + 1} / {epochs} ========\n')
        
        # 记录每个epoch的统计信息
        epoch_start_time = time.time()
        
        # 训练阶段
        print('Training...')
        
        
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0
        
        # 创建进度条
        train_progress_bar = tqdm(train_loader, desc="Training", unit="batch")
        
        model.train()
        for batch in train_progress_bar:
            # 清除之前的梯度
            optimizer.zero_grad()
            
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累计损失和样本数
            total_train_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(logits, 1)
            total_train_correct += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)
            
            # 更新进度条
            train_progress_bar.set_postfix(loss=loss.item())
        
        # 计算平均损失和准确率
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = total_train_correct / total_train_samples
        
        training_time = time.time() - epoch_start_time
        
        print(f"  Average training loss: {avg_train_loss:.4f}")
        print(f"  Training accuracy: {train_accuracy:.4f}")
        print(f"  Training epoch took: {training_time:.2f}s")
        
        # 验证阶段
        print('Validating...')
        
        
        total_val_loss = 0
        total_val_correct = 0
        total_val_samples = 0
        
        # 创建验证集进度条
        val_progress_bar = tqdm(val_loader, desc="Validation")

        model.eval()
        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)
                
                # 前向传播
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                # 计算损失
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(logits, 1)
                total_val_correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)
        
        # 计算平均验证损失和准确率
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = total_val_correct / total_val_samples
        
        validation_time = time.time() - epoch_start_time - training_time
        
        print(f"  Average validation loss: {avg_val_loss:.4f}")
        print(f"  Validation accuracy: {val_accuracy:.4f}")
        print(f"  Validation took: {validation_time:.2f}s")
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
        if model_save_path:
            torch.save(model.state_dict(), model_save_path)
            print(f"  Saved best model with validation accuracy: {best_val_accuracy:.4f}")    
        
        # 收集统计信息
        training_stats.append({
            'epoch': epoch + 1,
            'training_loss': avg_train_loss,
            'validation_loss': avg_val_loss,
            'training_accuracy': train_accuracy,
            'validation_accuracy': val_accuracy,
            'training_time': training_time,
            'validation_time': validation_time
        })
    
    total_training_time = time.time() - total_start_time
    hours = int(total_training_time / 3600)
    minutes = int((total_training_time % 3600) / 60)
    seconds = int(total_training_time % 60)
    
    print(f"\nTotal training time: {hours}h {minutes}m {seconds}s")
    
    return training_stats


# 评估指标计算函数
def calculate_metrics(predictions, true_labels):
    """
    计算评估指标
    
    Args:
        predictions: 模型预测结果列表
        true_labels: 真实标签列表
        
    Returns:
        包含各种指标的字典
    """
    # 总体准确率
    correct = sum([1 for p, t in zip(predictions, true_labels) if p == t])
    total = len(true_labels)
    accuracy = correct / total if total > 0 else 0
    
    # 诈骗样本准确率
    fraud_indices = [i for i, label in enumerate(true_labels) if label == 1]
    fraud_correct = sum([1 for i in fraud_indices if predictions[i] == true_labels[i]])
    fraud_total = len(fraud_indices)
    accuracy_fraud = fraud_correct / fraud_total if fraud_total > 0 else 0
    
    # 非诈骗样本准确率
    nonfraud_indices = [i for i, label in enumerate(true_labels) if label == 0]
    nonfraud_correct = sum([1 for i in nonfraud_indices if predictions[i] == true_labels[i]])
    nonfraud_total = len(nonfraud_indices)
    accuracy_nonfraud = nonfraud_correct / nonfraud_total if nonfraud_total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'accuracy_fraud': accuracy_fraud,
        'accuracy_nonfraud': accuracy_nonfraud,
        'fraud_total': fraud_total,
        'nonfraud_total': nonfraud_total,
        'total': total
    }

# 预测函数
def predict(model, dataloader):
    """
    使用模型进行预测
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        
    Returns:
        predictions: 预测结果列表
        true_labels: 真实标签列表
    """
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测中"):
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 获取预测结果
            logits = outputs
            _, preds = torch.max(logits, dim=1)
            
            # 添加到列表中
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    
    return predictions, true_labels

# 保存预测结果
def save_predictions(predictions, true_labels, file_path):
    """
    保存预测结果到CSV文件
    
    Args:
        predictions: 预测结果列表
        true_labels: 真实标签列表
        file_path: 保存路径
    """
    import pandas as pd
    
    # 创建DataFrame
    df = pd.DataFrame({
        'prediction': predictions,
        'true_label': true_labels
    })
    
    # 保存到CSV
    df.to_csv(file_path, index=False)
    print(f"预测结果已保存到 {file_path}")

# 交互式测试
def interactive_test(model, tokenizer, max_length=256):
    """
    交互式输入文本，输出是否诈骗的判断结果。
    支持多行输入，输入空行或输入 END 结束当前对话输入。
    """
    print("\n=== 交互式测试（多行对话模式） ===")
    print("请输入对话文本，输入空行或 END 结束当前对话，输入 EXIT 退出程序")
    model.eval()
    while True:
        print("\n--- 新对话开始（输入空行或 END 结束当前对话） ---")
        lines = []
        while True:
            try:
                line = input("> ")
                if line.strip().upper() == "EXIT":
                    print("已退出程序。")
                    return
                if not line.strip() or line.strip().upper() == "END":
                    break
                lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\n已退出程序。")
                return
        
        # 如果没有输入任何内容，继续等待输入
        if not lines:
            continue
            
        # 将多行文本合并成一个字符串，保留换行符
        text = "\n".join(lines)

        # 编码
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        token_type_ids = encoding.get('token_type_ids')
        token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None

        # 推理
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()

        label = "诈骗" if pred == 1 else "非诈骗"
        print(f"预测结果: {label}  置信度: {conf*100:.2f}%")

# 绘制训练过程图表
def plot_training_stats(training_stats, title_prefix=""):
    # 提取数据
    epochs = [stat['epoch'] for stat in training_stats]
    train_loss = [stat['training_loss'] for stat in training_stats]
    val_loss = [stat['validation_loss'] for stat in training_stats]
    train_acc = [stat['training_accuracy'] for stat in training_stats]
    val_acc = [stat['validation_accuracy'] for stat in training_stats]
    
    # 创建图表
    plt.figure(figsize=(12, 5))
    
    # 绘制损失图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title(f'{title_prefix} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title(f'{title_prefix} Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, f'{title_prefix.strip()}_training_cv_stats.png'))
    plt.show()



# 主程序
if __name__ == "__main__":
    print("\n=== 诈骗对话检测 - BERT模型训练程序 ===")
    
    # 设置数据路径
    TRAIN_PATH = os.path.join("训练集结果.csv")
    TEST_PATH = os.path.join("测试集结果.csv")
    
    # 设置模型保存路径
    MODEL_PATH = 'bert-model'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    
    # 设置模型文件路径
    model_save_path = os.path.join(MODEL_PATH, 'fraud_detection_model.pt')
    
    # 选择模式
    print("\n请选择模式:")
    print("1. 训练新模型")
    print("2. 仅交互式测试(加载已有模型)")
    print("3. 评估测试集(加载已有模型)")
    
    try:
        mode_choice = input("\n请选择 (1/2/3): ").strip()
    except (EOFError, KeyboardInterrupt):
        mode_choice = '2'  # 默认交互模式
    
    # 评估测试集模式
    if mode_choice == '3':
        if not os.path.exists(model_save_path):
            print(f"\n错误: 模型文件不存在: {model_save_path}")
            print("请先训练模型或确认模型文件路径正确")
            exit(1)
        
        print(f"\n=== 评估测试集模式 ===")
        print(f"模型路径: {model_save_path}")
        print(f"测试集路径: {TEST_PATH}")
        
        # 初始化tokenizer
        if os.path.exists(os.path.join(MODEL_PATH, 'tokenizer')):
            tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_PATH, 'tokenizer'))
        else:
            print("下载tokenizer...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 加载测试数据
        print("\n加载测试数据...")
        test_data = load_dataset(TEST_PATH)
        test_dataset = FraudDetectionDataset(test_data, tokenizer=tokenizer, max_length=256)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 创建模型并加载权重
        print("\n加载模型...")
        model = FraudDetectionModel()
        model.to(device)
        model.load_trained_model(model_save_path)
        print(f"成功加载模型: {model_save_path}")
        
        # 在测试集上进行预测
        print("\n开始评估测试集...")
        test_predictions, test_true_labels = predict(model, test_loader)
        test_metrics = calculate_metrics(test_predictions, test_true_labels)
        
        # 输出详细统计结果
        print("\n" + "="*60)
        print("测试集评估结果")
        print("="*60)
        print(f"总样本数: {test_metrics['total']}")
        print(f"  - 诈骗样本数: {test_metrics['fraud_total']}")
        print(f"  - 非诈骗样本数: {test_metrics['nonfraud_total']}")
        print(f"\n总体准确率: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
        print(f"  - 诈骗样本准确率: {test_metrics['accuracy_fraud']:.4f} ({test_metrics['accuracy_fraud']*100:.2f}%)")
        print(f"  - 非诈骗样本准确率: {test_metrics['accuracy_nonfraud']:.4f} ({test_metrics['accuracy_nonfraud']*100:.2f}%)")
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(test_true_labels, test_predictions)
        print(f"\n混淆矩阵:")
        print(f"                预测非诈骗  预测诈骗")
        print(f"实际非诈骗        {cm[0][0]:6d}    {cm[0][1]:6d}")
        print(f"实际诈骗          {cm[1][0]:6d}    {cm[1][1]:6d}")
        
        # 详细分类报告
        print(f"\n分类报告:")
        print(classification_report(test_true_labels, test_predictions, 
                                   target_names=['非诈骗', '诈骗'], digits=4))
        
        # 保存预测结果
        output_path = os.path.join(MODEL_PATH, 'test_evaluation_results.csv')
        save_predictions(test_predictions, test_true_labels, output_path)
        print(f"\n预测结果已保存到: {output_path}")
        print("="*60)
        
        exit(0)
    
    # 仅交互模式
    if mode_choice == '2':
        if not os.path.exists(model_save_path):
            print(f"\n错误: 模型文件不存在: {model_save_path}")
            print("请先训练模型或确认模型文件路径正确")
            exit(1)
            
        # 初始化tokenizer
        if os.path.exists(os.path.join(MODEL_PATH, 'tokenizer')):
            tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_PATH, 'tokenizer'))
        else:
            print("下载tokenizer...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 创建模型并加载权重
        model = FraudDetectionModel()
        model.to(device)
        model.load_trained_model(model_save_path)
        print(f"\n成功加载模型: {model_save_path}")
        
        # 进入交互式测试
        interactive_test(model, tokenizer, max_length=256)
        exit(0)
    
    # 训练模式参数
    epochs = 5
    batch_size = 40
    learning_rate = 2e-5
    
    print(f"\n数据路径:")
    print(f"  训练集: {TRAIN_PATH}")
    print(f"  测试集: {TEST_PATH}")
    print(f"  模型保存路径: {MODEL_PATH}")
    print(f"\n训练参数:")
    print(f"  训练轮数: {epochs}")
    print(f"  批量大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    
    # 准备数据加载器
    print("\n准备数据...")
    train_loader, val_loader, test_loader, tokenizer = prepare_dataloaders(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        batch_size=batch_size
    )
    
    # 创建模型
    print("\n创建模型...")
    model = FraudDetectionModel()
    model.to(device)
    
    # 设置模型保存路径
    model_save_path = os.path.join(MODEL_PATH, 'fraud_detection_model.pt')
    
    # 交互式测试入口
    try:
        choice = input("\n是否进入交互式测试？(y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = 'n'
    if choice in ['y', 'yes']:
        # 确保加载最新权重
        model.load_trained_model(model_save_path)
        interactive_test(model, tokenizer, max_length=256)

    # 训练模型
    print(f"\n开始训练模型...")
    training_stats = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        model_save_path=model_save_path
    )
    
    # 绘制训练统计信息
    plot_training_stats(training_stats, title_prefix="Fraud Detection")
    
    # 加载最佳模型进行预测
    print("\n加载最佳模型进行预测...")
    model.load_trained_model(model_save_path)
    
    # 在训练集上进行预测和评估
    print("\n在训练集上进行预测和评估...")
    train_predictions, train_true_labels = predict(model, train_loader)
    train_metrics = calculate_metrics(train_predictions, train_true_labels)
    
    # 在测试集上进行预测和评估
    print("\n在测试集上进行预测和评估...")
    test_predictions, test_true_labels = predict(model, test_loader)
    test_metrics = calculate_metrics(test_predictions, test_true_labels)
    
    # 保存预测结果
    print("\n保存预测结果...")
    save_predictions(train_predictions, train_true_labels, os.path.join(MODEL_PATH, 'train_predictions.csv'))
    save_predictions(test_predictions, test_true_labels, os.path.join(MODEL_PATH, 'test_predictions.csv'))
    
    # 输出评估指标
    print("\n=== 训练集评估指标 ===")
    print(f"  总样本数: {train_metrics['total']}")
    print(f"  诈骗样本数: {train_metrics['fraud_total']}")
    print(f"  非诈骗样本数: {train_metrics['nonfraud_total']}")
    print(f"  总体准确率 (accuracy): {train_metrics['accuracy']:.4f}")
    print(f"  诈骗样本准确率 (accuracy_fraud): {train_metrics['accuracy_fraud']:.4f}")
    print(f"  非诈骗样本准确率 (accuracy_nonfraud): {train_metrics['accuracy_nonfraud']:.4f}")
    
    print("\n=== 测试集评估指标 ===")
    print(f"  总样本数: {test_metrics['total']}")
    print(f"  诈骗样本数: {test_metrics['fraud_total']}")
    print(f"  非诈骗样本数: {test_metrics['nonfraud_total']}")
    print(f"  总体准确率 (accuracy): {test_metrics['accuracy']:.4f}")
    print(f"  诈骗样本准确率 (accuracy_fraud): {test_metrics['accuracy_fraud']:.4f}")
    print(f"  非诈骗样本准确率 (accuracy_nonfraud): {test_metrics['accuracy_nonfraud']:.4f}")
    
    print("\n=== 训练完成 ===")
    print("\n模型已保存到: " + model_save_path)
    print("预测结果已保存到: " + MODEL_PATH + " 目录下")
    
