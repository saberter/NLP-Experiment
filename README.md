bert-model/：训练好的 fraud_detection_model.pt 权重（通过 Git LFS 管理）
deepseek_multithreaded_results/：多线程 DeepSeek 对抗生成实验输出
2024.findings-acl.292.pdf：参考论文 Evaluating the Validity of Word-level Adversarial Attacks with LLMs
adversarial_attack_framework.py：词重要性、同义词替换、语义相似度检测与 LLM 评估的核心实现
adversarial_experiment.py：基线评估、TextFooler 攻击、消融实验与报告生成入口
deepseek.py：多线程调用 DeepSeek API 生成对抗文本并与 BERT 分类器对比
bert2.py：BERT 欺诈检测模型定义、训练与推理工具
test_textattack.py：最小化 TextAttack 集成示例（加载已有模型，对少量样本执行 TextFooler）
.gitattributes：Git LFS 配置（追踪 PDF 与模型权重等大文件）
