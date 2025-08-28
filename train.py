# ======================================================================================
# 1. 导入依赖 (Imports)
# ======================================================================================

# --- 标准库 ---
import os
from string import Template
from math import ceil

# --- 第三方库 ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datasets import Dataset

from accelerate import Accelerator, DistributedDataParallelKwargs
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    InputExample
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# --- 量化相关库 ---
from torchao.quantization import quantize_
from torchao.quantization.qat import FakeQuantizeConfig, IntXQuantizationAwareTrainingConfig


# ======================================================================================
# 2. 全局配置与初始化 (Global Configuration & Initialization)
# ======================================================================================

# 加速器和设备配置
DDP_KWARGS = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[DDP_KWARGS])
device = accelerator.device


# ======================================================================================
# 3. 自定义损失函数 (Custom Loss Functions)
# ======================================================================================

class NormQuantAwareCosineSimilarityLoss(nn.Module):
    """
    一个考虑了量化感知的余弦相似度损失函数。
    它结合了原始嵌入的MSE损失和模拟量化后嵌入的MSE损失。

    Args:
        model (SentenceTransformer): 用于生成嵌入的模型。
        alpha (float): 量化损失的权重。
        quant_levels (int): 模拟量化的级别数。
    """
    def __init__(self, model, alpha=0.2, quant_levels=16):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.quant_levels = quant_levels

    def fake_int4_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """对 [-1, 1] 范围内的张量进行对称的伪量化。"""
        x_clamped = torch.clamp(x, -1, 1)
        scale = 2 / (self.quant_levels - 1)
        q = torch.round((x_clamped + 1) / scale) * scale - 1
        return q

    def forward(self, sentence_features, labels: torch.Tensor):
        # 1. 获取原始 embedding
        reps = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
        emb1, emb2 = reps

        # 2. 归一化
        emb1_norm = F.normalize(emb1, p=2, dim=1).clamp(min=1e-12)
        emb2_norm = F.normalize(emb2, p=2, dim=1).clamp(min=1e-12)

        # 3. 原始 CosSim loss (基于标签)
        cos_sim = F.cosine_similarity(emb1_norm, emb2_norm)
        loss_orig = F.mse_loss(cos_sim, labels)

        # 4. 量化后的 CosSim loss (同样基于标签)
        emb1_q = self.fake_int4_quantize(emb1_norm)
        emb2_q = self.fake_int4_quantize(emb2_norm)
        cos_sim_q = F.cosine_similarity(emb1_q, emb2_q)
        loss_quant = F.mse_loss(cos_sim_q, labels)

        # 5. 组合 loss
        loss = loss_orig + self.alpha * loss_quant
        return loss


class NormQCosineSimilarityLoss(nn.Module):
    """
    一个标准的、基于归一化嵌入的余弦相似度损失函数。
    使用 MSE 计算预测相似度与标签之间的差距。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sentence_features, labels: torch.Tensor):
        reps = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
        emb1_from_reps, emb2_from_reps = reps # 正确的嵌入

        emb1, emb2 = emb1_from_reps, emb2_from_reps # 使用通过模型计算出的嵌入

        # 2. 归一化
        emb1_norm = F.normalize(emb1, p=2, dim=1).clamp(min=1e-12)
        emb2_norm = F.normalize(emb2, p=2, dim=1).clamp(min=1e-12)

        # 3. 计算损失
        cos_sim = F.cosine_similarity(emb1_norm, emb2_norm)
        loss = F.mse_loss(cos_sim, labels)
        return loss


# --- 损失函数获取的辅助函数 ---

def get_norm_loss(model):
    """返回一个归一化的余弦相似度损失函数。"""
    return NormQCosineSimilarityLoss(model)

def get_norm_quantaware_loss(model, alpha=0.2, quant_levels=32):
    """返回一个量化感知的归一化余弦相似度损失函数。"""
    return NormQuantAwareCosineSimilarityLoss(model, alpha, quant_levels)

def get_cosine_similarity_loss(model):
    """
    获取 sentence-transformers 内置的 CosineSimilarityLoss。
    注意：此损失函数期望标签在 [0, 1] 范围内，并在内部将其映射到 [-1, 1]。
    """
    return losses.CosineSimilarityLoss(model=model)


# ======================================================================================
# 4. 数据处理与加载 (Data Processing & Loading)
# ======================================================================================

def load_and_prepare_data(file_path: str = 'processed_dataset_mul_turn.csv'):
    """
    从 CSV 文件加载数据，格式化为模型所需的查询和文档，并划分为训练集和开发集。

    Returns:
        tuple: 包含训练和开发集的 InputExample 列表 (train_examples, dev_examples)。
    """
    df = pd.read_csv(file_path)
    train_df = df.sample(frac=0.95, random_state=42)
    dev_df = df.drop(train_df.index)

    config = {
        "prompts": {
            "query": "Instruct: Find the most suitable action to continue this conversation.\nConversation History\n${history}\nAction:",
            "document": "Action Name\n${action_name}\n\nDescription\n${description}"
        }
    }
    
    query_template = Template(config["prompts"]["query"])
    doc_template = Template(config["prompts"]["document"])

    def get_desc_in_chat(chat_item):
        if chat_item['role'] == 'assistant' and 'tool_calls' in chat_item and chat_item['tool_calls']:
            return f"assistant: Action Name\n{chat_item['tool_calls'][0]['function']['name']}\n"
        elif chat_item['role'] == 'tool':
            return f"tool: {chat_item['content']}\n"
        else:
            return f"{chat_item['role']}: {chat_item['content']}\n"

    def format_query(history_list):
        history_text = "".join(get_desc_in_chat(item) for item in history_list)
        return query_template.substitute(history=history_text)

    def format_document(action_name, description):
        return doc_template.substitute(action_name=action_name, description=description)

    # 兼容 `eval()` 中可能出现的 true/false
    true = True
    false = False

    print("Generating train and dev examples...")
    train_examples = [
        InputExample(
            texts=[
                format_query(eval(row.text1)), 
                format_document(eval(row.text2)['name'], eval(row.text2)['description'])
            ], 
            label=float(row.label)
        ) for row in train_df.itertuples()
    ]
    dev_examples = [
        InputExample(
            texts=[
                format_query(eval(row.text1)),
                format_document(eval(row.text2)['name'], eval(row.text2)['description'])
            ],
            label=float(row.label)
        ) for row in dev_df.itertuples()
    ]
    print(f"Generated {len(train_examples)} training examples and {len(dev_examples)} dev examples.")
    return train_examples, dev_examples


def convert_to_hf_dataset(examples: list[InputExample]) -> Dataset:
    """将 InputExample 列表转换为 Hugging Face Dataset 对象。"""
    dataset_dict = [{"sentence1": ex.texts[0], "sentence2": ex.texts[1], "label": ex.label} for ex in examples]
    return Dataset.from_list(dataset_dict)


# ======================================================================================
# 5. 评估器 (Evaluator)
# ======================================================================================

def get_evaluator(dev_examples: list[InputExample], batch_size: int = 32):
    """
    创建用于评估的 EmbeddingSimilarityEvaluator。
    该评估器计算余弦相似度的 Spearman/Pearson 相关性以及与标签的 MSE。
    """
    return EmbeddingSimilarityEvaluator.from_input_examples(
        dev_examples,
        name="dev-eval",
        batch_size=batch_size,
        show_progress_bar=True,
    )


# ======================================================================================
# 6. 主训练流程 (Main Training Logic)
# ======================================================================================

def main():
    # --- 1. 参数配置 (Hyperparameters & Configs) ---
    BASE_MODEL = "heyanzhuo/Qwen3-Embedding-0.6B-Base-Mod"
    OUTPUT_DIR = "./trained_sent_transformer_v2_qat_mul"
    NUM_EPOCHS = 1
    EFFECTIVE_BATCH_SIZE = 32
    GRAD_ACCUMULATION_STEPS = 32
    PER_DEVICE_TRAIN_BATCH_SIZE = 1 # EFFECTIVE_BATCH_SIZE // GRAD_ACCUMULATION_STEPS
    PER_DEVICE_EVAL_BATCH_SIZE = 2
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    DATALOADER_NUM_WORKERS = 12
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 量化配置
    activation_config = FakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False, scale_precision=torch.float16)
    weight_config = FakeQuantizeConfig(torch.int8, group_size=256, is_symmetric=True, scale_precision=torch.float16)
    qat_config = IntXQuantizationAwareTrainingConfig(activation_config, weight_config)

    # --- 2. 创建模型 (Model Creation) ---
    print("Creating SentenceTransformer model...")
    model = SentenceTransformer(
        BASE_MODEL,
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.to(device)

    # 应用 QAT
    quantize_(model, qat_config)
    print(f"Model built on {device}, embedding dimension = {model.get_sentence_embedding_dimension()}")
    print("Quantization-Aware Training (QAT) has been applied to the model.")

    # --- 3. 数据加载与转换 (Data Loading & Conversion) ---
    train_examples, dev_examples = load_and_prepare_data()
    train_dataset = convert_to_hf_dataset(train_examples)
    print("Converted training data to Hugging Face Dataset format.")
    print("Train dataset features:", train_dataset.features)
    
    # --- 4. 损失函数 (Loss Function) ---
    loss = losses.CosineSimilarityLoss(model)

    # --- 5. 评估器 (Evaluator) ---
    evaluator = get_evaluator(dev_examples, batch_size=PER_DEVICE_EVAL_BATCH_SIZE)

    # --- 6. 训练参数配置 (Training Arguments) ---
    steps_per_epoch = len(train_dataset) // EFFECTIVE_BATCH_SIZE
    warmup_steps = int(WARMUP_RATIO * steps_per_epoch * NUM_EPOCHS)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=50,
        save_total_limit=2,
        max_grad_norm=2.0,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        optim="adamw_8bit", # 使用 8-bit AdamW 优化器
    )
    
    # --- 7. 创建并启动训练器 (Trainer Setup & Execution) ---
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator, # 添加评估器到训练器
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # --- 8. 保存最终模型 (Save Final Model) ---
    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save(final_path)
    print(f"Final model has been saved to {final_path}")


if __name__ == "__main__":
    main()
