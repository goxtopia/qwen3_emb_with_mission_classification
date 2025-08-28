import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import losses
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from math import ceil
from accelerate import Accelerator, DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
device = accelerator.device

class NormQuantAwareCosineSimilarityLoss(nn.Module):
    def __init__(self, model, alpha=0.2, quant_levels=16):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.quant_levels = quant_levels

    def fake_int4_quantize(self, x):
        # 简单的对称假量化 [-1, 1]
        x_clamped = torch.clamp(x, -1, 1)
        scale = 2 / (self.quant_levels - 1)
        q = torch.round((x_clamped + 1) / scale) * scale - 1
        return q

    def forward(self, sentence_features, labels):
        # 1. 获取原始 embedding
        reps = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
        emb1, emb2 = reps

        # 2. 归一化
        emb1_norm = F.normalize(emb1, p=2, dim=1).clamp(min=1e-12)
        emb2_norm = F.normalize(emb2, p=2, dim=1).clamp(min=1e-12)

        # 3. 原始 CosSim loss（用标签）
        cos_sim = F.cosine_similarity(emb1_norm, emb2_norm)
        loss_orig = F.mse_loss(cos_sim, labels)

        # 4. 量化后的 CosSim loss（同样用标签）
        emb1_q = self.fake_int4_quantize(emb1_norm)
        emb2_q = self.fake_int4_quantize(emb2_norm)
        cos_sim_q = F.cosine_similarity(emb1_q, emb2_q)
        loss_quant = F.mse_loss(cos_sim_q, labels)

        # 5. 总 loss
        loss = loss_orig + self.alpha * loss_quant
        return loss

class NormQCosineSimilarityLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sentence_features, labels):
        # 1. 获取原始 embedding
        reps = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
        emb1, emb2 = sentence_features[0], sentence_features[1]

        # 2. 归一化
        emb1_norm = F.normalize(emb1, p=2, dim=1).clamp(min=1e-12)
        emb2_norm = F.normalize(emb2, p=2, dim=1).clamp(min=1e-12)

        # 3. 原始 CosSim loss（用标签）
        cos_sim = F.cosine_similarity(emb1_norm, emb2_norm)
        loss_orig = F.mse_loss(cos_sim, labels)

        # 4. 总 loss
        loss = loss_orig
        return loss

def get_norm_loss(model):
    """
    返回一个归一化的余弦相似度损失函数。
    """
    return NormQCosineSimilarityLoss(model)

def get_norm_quantaware_loss(model, alpha=0.2, quant_levels=32):
    return NormQuantAwareCosineSimilarityLoss(model, alpha, quant_levels)

def get_cosine_similarity_loss(model):
    """
    CosineSimilarityLoss 要求标签在 [0, 1]，在训练时会把标签乘以 2 再减 1 变为 [-1, 1]。
    """
    return losses.CosineSimilarityLoss(model=model)


from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

def get_evaluator(dev_examples, batch_size: int = 32):
    """
    `EmbeddingSimilarityEvaluator` 会计算余弦相似度的 Spearman / Pearson
    以及对比标签的 MSE。
    """
    return EmbeddingSimilarityEvaluator.from_input_examples(
        dev_examples,
        name="dev-eval",
        batch_size=batch_size,
        show_progress_bar=True,
    )

import pandas as pd
from string import Template
from datasets import Dataset # 1. Import the Dataset class
import os
# 2. Import the correct loss function
from sentence_transformers import losses, InputExample
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments

# --- Quantization Imports ---
from torchao.quantization import quantize_
from torchao.quantization.qat import FakeQuantizeConfig, IntXQuantizationAwareTrainingConfig

# --- Your Data Loading and Formatting Logic (Moved here for a single file example) ---
# This part is your code, unchanged.
def load_and_prepare_data():
    df = pd.read_csv('processed_dataset_mul_turn.csv')
    train_df = df.sample(frac=0.95, random_state=42)
    dev_df = df.drop(train_df.index)

    config = {
        "prompts": {
            "query": "Instruct: Find the most suitable action to continue this conversation.\nConversation History\n${history}\nAction:",
            "document": "Action Name\n${action_name}\n\nDescription\n${description}"
        }
    }

    def get_desc_in_chat(chat_item):
        if chat_item['role'] == 'assistant' and 'tool_calls' in chat_item and chat_item['tool_calls']:
            return f"{chat_item['role']}: Action Name\n{chat_item['tool_calls'][0]['function']['name']}\n"
        elif chat_item['role'] == 'tool':
            return f"{chat_item['role']}: {chat_item['content']}\n"
        else:
            return f"{chat_item['role']}: {chat_item['content']}\n"

    def format_query(history_list):
        history_text = "".join(get_desc_in_chat(item) for item in history_list)
        return Template(config["prompts"]["query"]).substitute(history=history_text)

    def format_document(action_name, description):
        return Template(config["prompts"]["document"]).substitute(action_name=action_name, description=description)

    # These are needed for the `eval()` calls
    true = True
    false = False

    print("Generating train and dev examples...")
    train_examples = [InputExample(texts=[format_query(eval(row.text1)), format_document(eval(row.text2)['name'], eval(row.text2)['description'])], label=float(row.label)) for row in train_df.itertuples()]
    dev_examples = [InputExample(texts=[format_query(eval(row.text1)), format_document(eval(row.text2)['name'], eval(row.text2)['description'])], label=float(row.label)) for row in dev_df.itertuples()]
    print(f"Generated {len(train_examples)} training examples and {len(dev_examples)} dev examples.")
    return train_examples, dev_examples

# --- Quantization Config (no change) ---
activation_config = FakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False, scale_precision=torch.float16)
weight_config = FakeQuantizeConfig(torch.int8, group_size=256, is_symmetric=True, scale_precision=torch.float16)


def main():
    # --------------------------------------------------
    # 1️⃣ 参数配置
    # --------------------------------------------------
    base_model = "heyanzhuo/Qwen3-Embedding-0.6B-Base-Mod"
    use_flash = True
    output_dir = "./trained_sent_transformer_v2_qat_mul"
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # 2️⃣ 创建模型
    # --------------------------------------------------
    model = SentenceTransformer(
        "heyanzhuo/Qwen3-Embedding-0.6B-Base-Mod",
        model_kwargs={}, # , "torch_dtype": "bfloat16" "attn_implementation": "flash_attention_2",  "device_map": "auto"
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.to(device)

    model[0].auto_model.gradient_checkpointing_enable() 
    
    quantize_(model, IntXQuantizationAwareTrainingConfig(activation_config, weight_config))
    print(f"Model built on {device}, embedding dimension = {model.get_sentence_embedding_dimension()}")

    # --------------------------------------------------
    # 3️⃣ 数据加载和转换
    # --------------------------------------------------
    train_examples, dev_examples = load_and_prepare_data()

    # 3. Convert your list of InputExamples into a Hugging Face Dataset
    train_dataset_dict = [{"sentence1": ex.texts[0], "sentence2": ex.texts[1], "label": ex.label} for ex in train_examples]
    train_dataset = Dataset.from_list(train_dataset_dict)
    print("Converted training data to Hugging Face Dataset format.")
    print("Train dataset features:", train_dataset.features)


    # --------------------------------------------------
    # 4️⃣ 损失函数
    # --------------------------------------------------
    # 4. Use the correct loss for your paired data!
    loss = losses.CosineSimilarityLoss(model)

    # --------------------------------------------------
    # 5️⃣ 评估器（每个 epoch 结束后评估一次）
    # --------------------------------------------------
    # Your evaluator setup is likely already perfect for this, as it probably
    # uses EmbeddingSimilarityEvaluator which works with paired data.
    evaluator = get_evaluator(dev_examples, batch_size=2)

    # --------------------------------------------------
    # 6️⃣ 训练
    # --------------------------------------------------
    num_epochs = 1
    # We need to calculate warmup_steps based on the dataset size and batch size
    # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
    effective_batch_size = 32
    steps_per_epoch = len(train_dataset) // effective_batch_size
    warmup_steps = int(0.1 * steps_per_epoch * num_epochs)

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset, # 5. Use the new Dataset object
        loss=loss,
        args=SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32, # Effective batch size = 4 * 8 = 32
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            learning_rate=2e-5,
            save_strategy="steps",
            save_steps=1000, # A more reasonable save step
            logging_steps=50,
            save_total_limit=2,
            # Use bf16 if supported for better performance with Flash Attention 2
            bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
            fp16=True if torch.cuda.is_available() and not torch.cuda.is_bf16_supported() else False,
            dataloader_num_workers=12,
            max_grad_norm=2.0, # Add gradient clipping to prevent explosion
        )
    )

    trainer.train()

    # --------------------------------------------------
    # 7️⃣ 训练结束后再次保存模型
    # --------------------------------------------------
    final_path = os.path.join(output_dir, "final")
    model.save(final_path)
    print(f"模型已保存到 {final_path}")

    # ... (Your testing code remains the same) ...

if __name__ == "__main__":
    main()
