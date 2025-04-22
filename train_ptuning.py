import json
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, TaskType, PeftConfig, PeftModel, PromptTuningConfig, PromptTuningInit
import torch

# --- 配置 --- #
MODEL_NAME = "THUDM/glm-4-9b-chat"  # 确认模型标识符是否准确
DATA_PATH = "/Users/dxj/Desktop/customer_requirment/glm_PTuning/output_all.json"
OUTPUT_DIR = "/Users/dxj/Desktop/customer_requirment/glm_PTuning/ptuning_v2_output"
PEFT_MODEL_ID = "chatglm4-9b-ptuning-v2"

# P-Tuning v2 配置
PEFT_CONFIG = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM, # 任务类型为因果语言模型
    prompt_tuning_init=PromptTuningInit.TEXT, # 初始化方式
    num_virtual_tokens=10, # 虚拟token数量，可以调整
    prompt_tuning_init_text="请根据以下问题回答：", # 用于初始化的文本，可以修改
    tokenizer_name_or_path=MODEL_NAME
)

# 训练参数
TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, # 根据显存调整
    gradient_accumulation_steps=4, # 梯度累积
    num_train_epochs=3, # 训练轮数
    logging_steps=10,
    save_steps=50,
    learning_rate=1e-3, # P-Tuning 通常需要稍高的学习率
    fp16=torch.cuda.is_available(), # 如果有GPU且支持fp16，则启用
    save_total_limit=2,
    logging_dir=f"{OUTPUT_DIR}/logs",
    remove_unused_columns=False, # 保留所有原始列
)

MAX_LENGTH = 1024 # 根据需要调整最大序列长度

# --- 数据加载与预处理 --- #
def load_and_preprocess_data(data_path, tokenizer):
    """加载数据并进行预处理"""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 假设 JSON 结构是 [{'instruction': '...', 'input': '...', 'output': '...'}, ...]
    # 或者 [{'question': '...', 'answer': '...'}, ...]
    # 需要根据实际 JSON 结构调整下面的处理逻辑
    processed_data = []
    for item in raw_data:
        prompt_key = item.get('prompt', '') # 新增：获取 prompt
        response_key = item.get('response', '') # 新增：获取 response

        prompt = prompt_key
        target = response_key
       
        full_text = prompt + " " + target 
        processed_data.append({"text": full_text})

    if not processed_data:
        raise ValueError("No valid data found in the JSON file. Please check the file format and keys ('instruction'/'input'/'output', 'question'/'answer', or 'prompt'/'response').")

    dataset = Dataset.from_list(processed_data)

    def tokenize_function(examples):
        # 对 'text' 列进行分词
        tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        # 创建 labels，将 padding 部分设为 -100
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        # 注意：对于 Causal LM，通常不需要将 prompt 部分的 label 设为 -100，模型需要学习预测整个序列
        # 但如果只想让模型学习预测 target 部分，需要更复杂的处理来 mask 掉 prompt 部分的 loss
        return tokenized_output

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# --- 模型加载 --- #
def load_model(model_name, peft_config):
    """加载基础模型和 Tokenizer，并应用 PEFT 配置"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 设定 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # 如果显存不足，可以尝试加载 8-bit 或 4-bit 量化模型
        # load_in_8bit=True,
        # device_map='auto' # 自动分配设备
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 # 优先使用 bfloat16
    )

    # 应用 P-Tuning v2
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer

# --- 训练 --- #
def train_model(model, tokenizer, train_dataset, training_args):
    """初始化 Trainer 并开始训练"""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        # Data collator 会处理 padding，确保 labels 正确
        # data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), # Causal LM 不需要 MLM
    )
    trainer.train()
    # 保存最终的 PEFT 模型
    final_save_path = os.path.join(training_args.output_dir, PEFT_MODEL_ID)
    trainer.save_model(final_save_path)
    print(f"P-Tuning v2 adapter saved to: {final_save_path}")

# --- 主函数 --- #
if __name__ == "__main__":
    print("Loading model and tokenizer...")
    model, tokenizer = load_model(MODEL_NAME, PEFT_CONFIG)

    print("Loading and preprocessing data...")
    # try:
    tokenized_train_dataset = load_and_preprocess_data(DATA_PATH, tokenizer)
    print(f"Data loaded successfully. Number of samples: {len(tokenized_train_dataset)}")


    print("Starting training...")
    # try:
    train_model(model, tokenizer, tokenized_train_dataset, TRAINING_ARGS)
    print("Training finished successfully.")