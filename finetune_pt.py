import json
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments # Removed Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, PromptTuningConfig, TaskType, PeftModel
import torch
from trl import SFTTrainer # Added SFTTrainer

# --- 配置参数 ---
MODEL_NAME = "THUDM/glm-4-9b-chat"  # 预训练模型名称或路径 (请根据实际情况修改为 ChatGLM4-9B 的正确标识符)
DATA_PATH = "output_all.json"  # 训练数据路径
OUTPUT_DIR = "./chatglm4-pt-output"  # 微调后模型输出目录
PEFT_MODEL_DIR = "./chatglm4-pt-adapter" # PEFT adapter 输出目录

# Prompt Tuning 配置
PROMPT_TUNING_CONFIG = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM, # 任务类型为因果语言模型
    num_virtual_tokens=10, # 虚拟 token 的数量
    # prompt_tuning_init=PromptTuningInit.TEXT, # 初始化方式，可选
    # prompt_tuning_init_text="Your init text here", # 初始化文本，如果选择 TEXT 初始化
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
    learning_rate=1e-3,
    fp16=torch.cuda.is_available(), # 如果有 GPU 且支持 fp16，则开启
    remove_unused_columns=False, # SFTTrainer 可能需要保留原始列
)

MAX_SEQ_LENGTH = 512 # 最大序列长度

# --- 数据加载与处理 ---
def load_and_prepare_data(data_path):
    """加载 JSON 数据并返回 Dataset 对象"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        prompts = []
        responses = []
        for item in data:
            prompts.append(item['prompt'])
            responses.append(item['response'])

        dataset_dict = {"prompt": prompts, "response": responses}
        dataset = Dataset.from_dict(dataset_dict)
    return dataset

def format_example(example, tokenizer):
    """格式化单个样本用于 SFTTrainer"""
    # 构建输入文本，格式通常为 prompt + response + eos_token
    # SFTTrainer 会自动处理 tokenization 和 label 创建
    return f"{example['prompt']}\n{example['response']}{tokenizer.eos_token}"

# --- 模型加载与配置 ---
def setup_model(model_name, peft_config):
    """加载预训练模型和 tokenizer，并应用 PEFT 配置"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 确保 tokenizer 有 pad_token，如果没有，通常设置为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # SFTTrainer 可能需要 padding_side='right' for GPT-like models
        tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # 如果 GPU 可用且显存充足，可以尝试加载半精度模型
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # device_map="auto" # 自动分配设备, SFTTrainer 可能需要手动处理
    )

    # 应用 Prompt Tuning (SFTTrainer 也可以在初始化时接收 peft_config)
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters() # SFTTrainer 会处理这个
    return tokenizer, model

# --- 训练 ---
def train_model(model, tokenizer, train_dataset, training_args, peft_config, peft_model_dir):
    """使用 SFTTrainer 执行模型训练"""

    # 定义格式化函数，用于 SFTTrainer
    def formatting_prompts_func(example):
        # SFTTrainer 需要一个包含格式化后文本的列表
        output_texts = []
        for i in range(len(example['prompt'])):
            text = f"{example['prompt'][i]}\n{example['response'][i]}{tokenizer.eos_token}"
            output_texts.append(text)
        return output_texts

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        # dataset_text_field="text", # 如果数据集中有 'text' 列
        formatting_func=formatting_prompts_func, # 使用格式化函数
        peft_config=peft_config, # 将 PEFT 配置传递给 SFTTrainer
        max_seq_length=MAX_SEQ_LENGTH,
        # data_collator=None, # SFTTrainer 会处理 data collation
    )

    # 打印可训练参数 (SFTTrainer 应用 PEFT 后)
    model.print_trainable_parameters()

    print("开始使用 SFTTrainer 进行训练...")
    trainer.train()
    print("训练完成！")

    # 保存 PEFT adapter (SFTTrainer 会自动处理)
    # trainer.save_model(peft_model_dir) # SFTTrainer 的保存方法
    # 使用 PEFT 的标准保存方法，确保与加载逻辑一致
    print(f"保存 PEFT adapter 到 {peft_model_dir}...")
    model.save_pretrained(peft_model_dir)
    tokenizer.save_pretrained(peft_model_dir) # 同时保存 tokenizer 配置
    print("PEFT adapter 保存完成！")

# --- 主函数 ---
def main():
    print("1. 设置模型...")
    # 注意：这里先不应用 PEFT，交给 SFTTrainer 处理
    tokenizer, model = setup_model(MODEL_NAME, None) # 传递 None 或移除 peft_config

    print("2. 加载和准备数据...")
    # 加载原始数据，不进行 tokenize
    train_dataset = load_and_prepare_data(DATA_PATH)

    if train_dataset is None:
        print("数据加载失败，退出程序。")
        return

    print("3. 开始训练模型...")
    # 将 PROMPT_TUNING_CONFIG 传递给 train_model
    train_model(model, tokenizer, train_dataset, TRAINING_ARGS, PROMPT_TUNING_CONFIG, PEFT_MODEL_DIR)

    print("\n微调流程结束。PEFT adapter 已保存至:", PEFT_MODEL_DIR)
    print(f"训练日志和检查点保存在: {OUTPUT_DIR}")
    print("\n后续可以使用加载基础模型和 adapter 的方式进行推理:")
    print(f"  from peft import PeftModel")
    print(f"  base_model = AutoModelForCausalLM.from_pretrained('{MODEL_NAME}', trust_remote_code=True)")
    print(f"  peft_model = PeftModel.from_pretrained(base_model, '{PEFT_MODEL_DIR}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{PEFT_MODEL_DIR}', trust_remote_code=True)")
    print("  # 然后使用 peft_model 和 tokenizer 进行推理")

if __name__ == "__main__":
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(PEFT_MODEL_DIR):
        os.makedirs(PEFT_MODEL_DIR)

    main()