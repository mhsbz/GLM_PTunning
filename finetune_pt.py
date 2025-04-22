import json
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, PromptTuningConfig, TaskType, PeftModel
import torch

# --- 配置参数 ---
MODEL_NAME = "THUDM/chatglm3-6b"  # 预训练模型名称或路径 (请根据实际情况修改为 ChatGLM4-9B 的正确标识符)
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
    remove_unused_columns=False,
)

MAX_SEQ_LENGTH = 512 # 最大序列长度

# --- 数据加载与处理 ---
def load_and_prepare_data(data_path, tokenizer):
    """加载 JSON 数据并进行处理和 tokenize"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            # 假设 output_all.json 是一个包含字典列表的 JSON 文件
            # 每个字典包含 'instruction', 'input', 'output' 或类似的键
            # 需要根据实际 JSON 结构调整此部分
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON 数据应为列表格式")
            if not data:
                 raise ValueError("JSON 数据不能为空")
            if not all(isinstance(item, dict) for item in data):
                 raise ValueError("JSON 列表中的元素应为字典格式")

            # 示例：假设 JSON 结构为 [{'instruction': '...', 'input': '...', 'output': '...'}, ...]
            # 或者 [{'prompt': '...', 'response': '...'}, ...]
            # 需要根据实际情况调整下面的键名
            prompts = []
            responses = []
            for item in data:
                if 'prompt' in item and 'response' in item:
                    prompts.append(item['prompt'])
                    responses.append(item['response'])
                else:
                    # 如果结构不匹配，可以跳过或抛出错误
                    print(f"警告: 跳过格式不符的数据项: {item}")
                    continue

            if not prompts or not responses:
                raise ValueError("未能从 JSON 文件中提取有效的 prompt 和 response 数据")

            dataset_dict = {"prompt": prompts, "response": responses}
            dataset = Dataset.from_dict(dataset_dict)

    except FileNotFoundError:
        print(f"错误: 数据文件 {data_path} 未找到。")
        return None
    except json.JSONDecodeError:
        print(f"错误: 解析 JSON 文件 {data_path} 失败。请检查文件格式。")
        return None
    except ValueError as e:
        print(f"错误: 处理数据时出错 - {e}")
        return None
    except Exception as e:
        print(f"加载数据时发生未知错误: {e}")
        return None

    def tokenize_function(examples):
        """Tokenize 数据"""
        # 构建输入文本，格式通常为 prompt + response + eos_token
        inputs = [p + "\n" + r + tokenizer.eos_token for p, r in zip(examples["prompt"], examples["response"])]
        # 对输入进行 tokenize
        model_inputs = tokenizer(inputs, max_length=MAX_SEQ_LENGTH, padding="max_length", truncation=True)
        # labels 通常与 input_ids 相同，用于计算损失
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response"])
    return tokenized_dataset

# --- 模型加载与配置 ---
def setup_model(model_name, peft_config):
    """加载预训练模型和 tokenizer，并应用 PEFT 配置"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 确保 tokenizer 有 pad_token，如果没有，通常设置为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # 如果 GPU 可用且显存充足，可以尝试加载半精度模型
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # device_map="auto" # 自动分配设备
    )

    # 应用 Prompt Tuning
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # 打印可训练参数信息
    return tokenizer, model

# --- 训练 --- 
def train_model(model, tokenizer, train_dataset, training_args, peft_model_dir):
    """执行模型训练"""
    # Data collator 用于处理批次数据
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("开始训练...")
    trainer.train()
    print("训练完成！")

    # 保存 PEFT adapter
    print(f"保存 PEFT adapter 到 {peft_model_dir}...")
    model.save_pretrained(peft_model_dir)
    tokenizer.save_pretrained(peft_model_dir) # 同时保存 tokenizer 配置
    print("PEFT adapter 保存完成！")

# --- 主函数 ---
def main():
    print("1. 设置模型...")
    tokenizer, model = setup_model(MODEL_NAME, PROMPT_TUNING_CONFIG)

    print("2. 加载和准备数据...")
    train_dataset = load_and_prepare_data(DATA_PATH, tokenizer)

    if train_dataset is None:
        print("数据加载失败，退出程序。")
        return

    print("3. 开始训练模型...")
    train_model(model, tokenizer, train_dataset, TRAINING_ARGS, PEFT_MODEL_DIR)

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