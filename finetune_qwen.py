import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer,SFTConfig
# from peft import LoraConfig, get_peft_model
from peft import (
    get_peft_config,
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    PeftType,
)

def main():
    # 加载模型和分词器
    model_name = "THUDM/glm-4-9b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # 配置 LoRA
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=20,  # 提示令牌数量
        prompt_tuning_init_text="请根据以下内容生成回答：",  # 初始化提示文本
        tokenizer_name_or_path="THUDM/glm-4-9b-chat",
    )

    
    # 应用 LoRA
    model = get_peft_model(model, peft_config)

    # 检测CUDA可用性
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("检测到CUDA设备，已启用GPU加速")
    else:
        print("未检测到CUDA设备，使用CPU训练")

    # 加载数据集
    # 加载完整数据集
    full_dataset = load_dataset('json', data_files='output_all_train.json', split='train')
    
    # 随机切分数据集为训练集(80%)、验证集(10%)和测试集(10%)
    splits = full_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = splits['train']
    valid_dataset = splits["test"]

    # 训练参数
    training_args = SFTConfig(
        output_dir="checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        bf16=torch.cuda.is_available(),  # 自动根据CUDA可用性启用bf16
        logging_steps=100,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=3,
        optim="adamw_torch",
        warmup_ratio=0.1
    )

    # 初始化 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
        # tokenizer=tokenizer,
        # packing=False,
        # max_seq_length=2048,
    )

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()