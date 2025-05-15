import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer,SFTConfig
from peft import get_peft_model, PrefixTuningConfig, TaskType

def main():
    # 加载模型和分词器
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # 配置 LoRA
    config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10, prefix_projection=False)
    
    # 应用 LoRA
    model = get_peft_model(model, config)

    # 检测CUDA可用性
    if torch.cuda.is_available():
        model = model.cuda()
        print("检测到CUDA设备，已启用GPU加速")
    else:
        print("未检测到CUDA设备，使用CPU训练")

    model.print_trainable_parameters()

    # 加载数据集
    # 加载完整数据集
    full_dataset = load_dataset('json', data_files='sft_train.json', split='train')
    
    # 随机切分数据集为训练集(80%)、验证集(10%)和测试集(10%)
    splits = full_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_dataset = splits['train']
    valid_dataset = splits['test']

    # 训练参数
    training_args = SFTConfig(
        output_dir="checkpoints",
        num_train_epochs=1,
        max_grad_norm=0.3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        per_gpu_eval_batch_size=4,
        eval_accumulation_steps=4,
        learning_rate=6e-5,
        bf16=torch.cuda.is_available(),  # 自动根据CUDA可用性启用bf16
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        max_steps=300,
        save_total_limit=10,
        optim="adamw_torch",
        warmup_ratio=0.1,
        ddp_find_unused_parameters=False,
        report_to="wandb",
    )

    # 初始化 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
    )

    # 开始训练
    trainer.train()

### CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 train_llama_lora.py
if __name__ == "__main__":
    main()