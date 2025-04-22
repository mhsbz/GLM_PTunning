from datasets import load_dataset
from peft import (
    get_peft_config,
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    PeftType,
)
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM

dataset = load_dataset("json", data_files="output_all.json")
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat",trust_remote_code=True)

def preprocess_function(examples):
    texts = [q + "答：" + r for q, r in zip(
        examples["prompt"], 
        examples["response"]
    )]
    return tokenizer(texts, truncation=True, max_length=512)
    
processed_dataset = dataset.map(preprocess_function, batched=True)

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=20,  # 提示令牌数量
    prompt_tuning_init_text="请根据以下内容生成回答：",  # 初始化提示文本
    tokenizer_name_or_path="THUDM/glm-4-9b-chat",
)

model = AutoModelForCausalLM.from_pretrained("THUDM/glm-4-9b-chat",trust_remote_code=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 应显示仅提示参数可训练

training_args = TrainingArguments(
    output_dir="./chatglm-ptuning",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=5,
    fp16=True,  # 启用混合精度训练
    logging_steps=50,
    # save_strategy="epoch",
    # evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    # data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.model.save_pretrained("./chatglm-ptuning-final")