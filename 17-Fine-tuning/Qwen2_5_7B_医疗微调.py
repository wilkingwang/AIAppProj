import os
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TextStreamer
from transformers import TrainingArguments
from unsloth import is_bf16_supported

max_seq_length = 2048 # 设置最大序列长度，支持RoPE缩放
dtype = None # 自动检测数据类型
load_in_4bit = True # 启用4位量化加载，减少内存占用

# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen2_5_7B", # 模型名称
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 添加LoRA适配器，只需要更新1~10%的参数
model = FastLanguageModel.get_peft_model(
    model,
    r =16, # LoRA秩
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 需要应用LoRA的模块
    lora_alpha=16, # LoRA缩放因子
    lora_dropout=0, # LoRA dropout率，设置为0可以禁用dropout
    bias="none", # 偏置设置，none为优化设置
    use_gradient_checkpointing="unsloth", # 使用unsloth的梯度检查点，可减少30%显存使用
    random_state=3407, # 随机种子
    use_rslora=False, # 是否使用rank stabilized LoRA，可解决LoRA训练中的梯度消失问题
    loftq_config=None, # LoftQ配置
)

# 数据准备
medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题:
{}

### 回答:
{}
"""

# 获取结束标记
EOS_TOKEN = tokenizer.eos_token

def read_csv_with_encoding(file_path: str):
    """尝试使用不同的编码读取CSV文件"""
    encodings = ["utf-8", "gbk", "gb2312", "gb2312"]
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法使用任何编码读取文件 {file_path}")

def load_medical_data(data_dir: str):
    """加载医疗对话数据"""
    data = []
    departments = {
        'IM_内科': '内科',
        'Surgical_外科': '外科',
        'Pediatric_儿科': '儿科',
        'Oncology_肿瘤科': '肿瘤科',
        'OAGD_妇产科': '妇产科',
        'Andriatria_男科': '男科'
    }

    # 遍历所有科室目录
    for dept_dir, dept_name in departments.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            print(f"警告：目录 {dept_path} 不存在")
            continue

        print(f'正在加载 {dept_name} 数据...')

        # 获取该科室下的所有CSV文件
        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]

        for csv_file in csv_files:
            file_path = os.path.join(dept_path, csv_file)
            try:
                df = read_csv_with_encoding(file_path)

                print(f'文件 {csv_file} 的列明: {df.columns.tolist()}')
                for _, row in df.iterrows():
                    try:
                        question = None
                        answer = None

                        if 'question' in row:
                            question = str(row['question']).strip()
                        elif '问题' in row:
                            question = str(row['问题']).strip()
                        elif 'ask' in row:
                            question = str(row['ask']).strip()

                        if 'answer' in row:
                            answer = str(row['answer']).strip()
                        elif '回答' in row:
                            answer = str(row['回答']).strip()
                        elif 'response' in row:
                            answer = str(row['response']).strip()

                        # 过滤无效数据
                        if not question or not answer:
                            continue

                        # 限制长度
                        if len(question) > 200 or len(answer) > 200:
                            continue

                        # 添加到数据列表
                        data.append({
                            'instruction': '请回答以下医疗相关问题',
                            'input': question,
                            'output': answer,
                        })
                    except Exception as e:
                        print(f'处理数据时出错: {e}')
                        continue
            except Exception as e:
                print(f"加载文件 {csv_file} 时出错: {e}")

    if not data:
        raise ValueError(f"{dept_name} 没有有效数据")

    print(f'成功加载 {dept_name} 数据，共 {len(data)} 条')
    return Dataset.from_dict(data)

def formatting_prompts_func(examples):
    """格式化提示"""
    instructions = examples['instruction']
    input = examples['input']
    output = examples['output']
    texts = []

    for instruction, input, output in zip(instructions, input, output):
        text = medical_prompt.format(instruction, input, output)
        texts.append(text)
    return {"text": texts}

# 加载医疗数据集
dataset = load_medical_data("【数据集】中文医疗数据")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 模型训练

# 设置训练参数和训练器

# 定义训练参数
training_args = TrainingArguments(
    per_device_train_batch_size=2, # 每个设备的训练批次大小
    gradient_accumulation_steps=4, # 梯度累加步数，用于模拟更大的批次大小
    warmup_steps=5, # 预热步数，用于学习率衰减前的训练
    max_steps=-1, # 最大训练步数，-1表示训练到完成
    num_train_epochs=3, # 训练3个epoch
    learning_rate=2e-4, # 学习率
    fp16=not is_bf16_supported(), # 是否使用fp16
    bf16=is_bf16_supported(), # 是否使用bf16
    logging_steps=1, # 日志记录步数
    optim="adamw_torch", # 优化器
    weight_decay=0.01, # 权重衰减
    lr_scheduler_type="linear", # 学习率调度器类型
    seed=3047, # 随机种子，用于复现实验
    save_strategy="epoch",
    report_to="none",
    output_dir="./results", # 输出目录
)

# 创建SFTTrainer实例
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

# 显式当前GPU内存状态
gpu_stats = torch.cuda.get_device_capability(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
max_memory = round(gpu_stats.total_memory / 1024 ** 3, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# 开始训练
trainer_stats = trainer.train()

# 显示训练后的内存和时间统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# 模型推理示例
def generate_medical_response(question):
    """生成医疗答案"""
    FastLanguageModel.for_inference(model)
    inputs = tokenizer([medical_prompt.format(question, "")], return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penelty=1.1)
    
# 测试问题
test_questions = [
    "我最近总是感觉头晕，应该怎么办？",
    "感冒发烧应该吃什么药？",
    "高血压患者需要注意什么？"
]

for question in test_questions:
    print("\n" + "="*50)
    print(f"问题：{question}")
    print("回答：")
    generate_medical_response(question)


# 保存模型
model.save_pretrained("lora_model_medical")
tokenizer.save_pretrained("lora_model_medical")


# 加载保存的模型进行推理
if True:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model", # 训练时使用的模型
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question) 


# 加载保存的模型进行推理
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model_medical",  # 保存的模型
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # 启用原生2倍速推理
    
question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question) 

# 加载保存的模型进行推理
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",  # 基础模型
        adapter_name = "lora_model_medical",  # LoRA权重
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # 启用原生2倍速推理

question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question)

question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question)