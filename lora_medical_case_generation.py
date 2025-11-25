import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# ============= 配置部分 =============
MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
LORA_ADAPTER = "Easonwangzk/lora-llama31-med-adapter"
DATA_PATH = "Rad_filtered_data_final_v8.csv"  # 修改为你的数据路径
OUTPUT_PATH = "lora_comparison_results.csv"
NUM_SAMPLES = 10  # 测试样本数量
MAX_NEW_TOKENS = 256  # 最大生成长度
MAX_SEQ_LEN = 512  # 最大输入长度

# ============= 初始化模型和 Tokenizer =============
print("正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
tokenizer.padding_side = "left"

# 设置 pad token
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# 检测是否支持 bfloat16
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

print(f"正在加载基础模型 {MODEL_REPO}...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    torch_dtype=torch_dtype,
    device_map="auto",
)
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.eval()

print(f"正在加载 LoRA 适配器 {LORA_ADAPTER}...")
lora_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
lora_model.eval()

# ============= 生成函数 =============
@torch.no_grad()
def generate_answer(model, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    使用模型生成答案

    Args:
        model: 语言模型
        prompt: 输入提示词
        max_new_tokens: 最大生成的 token 数量

    Returns:
        生成的答案文本
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # 使用贪婪解码确保可复现
        pad_token_id=tokenizer.pad_token_id,
    )

    gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取 "Answer:" 后面的内容
    return gen.split("Answer:")[-1].strip()

def create_prompt(impression: str) -> str:
    """
    为放射学案例创建提示词

    Args:
        impression: 影像学印象/报告

    Returns:
        格式化的提示词
    """
    messages = [
        {
            "role": "system",
            "content": "You are an expert radiologist. Provide accurate, evidence-based answers using the provided medical context."
        },
        {
            "role": "user",
            "content": f"Context:\n{impression}\n\nQuestion: What are the key findings in this case?\n\nAnswer:"
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ============= 加载数据 =============
print(f"\n正在加载前 {NUM_SAMPLES} 个案例...")
df = pd.read_csv(DATA_PATH, nrows=NUM_SAMPLES)
print(f"成功加载 {len(df)} 个案例\n")

# ============= 基础模型生成 =============
print("=" * 60)
print("使用基础模型生成答案...")
print("=" * 60)
base_answers = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="基础模型"):
    prompt = create_prompt(row['impression'])
    answer = generate_answer(base_model, prompt)
    base_answers.append(answer)

    # 可选：打印每个案例的结果
    if idx < 2:  # 只打印前两个案例作为示例
        print(f"\n--- 案例 {idx + 1} ---")
        print(f"印象: {row['impression'][:100]}...")
        print(f"基础模型回答: {answer[:200]}...\n")

# ============= LoRA 模型生成 =============
print("\n" + "=" * 60)
print("使用 LoRA 微调模型生成答案...")
print("=" * 60)
lora_answers = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="LoRA 模型"):
    prompt = create_prompt(row['impression'])
    answer = generate_answer(lora_model, prompt)
    lora_answers.append(answer)

    # 可选：打印每个案例的结果
    if idx < 2:  # 只打印前两个案例作为示例
        print(f"\n--- 案例 {idx + 1} ---")
        print(f"印象: {row['impression'][:100]}...")
        print(f"LoRA 模型回答: {answer[:200]}...\n")

# ============= 保存结果 =============
results_df = df.copy()
results_df['base_model_answer'] = base_answers
results_df['lora_model_answer'] = lora_answers

# 保存到 CSV
results_df.to_csv(OUTPUT_PATH, index=False)
print("\n" + "=" * 60)
print(f"✅ 结果已保存到: {OUTPUT_PATH}")
print("=" * 60)

# ============= 简单对比展示 =============
print("\n对比示例（前 2 个案例）：\n")
for idx in range(min(2, len(results_df))):
    print(f"{'=' * 60}")
    print(f"案例 {idx + 1}")
    print(f"{'=' * 60}")
    print(f"📋 原始印象:\n{results_df.iloc[idx]['impression']}\n")
    print(f"🤖 基础模型回答:\n{results_df.iloc[idx]['base_model_answer']}\n")
    print(f"🎯 LoRA 模型回答:\n{results_df.iloc[idx]['lora_model_answer']}\n")
