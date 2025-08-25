import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== 配置 ==========
SYSTEM_PROMPT = """
按照如下格式生成：
<think>
...
</think>
<answer>
...
</answer>
"""

MODEL_PATH = "/root/LLM/output/checkpoint-100"


def main():
    # 加载模型和分词器
    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model = model.cuda()
    model.eval()
    print("Model loaded. You can start chatting! (输入 exit 退出)\n")

    # 进入交互循环
    while True:
        user_input = input("你: ")
        if user_input.strip().lower() in ["exit", "quit", "q"]:
            print("退出对话。")
            break
        # 构建对话格式
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
        # 拼接成模型输入
        prompt = ""
        for m in messages:
            prompt += f"{m['role'].upper()}: {m['content']}\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("模型:", response, "\n")


if __name__ == "__main__":
    main()