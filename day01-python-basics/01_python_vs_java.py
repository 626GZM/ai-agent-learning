"""
Day 1 下午 - DeepSeek API 入门
学会三种调用方式：基础对话、多轮对话、流式输出
"""
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ========== 1. 基础单轮对话 ==========
print("== 1. 基础对话 ==")
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个Java转AI的技术顾问"},
        {"role": "user", "content": "用一句话解释什么是AI Agent"}
    ]
)
print(resp.choices[0].message.content)
print(f"Token消耗: 输入{resp.usage.prompt_tokens} + 输出{resp.usage.completion_tokens} = {resp.usage.total_tokens}")
print()

# ========== 2. 多轮对话 ==========
print("== 2. 多轮对话 ==")
messages = [
    {"role": "system", "content": "你是一个Python教练，回答简洁"}
]

questions = ["Python的装饰器是什么？", "给我一个装饰器的例子", "这和Java的注解有什么区别？"]

for q in questions:
    messages.append({"role": "user", "content": q})
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    answer = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    print(f"Q: {q}")
    print(f"A: {answer}\n")

# ========== 3. 流式输出（Streaming） ==========
print("== 3. 流式输出 ==")
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "用3个要点解释RAG技术"}
    ],
    stream=True  # 开启流式
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print("\n")

# ========== 4. JSON模式输出 ==========
print("== 4. JSON结构化输出 ==")
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你只输出JSON格式，不要输出其他内容"},
        {"role": "user", "content": '分析这个用户意图："我的订单123什么时候到？"\n输出格式：{"intent": "意图", "entities": {"字段": "值"}}'}
    ],
    response_format={"type": "json_object"}
)
print(resp.choices[0].message.content)