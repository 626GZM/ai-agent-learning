from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好，用一句话介绍AI Agent"}]
)
print(resp.choices[0].message.content)