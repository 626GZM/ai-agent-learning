"""
Day 2 - LangChain 入门
对比昨天的手写代码，看框架简化了什么
"""
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# ========== 1. 创建模型 ==========
# 昨天：client = OpenAI(api_key=..., base_url=...)
# LangChain：一行搞定，而且可以随时换模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

print("== 1. 基础调用 ==")
# 昨天：client.chat.completions.create(model=..., messages=[...])
# LangChain：直接invoke
resp = llm.invoke("用一句话解释什么是AI Agent")
print(resp.content)
print()

# ========== 2. Prompt模板 —— 对标Spring的@Value模板 ==========
print("== 2. Prompt模板 ==")

# 昨天：手拼messages列表
# LangChain：模板化，可复用
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，回答简洁专业"),
    ("user", "{question}")
])

# 模板 + 模型 + 输出解析 = Chain（链）
# 这个 | 管道符就是LangChain的核心概念，对标Java的责任链
chain = prompt | llm | StrOutputParser()

# 同一个chain，换不同参数就能复用
result = chain.invoke({"role": "Python教练", "question": "装饰器是什么？"})
print(f"Python教练: {result}\n")

result = chain.invoke({"role": "架构师", "question": "微服务的缺点是什么？"})
print(f"架构师: {result}\n")

# ========== 3. JSON结构化输出 ==========
print("== 3. JSON输出 ==")

json_prompt = ChatPromptTemplate.from_messages([
    ("system", """分析用户意图，只输出JSON：
{{"intent": "意图类型", "confidence": 0.0-1.0, "entities": {{}}}}
意图类型：query_order/product_consult/complaint/other"""),
    ("user", "{input}")
])

json_chain = json_prompt | llm | JsonOutputParser()

# 直接得到Python dict，不用手动json.loads
result = json_chain.invoke({"input": "我的订单88892怎么还没到？"})
print(f"意图分析: {result}")
print(f"类型: {type(result)}")  # 直接是dict
print()

# ========== 4. Chain的威力：串联多步 ==========
print("== 4. 多步Chain ==")

# 第一步：分析意图
intent_chain = json_prompt | llm | JsonOutputParser()

# 第二步：根据意图生成回复
reply_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是客服。用户意图是{intent}，相关信息是{entities}。给出友好的回复。"),
    ("user", "{original_input}")
])
reply_chain = reply_prompt | llm | StrOutputParser()

# 手动串联（后面LangGraph会让这个更优雅）
user_input = "我买的手机壳有质量问题，要退货"
intent_result = intent_chain.invoke({"input": user_input})
print(f"第一步 - 意图: {intent_result}")

final_reply = reply_chain.invoke({
    "intent": intent_result.get("intent", "other"),
    "entities": str(intent_result.get("entities", {})),
    "original_input": user_input
})
print(f"第二步 - 回复: {final_reply}")
print()

print("=" * 50)
print("对比昨天的手写代码，LangChain简化了什么：")
print("1. 不用手拼messages列表 → 用Prompt模板")
print("2. 不用手动json.loads → JsonOutputParser自动解析")
print("3. | 管道符串联多步处理 → 对标Java责任链")
print("4. 换模型只改一行 → 不用改业务代码")
print("=" * 50)