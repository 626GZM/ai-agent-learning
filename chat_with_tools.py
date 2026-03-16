"""
Prompt Engineering实战 + Function Calling
这两个是Agent的核心：提示词控制输出 + LLM调用工具
"""
from openai import OpenAI
from dotenv import load_dotenv
import os, json

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ========== 1. Few-shot: 自然语言生成SQL ==========
print("== 1. NL2SQL（结合你的后端经验）==")

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": """你是SQL专家。用户用自然语言描述需求，你生成MySQL查询语句。
只输出JSON格式：{"sql": "SQL语句", "explanation": "简要说明"}

示例：
用户：查找所有VIP用户
输出：{"sql": "SELECT * FROM users WHERE vip_level > 0", "explanation": "筛选vip_level大于0的用户"}

用户：统计每个城市的订单数
输出：{"sql": "SELECT city, COUNT(*) as order_count FROM orders GROUP BY city ORDER BY order_count DESC", "explanation": "按城市分组统计订单数并降序排列"}"""},
        {"role": "user", "content": "查找最近30天下单超过3次的用户，显示用户名和订单数"}
    ],
    response_format={"type": "json_object"}
)
result = json.loads(resp.choices[0].message.content)
print(f"SQL: {result['sql']}")
print(f"说明: {result['explanation']}")
print()

# ========== 2. Chain-of-Thought: 让LLM一步步推理 ==========
print("== 2. CoT推理 ==")

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "解决问题时，先一步步分析，最后给出结论。"},
        {"role": "user", "content": "我们的客服系统需要处理三种请求：产品咨询、订单查询、投诉建议。请设计Agent的路由策略。"}
    ]
)
print(resp.choices[0].message.content)
print()

# ========== 3. Function Calling —— Agent的核心 ==========
print("== 3. Function Calling ==")
print("这是Agent的本质：LLM不执行代码，只决定调用哪个工具\n")

# 定义工具（对标Java的接口定义）
tools = [
    # check_weather
{
        "type": "function",
        "function": {
            "name": "check_weather",
            "description": "根据城市查询天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市"}
                },
                "required": ["city"]
            }
        }
    },
    #check_balance
{
        "type": "function",
        "function": {
            "name": "check_balance",
            "description": "根据账户号查询余额",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "账户号"}
                },
                "required": ["account_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "根据订单号查询订单状态和物流信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "从知识库中搜索产品相关信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "创建客服工单",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "工单标题"},
                    "description": {"type": "string", "description": "问题描述"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                },
                "required": ["title", "description"]
            }
        }
    }
]


# 模拟工具执行（对标Java的Service层）
def execute_tool(name: str, args: dict) -> str:
    if name == "query_order":
        return json.dumps({"order_id": args["order_id"], "status": "运输中", "eta": "明天下午"})
    elif name == "search_knowledge":
        return json.dumps({"results": [f"关于'{args['query']}'的产品说明：支持7天无理由退换..."]})
    elif name == "create_ticket":
        return json.dumps({"ticket_id": "TK-2024001", "status": "已创建"})
    elif name == "check_balance":
        return json.dumps({"account_id": args["account_id"], "余额":"10000"})
    elif name == "check_weather":
        return json.dumps({"city": args["city"], "天气" : "多云转晴" })
    return json.dumps({"error": "未知工具"})


# 完整的工具调用循环
def chat_with_tools(user_message: str):
    print(f"用户: {user_message}")
    messages = [
        {"role": "system", "content": "你是智能客服，根据用户需求调用合适的工具来回答问题。"},
        {"role": "user", "content": user_message}
    ]

    # 第一轮：LLM决定调用哪个工具
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )
    msg = resp.choices[0].message

    # 如果LLM决定调用工具
    if msg.tool_calls:
        messages.append(msg)  # 把LLM的决策加入对话历史

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"  → LLM决定调用: {func_name}({func_args})")

            # 执行工具
            result = execute_tool(func_name, func_args)
            print(f"  ← 工具返回: {result}")

            # 把工具结果加入对话历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # 第二轮：LLM根据工具结果生成最终回答
        final_resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        print(f"Agent: {final_resp.choices[0].message.content}")
    else:
        print(f"Agent: {msg.content}")
    print()


# 测试三种场景
# chat_with_tools("我的订单 ORD-88892 到哪了？")
# chat_with_tools("你们的退换货政策是什么？")
# chat_with_tools("我买的手机屏幕碎了，要投诉！")
# chat_with_tools("我的账号 990298还有多少钱？")
chat_with_tools("天津今天天气怎么样")

print("=" * 50)
print("核心认知：")
print("1. LLM不执行代码，只决定调用哪个工具、传什么参数")
print("2. 你的代码负责真正执行工具，然后把结果喂回给LLM")
print("3. 这个'LLM决策→工具执行→LLM总结'的循环就是Agent的本质")
print("=" * 50)