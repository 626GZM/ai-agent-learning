import json
import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态和物流信息"""
    # 模拟查数据库（你的项目里这里会调Java微服务）
    return json.dumps({
        "order_id": order_id,
        "status": "运输中",
        "eta": "明天下午3点",
        "courier": "顺丰快递"
    }, ensure_ascii=False)

@tool
def search_knowledge(query: str) -> str:
    """从企业知识库中搜索产品相关信息"""
    return json.dumps({
        "results": [f"关于'{query}'：支持7天无理由退换，需保持商品完好。"]
    }, ensure_ascii=False)

@tool
def create_ticket(title: str, description: str, priority: str = "medium") -> str:
    """创建客服工单，priority可选low/medium/high"""
    return json.dumps({
        "ticket_id": "TK-2024001",
        "title": title,
        "status": "已创建",
        "priority": priority
    }, ensure_ascii=False)

@tool
def check_weather(city: str) -> str:
    """查询指定城市的天气"""
    return json.dumps({
        "city": city,
        "weather": "晴",
        "temp": "28°C"
    }, ensure_ascii=False)

tools = [query_order, search_knowledge, create_ticket, check_weather]
llm_with_tools = llm.bind_tools(tools)

print("== 工具绑定后，LLM的决策过程 ==\n")

resp = llm_with_tools.invoke("我的订单ORD-88892到哪了？")
print(f"LLM返回类型: {type(resp)}")
print(f"文本内容: {resp.content}")
print(f"工具调用: {resp.tool_calls}")
print()

print("== 完整Agent对话 ==\n")

def agent_chat(user_input: str):
    print(f"用户: {user_input}")
    messages = [HumanMessage(user_input)]
    resp = llm_with_tools.invoke(messages)
    if resp.tool_calls:
        messages.append(resp)
        for tc in resp.tool_calls:
            print(f"  → 调用工具: {tc['name']}({tc['args']})")

            # LangChain的tool可以直接用名字找到并执行
            tool_map = {t.name: t for t in tools}
            result = tool_map[tc["name"]].invoke(tc["args"])
            print(f"  ← 返回: {result}")

            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            final = llm_with_tools.invoke(messages)
            print(f"Agent: {final.content}")
    else:
        print(f"Agent: {resp.content}")
        print()
agent_chat("我的订单ORD-88892到哪了？")
agent_chat("你们的退换货政策是什么？")
agent_chat("手机壳质量有问题，我要投诉！")
agent_chat("上海今天天气怎么样？")
