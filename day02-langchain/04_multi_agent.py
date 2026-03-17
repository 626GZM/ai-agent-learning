"""
Multi-Agent 编排
路由Agent判断意图，分发给不同的专职Agent处理
这就是你项目IntelliService的核心架构
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Literal
from dotenv import load_dotenv
import os, json

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ========== 1. 定义三个专职Agent ==========

# 知识问答Agent
def knowledge_agent(state: MessagesState):
    """处理产品咨询类问题"""
    messages = [
        SystemMessage(content="""你是产品知识专家。根据用户问题回答产品相关信息。
回答要专业、简洁。如果不确定，说"建议咨询人工客服"。
已知信息：
- 手机壳支持7天无理由退换，需保持商品完好
- 充电器保修期1年，非人为损坏可免费更换
- 会员每月有3次免运费机会""")
    ] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# 订单Agent（带工具）
@tool
def query_order(order_id: str) -> str:
    """查询订单状态"""
    # 模拟数据库查询（你的项目里会通过MCP调Java微服务）
    orders = {
        "ORD-88892": {"status": "运输中", "eta": "明天下午3点", "courier": "顺丰"},
        "ORD-66543": {"status": "已签收", "time": "昨天上午10点"},
    }
    order = orders.get(order_id, {"status": "未找到该订单"})
    return json.dumps({"order_id": order_id, **order}, ensure_ascii=False)

order_tools = [query_order]
llm_with_order_tools = llm.bind_tools(order_tools)

def order_agent(state: MessagesState):
    """处理订单查询"""
    messages = [
        SystemMessage(content="你是订单助手。帮用户查询订单状态，调用query_order工具。")
    ] + state["messages"]
    response = llm_with_order_tools.invoke(messages)
    return {"messages": [response]}

# 工单Agent（带工具）
@tool
def create_ticket(title: str, description: str, priority: str = "medium") -> str:
    """创建客服工单"""
    return json.dumps({
        "ticket_id": "TK-" + str(hash(title))[-6:],
        "title": title,
        "priority": priority,
        "status": "已创建，客服将在2小时内联系您"
    }, ensure_ascii=False)

ticket_tools = [create_ticket]
llm_with_ticket_tools = llm.bind_tools(ticket_tools)

def ticket_agent(state: MessagesState):
    """处理投诉和建议，创建工单"""
    messages = [
        SystemMessage(content="你是工单助手。用户有投诉或建议时，调用create_ticket创建工单。优先级：一般问题medium，紧急问题high。")
    ] + state["messages"]
    response = llm_with_ticket_tools.invoke(messages)
    return {"messages": [response]}

# ========== 2. 路由Agent —— 整个系统的大脑 ==========

def router(state: MessagesState) -> Literal["knowledge", "order", "ticket"]:
    """判断用户意图，决定交给哪个Agent"""
    messages = [
        SystemMessage(content="""判断用户意图，只回复一个词：
- knowledge：产品咨询、退换货政策、功能介绍等
- order：查订单、物流、快递等
- ticket：投诉、建议、质量问题、要退款等

只回复 knowledge 或 order 或 ticket，不要回复其他内容。"""),
        state["messages"][-1]  # 只看最后一条用户消息
    ]
    response = llm.invoke(messages)
    intent = response.content.strip().lower()

    # 容错处理
    if "order" in intent:
        return "order"
    elif "ticket" in intent:
        return "ticket"
    else:
        return "knowledge"

# ========== 3. 画图：把所有Agent连起来 ==========

graph = StateGraph(MessagesState)

# 添加节点（每个Agent是一个节点）
graph.add_node("knowledge", knowledge_agent)
graph.add_node("order", order_agent)
graph.add_node("order_tools", ToolNode(order_tools))
graph.add_node("ticket", ticket_agent)
graph.add_node("ticket_tools", ToolNode(ticket_tools))

# 起点 → 路由判断 → 分发到不同Agent
graph.add_conditional_edges(START, router)

# 知识Agent → 直接结束（不需要工具）
graph.add_edge("knowledge", END)

# 订单Agent → 可能调工具 → 回来总结 → 结束
graph.add_conditional_edges("order", tools_condition, {
    "tools": "order_tools",  # 需要调工具
    "__end__": END            # 不需要就结束
})
graph.add_edge("order_tools", "order")

# 工单Agent → 可能调工具 → 回来总结 → 结束
graph.add_conditional_edges("ticket", tools_condition, {
    "tools": "ticket_tools",
    "__end__": END
})
graph.add_edge("ticket_tools", "ticket")

# 编译
app = graph.compile()

# ========== 4. 测试所有场景 ==========

print("=" * 60)
print("IntelliService Multi-Agent 客服系统")
print("=" * 60)

test_cases = [
    "你们的退换货政策是什么？",
    "我的订单ORD-88892到哪了？",
    "我买的手机壳有裂痕，要投诉！",
    "充电器保修多久？",
    "帮我查一下订单ORD-66543",
    "你们的服务态度太差了，我要投诉",
]

for question in test_cases:
    print(f"\n用户: {question}")
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    print(f"Agent: {result['messages'][-1].content}")
    print("-" * 40)

print()
print("=" * 60)
print("这就是你项目的核心架构：")
print()
print("  用户提问")
print("    ↓")
print("  路由Agent（判断意图）")
print("    ├→ knowledge_agent → 直接回答")
print("    ├→ order_agent → 调工具查订单 → 回答")
print("    └→ ticket_agent → 调工具建工单 → 回答")
print()
print("后续要做的：")
print("  - knowledge_agent 接入RAG（从真实文档检索）")
print("  - order_agent 通过MCP调你的Java微服务")
print("  - Java后端负责会话存储、调度网关、监控")
print("=" * 60)