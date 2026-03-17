"""
LangGraph 入门
把Agent的工作流程变成一张状态图
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os, json

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ========== 1. 最简单的LangGraph：理解概念 ==========
print("== 1. 最简单的图：一个节点 ==\n")

# StateGraph需要一个State类型，MessagesState是内置的，里面就是messages列表
# 对标Java：相当于一个Context对象在各节点间传递

def chatbot(state: MessagesState):
    """一个节点就是一个函数，接收state，返回更新"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 构建图（对标Java：定义状态机的状态和转换）
graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)       # 添加节点
graph.add_edge(START, "chatbot")         # START → chatbot
graph.add_edge("chatbot", END)           # chatbot → END

# 编译（对标Java：build()）
app = graph.compile()

# 运行
result = app.invoke({"messages": [HumanMessage(content="用一句话解释LangGraph")]})
print(result["messages"][-1].content)
print()

# ========== 2. 加上工具调用的图 ==========
print("== 2. 带工具的Agent图 ==\n")

@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态"""
    return json.dumps({"order_id": order_id, "status": "运输中", "eta": "明天下午"}, ensure_ascii=False)

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库"""
    return json.dumps({"answer": f"关于'{query}'：支持7天无理由退换货"}, ensure_ascii=False)

@tool
def create_ticket(title: str, description: str) -> str:
    """创建客服工单"""
    return json.dumps({"ticket_id": "TK-001", "status": "已创建"}, ensure_ascii=False)

@tool
def check_weather(city: str) -> str:
    """查询天气"""
    return json.dumps({"city": city, "weather": "阴"}, ensure_ascii=False)

tools = [query_order, search_knowledge, create_ticket, check_weather]
llm_with_tools = llm.bind_tools(tools)

def agent(state: MessagesState):
    """Agent节点：LLM决定要不要调工具"""
    # 加上系统提示
    messages = [SystemMessage(content="你是智能客服，根据用户需求调用合适的工具")] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 构建带工具的图
#
# 流程：
#   START → agent → (有工具调用?) → tools → agent → ... → END
#                    (没有?)      → END
#
# 对标你昨天手写的循环，但现在是声明式的图

graph2 = StateGraph(MessagesState)
graph2.add_node("agent", agent)
graph2.add_node("tools", ToolNode(tools))  # LangGraph内置的工具执行节点

graph2.add_edge(START, "agent")
graph2.add_conditional_edges(
    "agent",
    tools_condition,  # 内置判断：有tool_calls就去tools节点，没有就END
)
graph2.add_edge("tools", "agent")  # 工具执行完，回到agent让LLM总结

app2 = graph2.compile()

# 测试
print("--- 查订单 ---")
result = app2.invoke({"messages": [HumanMessage(content="我的订单ORD-88892到哪了？")]})
print(f"Agent: {result['messages'][-1].content}\n")

print("--- 查知识库 ---")
result = app2.invoke({"messages": [HumanMessage(content="退换货政策是什么？")]})
print(f"Agent: {result['messages'][-1].content}\n")

print("--- 创建工单 ---")
result = app2.invoke({"messages": [HumanMessage(content="手机壳有质量问题，我要投诉")]})
print(f"Agent: {result['messages'][-1].content}\n")

print("--- 查询天气 ---")
result = app2.invoke({"messages": [HumanMessage(content="北京天气怎么样？")]})
print(f"Agent: {result['messages'][-1].content}\n")

print("--- 普通闲聊（不调工具） ---")
result = app2.invoke({"messages": [HumanMessage(content="你好呀")]})
print(f"Agent: {result['messages'][-1].content}\n")

# ========== 3. 看看图的结构 ==========
print("== 3. 图的结构 ==\n")
print("节点:", [n for n in app2.get_graph().nodes])
print()
print("这就是你项目的Agent核心架构：")
print("  START → agent(LLM决策) → tools(执行工具) → agent(总结) → END")
print("  如果不需要工具：START → agent → END")
print()

# ========== 对比总结 ==========
print("=" * 50)
print("三天的进化：")
print()
print("Day 1 手写：30行循环代码，if/else判断调哪个工具")
print("Day 2 LangChain：@tool简化定义，但调用循环还是手写")
print("Day 2 LangGraph：声明式画图，框架自动处理循环")
print()
print("LangGraph的核心优势：")
print("1. 流程可视化 — 画出来就是一张图")
print("2. 状态管理 — state在节点间自动传递")
print("3. 条件路由 — conditional_edges决定走哪条路")
print("4. 可扩展 — 加新功能就是加节点和边")
print("=" * 50)