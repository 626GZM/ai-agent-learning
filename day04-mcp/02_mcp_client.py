"""
Python Agent通过MCP协议调用工具
"""
import asyncio
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os, pathlib

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

script_dir = pathlib.Path(__file__).parent

async def main():
    print("正在连接MCP Server...")

    client = MultiServerMCPClient(
        {
            "shop_service": {
                "command": "python",
                "args": [str(script_dir / "01_mcp_server.py")],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    print(f"获取到 {len(tools)} 个工具")
    for t in tools:
        print(f"  - {t.name}: {t.description}")
    print()

    llm_with_tools = llm.bind_tools(tools)

    async def agent(state: MessagesState):
        # 修复：确保所有ToolMessage的content是字符串
        fixed_messages = []
        for msg in state["messages"]:
            if hasattr(msg, 'content') and isinstance(msg.content, list):
                # DeepSeek要求content是字符串，不能是列表
                import json
                msg.content = json.dumps(msg.content, ensure_ascii=False) if msg.content else ""
            fixed_messages.append(msg)

        messages = [
                       SystemMessage(content="你是智选商城客服。使用提供的工具来回答用户问题。")
                   ] + fixed_messages
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    app = graph.compile()

    print("=" * 50)
    print("通过MCP调用工具的Agent")
    print("=" * 50)

    test_cases = [
        "帮我查一下订单ORD-88892的状态",
        "查一下客户C-002的信息",
        "我要投诉，充电器用了一周就坏了",
    ]

    for question in test_cases:
        print(f"\n用户: {question}")
        # 改成异步调用
        result = await app.ainvoke({"messages": [HumanMessage(content=question)]})
        print(f"Agent: {result['messages'][-1].content}")
        print("-" * 40)

    print()
    print("=" * 50)
    print("MCP的价值：")
    print("1. Agent代码完全不知道工具怎么实现的")
    print("2. 工具可以是Python/Java/Go任何语言写的")
    print("3. 换一个MCP Server，Agent代码一行不用改")
    print("=" * 50)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"报错: {e}")
        import traceback
        traceback.print_exc()