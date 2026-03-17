"""
第一步：用Python写一个MCP Server，理解MCP的工作方式
MCP Server = 把你的功能按标准格式暴露出去，任何Agent都能调
"""
from mcp.server.fastmcp import FastMCP

# 创建MCP Server（对标Spring Boot的@RestController）
mcp = FastMCP("智选商城服务")

# 定义工具（对标@GetMapping / @PostMapping）
@mcp.tool()
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态和物流信息"""
    # 模拟数据库查询（你的项目里这里会查真实数据库）
    orders = {
        "ORD-88892": {"order_id": "ORD-88892", "status": "运输中", "courier": "顺丰", "eta": "明天下午3点"},
        "ORD-66543": {"order_id": "ORD-66543", "status": "已签收", "time": "昨天上午10点"},
    }
    import json
    order = orders.get(order_id, {"error": f"订单{order_id}不存在"})
    return json.dumps(order, ensure_ascii=False)

@mcp.tool()
def query_customer(customer_id: str) -> str:
    """查询客户信息"""
    customers = {
        "C-001": {"name": "张三", "level": "黄金会员", "total_orders": 28},
        "C-002": {"name": "李四", "level": "钻石会员", "total_orders": 156},
    }
    import json
    customer = customers.get(customer_id, {"error": f"客户{customer_id}不存在"})
    return json.dumps(customer, ensure_ascii=False)

@mcp.tool()
def create_ticket(title: str, description: str, priority: str = "medium") -> str:
    """创建客服工单"""
    import json
    ticket = {
        "ticket_id": "TK-20240001",
        "title": title,
        "description": description,
        "priority": priority,
        "status": "已创建"
    }
    return json.dumps(ticket, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run(transport="stdio")