"""
【本书配套示例】任务管理 MCP Server：create_task / list_tasks / complete_task 三个工具
对应章节：ch07 §7.3.3 开发MCP服务器端（书稿对应小节，完整示例 task_server.py）
运行方式：
    # 默认 STDIO（供 Host/客户端以子进程拉起）
    python ch07/task_server.py
    # 想用 SSE（书稿对应小节 那种改法）：
    python ch07/task_server.py --sse        # 监听 127.0.0.1:8011/sse
    # 用官方调试器
    mcp dev ch07/task_server.py
依赖：fastmcp>=2.0（实测 4.1.4）。缺 fastmcp → ImportError: No module named 'fastmcp'
    实测解释器：.venv/bin/python（fastmcp 4.1.4）
实测状态：✅ 跑通（本文件重跑取证）
    客户端：fastmcp 4.1.4（stdio 子进程）
        tools = ['create_task', 'list_tasks', 'complete_task']
        create_task({"title":"写书","priority":"high"})
            -> {"message":"任务创建成功","task":{"id":"xxxxxxxx","title":"写书","priority":"high",
                "status":"pending","created_at":"T23:xx:xx"}}
        list_tasks({})          -> {"count":1,"tasks":[...]}
        create_task({"title":"x","priority":"urgent"})
            -> {"error":"无效的优先级 'urgent'，可选值：{'high', 'normal', 'low'}"}（参数校验生效）
        complete_task({"task_id":"xxxxxxxx"}) -> status 变为 done，带 completed_at
    实测时间：

与书稿差异：
  1. 业务逻辑（三个工具、优先级/状态白名单校验、内存字典 tasks_store）**与书稿对应小节 完全一致**。
  2. 末尾的启动方式：书稿对应小节 让你「把 mcp.run() 改为 mcp.run(transport="sse", ...)」，
     手工改一次只能跑一种。这里加 `--sse` 开关，默认 stdio，两种都能用、不用改代码。
  3. 文件头补了六项注释；函数体一个字未动。
"""

import sys
import uuid
from datetime import datetime
from typing import Optional

from fastmcp import FastMCP

mcp = FastMCP("TaskManager")

# --- 内存存储（生产环境请替换为数据库） ---
tasks_store = {}


@mcp.tool()
def create_task(title: str, priority: Optional[str] = "normal") -> dict:
    """
    创建一个新任务。

    参数:
        title: 任务标题
        priority: 优先级，可选 high/normal/low，默认 normal
    """
    valid_priorities = {"high", "normal", "low"}
    if priority not in valid_priorities:
        return {"error": f"无效的优先级 '{priority}'，可选值：{valid_priorities}"}

    task_id = str(uuid.uuid4())[:8]
    task = {
        "id": task_id,
        "title": title,
        "priority": priority,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
    }
    tasks_store[task_id] = task
    return {"message": "任务创建成功", "task": task}


@mcp.tool()
def list_tasks(status: Optional[str] = None) -> dict:
    """
    列出所有任务，可按状态过滤。

    参数:
        status: 可选的状态过滤。留空则返回全部任务。
               可选值：pending, in_progress, done
    """
    if status and status not in {"pending", "in_progress", "done"}:
        return {"error": f"无效的状态 '{status}'，可选值：pending, in_progress, done"}

    filtered = tasks_store.values()
    if status:
        filtered = [t for t in filtered if t["status"] == status]

    return {"count": len(list(filtered)), "tasks": list(filtered)}


@mcp.tool()
def complete_task(task_id: str) -> dict:
    """
    将指定任务标记为已完成。

    参数:
        task_id: 任务的唯一 ID
    """
    task = tasks_store.get(task_id)
    if task is None:
        return {"error": f"任务 {task_id} 不存在"}

    task["status"] = "done"
    task["completed_at"] = datetime.now().isoformat()
    return {"message": f"任务 {task_id} 已完成", "task": task}


if __name__ == "__main__":
    if "--sse" in sys.argv:
        mcp.run(transport="sse", host="127.0.0.1", port=8011)
    else:
        mcp.run()  # 默认 stdio
