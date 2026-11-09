"""
【本书配套示例】把本机 MySQL 包装成 MCP Server：3 个工具 + 1 个数据采样资源模板
① 这是什么：query_database / list_tables / list_databases 三个工具，外加
   mysql://{table_name}/data 动态资源（取指定表前 50 行，供模型先看真实数据再写 SQL）。
② 对应书稿：ch07 §7.3.3 开发MCP服务器端的进阶形态 —— 把真实数据源挂成工具；
   服务端骨架另可对照 ch11 §11.5.2（书稿对应小节 起）。
③ 怎么跑：
   export MYSQL_PASSWORD=<你的 MySQL 口令>      # 未设置时脚本直接 RuntimeError 退出，不会静默用空口令
   .venv/bin/python ch07/mcp_server_mysql.py             # 默认 stdio，供客户端以子进程拉起
   # 要用 SSE：把文件末尾 mcp.run() 换成
   #   mcp.run(transport="sse", host="127.0.0.1", port=8011)
④ 依赖：fastmcp>=2.0（实测 4.1.4）、pymysql（实测 1.2.3）；数据库需存在 `coingeek` 库。
⑤ 实测解释器与结果：python
   （fastmcp 4.1.4 / mcp 2.3.0 / pymysql 1.2.3）， 复测
   - 不设 MYSQL_PASSWORD → rc=1，RuntimeError: 缺少环境变量 MYSQL_PASSWORD ...
   - 设 MYSQL_PASSWORD=占位口令 + mcp 2.3.0 低层 ClientSession（stdio）：
       ✅ 握手通过，list_tools → ['query_database','list_tables','list_databases']
       ✅ list_resource_templates → ['mysql://{table_name}/data']（list_resources 为空，符合预期）
       ❌ call_tool('list_tables') → is_error=True，
          Error calling tool 'list_tables': (1045, "Access denied for user 'root'@'172.17.0.1'
          (using password: YES)")
          —— 我没有也不该有本机 MySQL 的 root 口令，且本机没有 coingeek 库。
          换成你自己的 MySQL + 正确库名口令即可，代码不用改。
⑥ 与书稿的差异：
   1. 口令来源：书稿把 MySQL 口令硬编码在 DB_CONFIG 里，这里读环境变量 MYSQL_PASSWORD 并在启动时校验
      （缺变量直接报错退出）。安全改动，业务逻辑不变。
   2. 末尾 `mcp.run()` 为默认 stdio；书稿那行 SSE 配置以注释保留，按需启用。
   3. 保留作者原注释「关键黑科技：手动挂载资源列表提供者，绕过装饰器 Bug」—— 该行本就是注释，
      保留是为了让读者知道他踩过什么坑。
   附带一条实测提醒：用 fastmcp 4.1.4 的 PythonStdioTransport 拉本文件会 MCPError: Connection closed，
   因为该 transport 默认不透传环境变量、服务端因缺 MYSQL_PASSWORD 直接退出；
   取证请用 mcp 2.3.0 低层 ClientSession 并显式传入 env。
"""

import os
from fastmcp import FastMCP
import fastmcp
import pymysql
from mcp.types import Resource

# 你的数据库配置
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": os.environ.get("MYSQL_PASSWORD", ""),  # 口令改为环境变量，见 .env.example
    "database": "coingeek",
    "cursorclass": pymysql.cursors.DictCursor
}

if not DB_CONFIG["password"]:
    raise RuntimeError("缺少环境变量 MYSQL_PASSWORD：请先 export MYSQL_PASSWORD=<mysql 口令>（见 .env.example）")

mcp = FastMCP("MySQL-Reflection-Server")


@mcp.tool()
def query_database(sql: str):
    """执行 SQL 查询语句。
    请先通过 list_tables 和 describe_table 确认表结构后再执行。
    """
    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetchall()
    finally:
        connection.close()


@mcp.tool()
def list_tables():
    """列出数据库（coingeek）中所有的表及其注释。"""
    return query_database("SELECT table_name, table_comment FROM information_schema.tables WHERE table_schema = 'coingeek'")


@mcp.tool()
def list_databases():
    """列出当前服务器上所有的数据库名称及其业务用途。"""
    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            # 获取所有数据库
            cursor.execute("SHOW DATABASES")
            databases = cursor.fetchall()

            # MySQL 5.7的information_schema.SCHEMATA没有SCHEMA_COMMENT列
            # 所以只返回数据库名称，业务用途需要根据名称推断
            result = []
            for db in databases:
                db_name = db['Database']
                # 根据数据库名称推断业务用途
                description = "无描述"
                if db_name == "information_schema":
                    description = "MySQL系统数据库，存储元数据信息"
                elif db_name == "mysql":
                    description = "MySQL系统数据库，存储用户权限等信息"
                elif db_name == "performance_schema":
                    description = "MySQL性能监控数据库"
                elif db_name == "sys":
                    description = "MySQL系统数据库，提供性能监控视图"
                elif "coingeek" in db_name.lower():
                    description = "加密货币相关业务数据库"
                elif "ai_agent" in db_name.lower():
                    description = "AI代理相关业务数据库"
                elif "busenv" in db_name.lower():
                    description = "业务环境配置数据库"
                elif "cpne" in db_name.lower():
                    description = "CPNE相关业务数据库"
                elif "eshowcrm" in db_name.lower():
                    description = "EShow CRM系统数据库"
                elif "fabric" in db_name.lower():
                    description = "Fabric相关业务数据库"
                elif "motozolo" in db_name.lower():
                    description = "Motozolo控制台库数据库"
                elif "poolguarder" in db_name.lower():
                    description = "PoolGuarder模型训练数据库"
                elif "vonescan" in db_name.lower():
                    description = "VoneScan相关业务数据库"

                result.append({
                    "数据库名称": db_name,
                    "业务用途": description
                })

            return result
    finally:
        connection.close()

# --- 1. 定义一个普通函数（不加装饰器！） ---
# --- 核心资源：按需读取（这是你之前唯一跑通的装饰器） ---


@mcp.resource("mysql://{table_name}/data")
def table_data_resource(table_name: str) -> str:
    """
    【动态数据采样资源】

    作用：获取指定数据库表的前 50 行真实采样数据。

    参数说明：
    - {table_name}: 必须替换为实际的 MySQL 表名（例如：t_chip_distribution 或 t_stock_l4）。

    使用场景：
    当你想编写 SQL 但不确定某个字段的具体数值分布、精度（如 profit_ratio 的范围）或日期格式时，请读取此资源。
    """
    # 1. 基础校验：防止 SQL 注入（虽然是只读，但架构师的习惯要好）
    # 简单过滤非字符内容
    clean_table_name = "".join(filter(lambda x: x.isalnum() or x == '_', table_name))

    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            # 2. 执行查询：只取前 50 条，避免 Token 爆炸
            sql = f"SELECT * FROM `{clean_table_name}` LIMIT 50"
            cursor.execute(sql)

            # 3. 获取列名 (cursor.description 包含了元数据)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                return f"表 `{clean_table_name}` 目前是空的，或者不存在。"

            # 4. 格式化输出：采用 Markdown 表格格式（AI 对 Markdown 的理解力最强）
            # 生成表头
            header = "| " + " | ".join(columns) + " |"
            separator = "| " + " | ".join(["---"] * len(columns)) + " |"

            # 生成数据行
            data_rows = []
            for row in rows:
                # 处理 None 值，并转为字符串
                formatted_row = [str(val) if val is not None else "NULL" for val in row.values()]
                data_rows.append("| " + " | ".join(formatted_row) + " |")

            # 5. 组合最终文档
            result = [
                f"### 数据样本: {clean_table_name} (Top 50 rows)",
                f"提示：以下是数据库表 `{clean_table_name}` 的真实采样数据。",
                "",
                header,
                separator
            ] + data_rows

            return "\n".join(result)

    except Exception as e:
        return f"读取资源失败: 错误信息 {str(e)}"
    finally:
        connection.close()


# --- 3. 在启动前，手动把逻辑“塞”进去 ---
if __name__ == "__main__":
    # 关键黑科技：手动挂载资源列表提供者，绕过装饰器 Bug
    # mcp.list_resources(list_my_databases_logic)
    mcp.run()
    # mcp.run(transport='sse', host="127.0.0.1", port=8011)
