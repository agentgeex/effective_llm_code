"""
【本书配套示例】FastMCP 入门服务端：1 个工具 get_stock_price + 静态/动态资源 + 1 个 prompt
对应章节：ch07 §7.3.2 MCP开发神器：FastMCP（书稿对应小节： 工具、 资源与 prompt）
运行方式：
    # 默认 STDIO（书稿对应小节 的 `python stock_server.py`）
    python ch07/stock_server.py
    # SSE 模式（配合 mcp/mcp_client.py 那种客户端）
    python ch07/stock_server.py --sse       # 监听 127.0.0.1:8011/sse
依赖：fastmcp>=2.0（实测 4.1.4）。缺 fastmcp → ImportError: No module named 'fastmcp'
    实测解释器：.venv/bin/python（fastmcp 4.1.4）
实测状态：✅ 跑通（本文件重跑取证）
    客户端：fastmcp 4.1.4（stdio 子进程）
        tools      = ['get_stock_price']
        resources  = ['config://app']                 # 静态资源直接列在这里
        templates  = ['user://{user_id}']             # 动态资源在 resource_templates 里
        prompts    = ['code_review_template']
        get_stock_price({"ticker":"aapl"})   -> AAPL 最新价：198.5（大小写不敏感）
        get_stock_price({"ticker":"600519"}) -> 600519 最新价：1650.0
        get_stock_price({"ticker":"XXXX"})   -> 未找到股票代码 XXXX 的行情数据
        read_resource("user://42")           -> {"id": "42", "name": "\u5f20\u4e09", "role": "engineer"}
                                                （张三被 JSON 序列化成 unicode 转义，正常现象）
        read_resource("config://app")        -> "# 示例内容（未找到 config.yaml）…"
    实测时间：

与书稿差异（两处）：
  1. **`config://app` 不再直接 `open("config.yaml").read()`**。书稿那样写，仓库里没有
     config.yaml 时一读资源就抛 FileNotFoundError（整个资源不可用）。
     这里改成：文件存在就读真的，不存在就返回一段示例文本并注明「示例内容」。
     这样读者第一次跑不会撞报错，真放一个 config.yaml 进去又能读到真实内容。
  2. 启动方式加 `--sse` 开关（书稿只演示默认 stdio）。
  其余与书稿对应小节、 一致：工具逻辑（三只票的价格表、未命中提示）、
  `user://{user_id}` 动态资源、`code_review_template` prompt 均原样。
"""

import os
import sys

from fastmcp import FastMCP

mcp = FastMCP("MyStockServer")


@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """获取实时股价，支持 A 股和美股代码。"""
    # 此处对接你的数据库或API逻辑
    prices = {
        "AAPL": 198.50,
        "TSLA": 245.30,
        "600519": 1650.00,  # 贵州茅台
    }
    price = prices.get(ticker.upper())
    if price is None:
        return f"未找到股票代码 {ticker} 的行情数据"
    return f"{ticker.upper()} 最新价：{price}"


# 暴露静态资源
@mcp.resource("config://app")
def get_config() -> str:
    """返回应用配置文件内容。"""
    path = os.environ.get("APP_CONFIG_PATH", "config.yaml")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    return ("# 示例内容（未找到 %s）\n"
            "# 把真实的 config.yaml 放到当前目录，或 export APP_CONFIG_PATH=/path/to/config.yaml\n"
            "app:\n  name: demo\n  debug: false\n") % path


# 暴露动态资源（带参数）
@mcp.resource("user://{user_id}")
def get_user_profile(user_id: str) -> dict:
    """返回指定用户的资料。"""
    return {"id": user_id, "name": "张三", "role": "engineer"}


# 暴露提示词模板
@mcp.prompt()
def code_review_template() -> str:
    """生成代码审查的标准提示词。"""
    return "请对以下代码进行全面审查，重点关注安全漏洞和性能问题：\n\n{code}"


if __name__ == "__main__":
    if "--sse" in sys.argv:
        mcp.run(transport="sse", host="127.0.0.1", port=8011)
    else:
        mcp.run()  # 默认 stdio：书稿对应小节 的用法
