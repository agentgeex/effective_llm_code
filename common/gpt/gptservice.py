## Temperature 设置  https://api-docs.deepseek.com/zh-cn/quick_start/parameter_settings
## 测试的路径  <你的 OpenAI 兼容服务地址>/v1/   （或 https://api.deepseek.com）
## 本地 xinference 环境模式测试路径  https://api.deepseek.com
## deepseek-reasoner deepseek-chat

"""
如果是思维链中的所有内容都需要大模型阅读，可以尝试加上以下内容：

{"role": "system", "content": "请记住用户分步发送的所有数据块，最后统一回答问题。"}

在最后需要回答问题的时候加上这句话

{
    "role": "user", 
    "content": "根据你记住的所有数据块，回答问题：量子计算如何解决加密问题？"
}


"""


"""

oepnapi返回的数据格式


{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "message": {
                "content": "十字花科是一个经济价值较大的科",
                "role": "assistant"
            }
        }
    ],
    "created": 1744365415,
    "id": "0217443653942326a19b6f80fc3c2ee82a57ca814e89664215f24",
    "model": "doubao-1-5-pro-32k-250115",
    "service_tier": "default",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 481,
        "prompt_tokens": 26,
        "total_tokens": 507,
        "prompt_tokens_details": {
            "cached_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0
        }
    }
}
"""



import os
from openai import OpenAI
from config.cfg import Syscfg
from common.log import logservice
cfg = Syscfg()
logger = logservice.getLog("")

## 密钥与中转地址改为从环境变量读取；未设置时用显式占位符构造客户端，
## 这样 import 不会因缺 key 直接崩，真正调用时才报鉴权错误（占位符不是有效凭据）
GPT_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or "sk-REPLACE_ME"
GPT_BASE_URL = os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com"

client = OpenAI(api_key=GPT_API_KEY, base_url=GPT_BASE_URL)


def get_gpt_client(api_key="",base_url="https://api.deepseek.com"):
    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client



def simplechat(messages_cont, stream=False, max_tokens=4096, temperature=1.0, response_format={}, client=client,model="deepseek-chat"):

    logger.info(f"GPT SERVER URL IS {client.base_url}")
    """
    简单聊天函数，用于与模型进行对话。

    参数:
    messages_cont (list): 对话消息列表。
    stream (bool): 是否使用流式传输。
    max_tokens (int): 限制输出长度，默认为4096。
    temperature (float): 控制随机性，默认为1.0。
    response_format (dict): 响应格式，默认为空字典。
    _client (object): 客户端对象。

    返回:
    content (str): 模型生成的回复内容。
    response (object): 完整的响应对象。
    """

    if not response_format:
        response = client.chat.completions.create(
                                                    model=model,
                                                    messages=messages_cont,
                                                    stream=stream,
                                                    max_tokens=max_tokens,  # 限制输出长度 32*1000=32K
                                                    temperature=temperature,  # 控制随机性（0-2）
                                                    # top_p=0.9,       # 控制多样性（0-1）
                                                    #response_format=response_format
                                                )
    # 如果response_format不为空，则传入response_format参数
    else:
        response = client.chat.completions.create(
                                                    model=model,
                                                    messages=messages_cont,
                                                    stream=stream,
                                                    max_tokens=max_tokens,  # 限制输出长度 32*1000=32K
                                                    temperature=temperature,  # 控制随机性（0-2）
                                                    # top_p=0.9,       # 控制多样性（0-1）
                                                    response_format=response_format
                                                )

    ## 推理过程
    #reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content

    return content, response


def  reasoning_chat(messages_cont,stream=False,max_tokens=4096,temperature=1.0,response_format={},client=client):

    # 如果response_format为空，则使用默认的response_format
    if not response_format:
        response = client.chat.completions.create(
                                                    model="deepseek-chat",
                                                    messages=messages_cont,
                                                    stream=stream,
                                                    max_tokens=max_tokens,  # 限制输出长度 32*1000=32K
                                                    temperature=temperature,  # 控制随机性（0-2）
                                                    # top_p=0.9,       # 控制多样性（0-1）
                                                    #response_format=response_format
                                                )
    else:
        response = client.chat.completions.create(
                                                    model="deepseek-chat",
                                                    messages=messages_cont,
                                                    stream=stream,
                                                    max_tokens=max_tokens,  # 限制输出长度 32*1000=32K
                                                    temperature=temperature,  # 控制随机性（0-2）
                                                    # top_p=0.9,       # 控制多样性（0-1）
                                                    #response_format=response_format
                                                )

    # 获取推理内容
    reasoning_content = response.choices[0].message.reasoning_content
    ##print(reasoning_content)
    # 获取输出内容
    content = response.choices[0].message.content

    # 返回输出内容和推理内容
    return content,reasoning_content,response


