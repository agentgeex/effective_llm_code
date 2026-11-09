##  

import os
from volcenginesdkarkruntime import Ark

client = Ark(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=os.environ.get("ARK_API_KEY"),
                timeout=120,
                max_retries=2,
            )

completion = client.chat.completions.create(
    model="doubao-1-5-pro-32k-250115",
    messages=[
                {"role": "system", "content": "你是一个模型助手"},
                {"role": "user", "content": "今天天气不错"},
            ],
)
print(completion.choices[0].message.content)