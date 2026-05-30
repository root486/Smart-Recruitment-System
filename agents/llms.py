import os
from langchain_openai import ChatOpenAI
ds_api_key = os.getenv("DEEPSEEK_API_KEY")
api_key = os.getenv("DASHSCOPE_API_KEY")




deepseek_direct_llm= ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=ds_api_key
)
qwen_llm = ChatOpenAI(
    model="qwen3-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key
)

