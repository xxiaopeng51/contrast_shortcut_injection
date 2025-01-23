import openai

# 设置你的 OpenAI API 密钥

openai.api_key = "sk-NWanOVpX6msv9GzjC2Bc2b757eBb470bBfDc23D2037c8a14"

proxy = {
'http': 'http://122.96.144.223:7890',
'https': 'http://122.96.144.223:7890'
}

openai.proxy = proxy

#限速率：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
#苏：openai.api_key = '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'
# 发送请求
response = openai.Completion.create(
    engine="text-davinci-002",  # 或者其他你想使用的引擎
    prompt="Say this is a test!",
    temperature=0.7
)

# 输出生成的文本
print(response["choices"][0]["text"])