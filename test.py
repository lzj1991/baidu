from time import sleep

import dashscope
import json
import re

# 定义 API Key 和模型
api_key = "xxxxxxxx"

plus_model = 'qwen-vl-plus'
max_model_0201 = 'qwen-vl-max-0201'
max_model = 'qwen-vl-max'
instruct_model = 'qwen2-vl-7b-instruct'

model = plus_model

# 定义基础 URL
base_url = "http://xxxxxx.oss-cn-zhangjiakou.aliyuncs.com/xxxxxx/xxxxx/"

origin_mark = {"baidu1":5,"baidu2":5,"baidu3":3,"baidu4":2,"baidu5":3,"baidu6":2,"baidu7":2,"baidu8":4,"baidu9":2,"baidu10":3,"baidu11":1,"baidu12":3,"baidu13":7,"baidu14":2,"baidu15":3,"baidu16":2,"baidu17":2,"baidu18":4,"baidu19":2,"baidu20":2,"baidu21":4,"baidu22":3,"baidu23":6,"baidu24":5,"baidu25":2,"baidu26":3,"baidu27":2,"baidu28":2,"baidu29":2,"baidu30":2,"baidu31":4,"baidu32":2,"baidu33":2,"baidu34":4,"baidu35":4,"baidu36":3,"baidu37":3,"baidu38":4,"baidu39":1,"baidu40":2,"baidu41":1,"baidu42":4,"baidu43":4,"baidu44":6,"baidu45":3,"baidu46":1,"baidu47":1,"baidu48":3,"baidu49":2,"baidu50":1}

max_retries = 3

def extract_dict_from_response(response):
    """
    从响应中提取所需的字典
    :param response: API 响应
    :return: 提取的字典或 None
    """
    print(response)
    if not response or 'output' not in response:
        print("Response is None or does not contain 'output'")
        return {"count": -1, "confidence": -1}

    output = response['output']
    if not output or 'choices' not in output:
        print("Output is None or does not contain 'choices'")
        return {"count": -1, "confidence": -1}

    choices = output['choices']
    if not choices or len(choices) == 0:
        print("Choices is None or empty")
        return {"count": -1, "confidence": -1}

    choice = choices[0]
    if not choice or 'message' not in choice:
        print("Choice is None or does not contain 'message'")
        return {"count": -1, "confidence": -1}

    message = choice['message']
    if not message or 'content' not in message:
        print("Message is None or does not contain 'content'")
        return {"count": -1, "confidence": -1}

    content = message['content']
    if not content or len(content) == 0:
        print("Content is None or empty")
        return {"count": -1, "confidence": -1}

    content_item = content[0]
    if not content_item or 'text' not in content_item:
        print("Content item is None or does not contain 'text'")
        return {"count": -1, "confidence": -1}

    content_text = content_item['text']

    # 使用正则表达式提取 JSON 字符串
    match = re.search(r'{.*?}', content_text, re.DOTALL)
    if match:
        extracted_dict_str = match.group(0)
        # print(f"Extracted JSON string: {extracted_dict_str}")  # 打印提取的字符串
        try:
            # 将提取的字符串解析为字典
            extracted_dict = json.loads(extracted_dict_str.replace("'", '"').replace('\n', '').replace(' ', ''))
            return extracted_dict
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print(f"Failed to parse: {extracted_dict_str}")
            return {"count": -1, "confidence": -1}
    else:
        print("未找到匹配的 JSON 字符串")
        return {"count": -1, "confidence": -1}


# 存储所有图片的字典结果
plus_results = []

# 循环调用 MultiModalConversation.call
for i in range(1, 51):
    image_url = f"{base_url}baidu{i}.jpg"
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_url},
                {
                    "text": "在交通规则为右侧通行的背景下，图片中从下向上行驶的主路道路，有几条车道可以进入路口？参考'{\"count\": 5, \"confidence\": 0.8}'这种json格式的字符串输出你推理的结果，其中count代表可以进入路口的车道数，confidence代表你的答案的置信度。"
                }
            ]
        }
    ]

    retries = 0
    while retries < max_retries:
        response = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages
        )

        # 提取所需的字典
        extracted_dict = extract_dict_from_response(response)

        if extracted_dict and 'usage' in response and response['usage']:
            usage = response.get('usage', {})
            total_tokens = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
            # 从 origin_mark 获取对应 baidu{i} 的值
            origin_value = origin_mark.get(f"baidu{i}", "default_value")
            # 判断 origin_value 和 extracted_dict['count'] 是否相等
            judge = origin_value == extracted_dict.get('count', -1)
            # 将图片编号和字典结果一起存储
            plus_results.append(
                {"image": f"baidu{i}", "origin": origin_value, "result": extracted_dict, "tokens": total_tokens,
                 "judge": judge})
            break  # 成功提取结果，跳出重试循环
        else:
            print(f"Failed to extract valid result for baidu{i}, retrying... (attempt {retries + 1}/{max_retries})")
            retries += 1
            sleep(7)

    if retries == max_retries:
        print(f"Max retries reached for baidu{i}, giving up.")


print("======results======")
print(plus_results)

