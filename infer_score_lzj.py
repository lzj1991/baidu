import json
import os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel, PeftConfig
import ast


# 定义基础 URL
base_url = "/data0/lj/dataset/baidu/baidu_cut/"
lora_path="/data0/lj/LLaMA-Factory/saves/qwen2_vl-7b/lora/baidu_cut_4"
model_dir = "/data0/lj/model/Qwen2-VL-7B-Instruct"


# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")
# model = PeftModel.from_pretrained(model, lora_path, device_map="auto")

processor = AutoProcessor.from_pretrained(model_dir)

origin_mark = {"baidu1":5,"baidu2":5,
"baidu3":3,"baidu4":2,"baidu5":3,
"baidu6":2,"baidu7":2,"baidu8":4,"baidu9":2,"baidu10":3,"baidu11":1,"baidu12":3,"baidu13":7,"baidu14":2,"baidu15":3,"baidu16":2,"baidu17":2,"baidu18":4,"baidu19":2,"baidu20":2,"baidu21":4,"baidu22":3,"baidu23":6,"baidu24":5,"baidu25":2,"baidu26":3,"baidu27":2,"baidu28":2,"baidu29":2,"baidu30":2,"baidu31":4,"baidu32":2,"baidu33":2,"baidu34":4,"baidu35":4,"baidu36":3,"baidu37":3,"baidu38":4,"baidu39":1,"baidu40":2,"baidu41":1,"baidu42":4,"baidu43":4,"baidu44":6,"baidu45":3,"baidu46":1,"baidu47":1,"baidu48":3,"baidu49":2,"baidu50":1}
text = "在交通规则为右侧通行的背景下，图片中从下向上行驶的主路道路，有几条车道可以进入路口？要求输出格式为json，样式为：{\"count\": 5, \"confidence\": 0.8}，其中count代表可以进入路口的车道数，confidence代表你的答案的置信度。"
# text =  "以图中从下往上方向的黄色隔离线或浅绿色隔离带作为隔离标志，那么在隔离标志的右侧道路从下往上行驶，有几条车道可以进入路口？参考'{\"count\": 5, \"confidence\": 0.8}'这种json格式的字符串输出你推理的结果，其中count代表可以进入路口的车道数，confidence代表你的答案的置信度。"
infer_res_path = "res/qw2vl_7b_lora_infer_cut_prompt_a.json"
ground_truth_path = "/data0/lj/LLaMA-Factory/data/baidu_cut.json"
score_path = "res/qw2vl_7b_lora_score_cut_prompt_a.json"
def get_response(image_path):
    # image_path = "/data1/lj/dataset/baidu/baidu_origin/baidu1.jpg"
    image = Image.open(image_path)

    conversation =  [
            {
                "role": "user",
                "content": [
                    {"image": image_path},
                    {"text": text}
                ]
            }
        ]

    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text


def post_process():
    with open(infer_res_path, "r") as f:
        responses = json.load(f)

        plus_results = []
        for resp in responses:
            try:
                extracted_dict = ast.literal_eval(resp["response"][0])
            except:
                print("Error format:", resp)
                continue
            if extracted_dict:
                image_name = resp["image_id"].split(".")[0]
                origin_value = origin_mark.get(image_name, "default_value")
                # 判断 origin_value 和 extracted_dict['count'] 是否相等
                judge = origin_value == extracted_dict.get('count', -1)
                # 将图片编号和字典结果一起存储
                plus_results.append(
                    {"image": resp["image_id"], "origin": origin_value, "result": extracted_dict,
                        "judge": judge})

        print("======results======")
        accuracy = sum(1 for item in plus_results if item["judge"]) / len(plus_results)
        plus_results.append({"accuracy":accuracy})
        print("accuracy:", accuracy)
        with open(score_path, "w") as f:
            json.dump(plus_results, f,ensure_ascii=False,indent=4)

def post_process_v2():
    # 处理输出为 ["{'down-up': 4, 'down-right': 0}"]
    # get true label
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
        origin_mark = {}
        for item in ground_truth:
            image_name = item["images"][0].split("/")[-1]
            origin_mark[image_name] = ast.literal_eval(item["conversations"][1]["value"])
    print(origin_mark)
    # get infer result

    with open(infer_res_path, "r") as f:
        responses = json.load(f)

        plus_results = []
        for resp in responses:
            try:
                extracted_dict = ast.literal_eval(resp["response"][0])
            except:
                print("Error format:", resp)
                continue
            if extracted_dict:
                image_name = resp["image_id"]
                origin_value = origin_mark.get(image_name, "default_value")
                # 判断 origin_value 和 extracted_dict['count'] 是否相等
                down_up = extracted_dict["down-up"] == origin_value["down-up"]
                down_right = extracted_dict["down-right"] == origin_value["down-right"]
                judge =[down_up, down_right]
                # 将图片编号和字典结果一起存储
                plus_results.append(
                    {"image": resp["image_id"], "origin": origin_value, "result": extracted_dict,
                        "judge": judge})

        print("======results======")
        down_up_acc = sum(1 for item in plus_results if item["judge"][0]) / len(plus_results)
        down_right_acc = sum(1 for item in plus_results if item["judge"][1]) / len(plus_results)
        accuracy = (down_up_acc + down_right_acc) / 2
        both_true = sum(1 for item in plus_results if item["judge"] == [True, True]) / len(plus_results)
        plus_results.append({"down-up":down_up_acc, "down-right":down_right_acc})
        plus_results.append({"accuracy":accuracy})
        plus_results.append({"both_true":both_true})
        print("accuracy:", accuracy)
        print("down-up:", down_up_acc)
        print("down-right:", down_right_acc)
        print("both_true:", both_true)
        with open(score_path, "w") as f:
            json.dump(plus_results, f,ensure_ascii=False,indent=4)

def infer():
    # 循环调用 MultiModalConversation.call
    responses=[]
    images = os.listdir(base_url)
    for image in images:
        image_url = os.path.join(base_url, image)
        response = get_response(image_url)
        responses.append({"image_id":image,"text":text,"response":response})
        print("response:", response)

    with open(infer_res_path, "w") as f:
        json.dump(responses, f,ensure_ascii=False,indent=4)
infer()
post_process()
#post_process_v2()