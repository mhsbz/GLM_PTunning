import json

with open("./output_all.json","r") as f:
    data = json.load(f)

new_data = []

for item in data:
    new_data.append({
            "messages": [
                {
                    "role": "user",
                    "content": item["prompt"]
                },
                {
                    "role": "assistant",
                    "content": item["response"]
                },
            ]      
    })

with open("./output_all_train.json","w") as f:
    json.dump(new_data,f,ensure_ascii=False,indent=4)