import json
with open("output_all.json","r") as f:
    data = json.load(f)


rows = []

for item in data:
    rows.append({
        "messages": [
            {
                "role": "user",
                "content": item["prompt"]
            },
            {
                "role": "assistant",
                "content": item["response"],
            }
        ]
    })

with open("sft_train.json","w") as f:
    json.dump(rows,f,ensure_ascii=False,indent=2)