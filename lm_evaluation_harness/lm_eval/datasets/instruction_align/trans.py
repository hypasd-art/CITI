import json
data_path = "/netdisk/yphao/mlpmoe/lm_evaluation_harness/lm_eval/datasets/instruction_align/question.jsonl"
data = []
json_data=[]
with open(data_path, "r" ) as f:
    for line in f:
        json_data.append(json.loads(line))
    for item in json_data:
        data.append({"instruction": "", "input": item["question"], "output": item["answer"]})

with open("/netdisk/yphao/mlpmoe/lm_evaluation_harness/lm_eval/datasets/instruction_align/test_align.json", 'w') as fb:
    json.dump(data, fb, indent=2)


