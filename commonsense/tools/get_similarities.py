from sentence_transformers import SentenceTransformer, util

import os
import json
from tqdm import tqdm

input_file = '../data/socialiqa/train_filter_mentions.jsonl'
ouput_file = '../data/socialiqa/train_filter_mentions_similarities.jsonl'

raw_datas = []

with open(input_file, "r", encoding="utf-8") as fin:
    lines = fin.readlines()
    fin.close()

for line in lines:
    line = json.loads(line.strip("\n"))
    raw_datas.append(line)

model = SentenceTransformer('./models/all-roberta-large-v1')
for raw_data in tqdm(raw_datas):
    num_mentions = len(raw_data['filter_mentions'])

    if num_mentions > 0:
        question = raw_data['context'] + ' ' + raw_data['question']
        entities = raw_data['filter_mentions']

        question_embedding = model.encode(question, convert_to_tensor=True)
        question_embedding = question_embedding.unsqueeze(0).repeat(num_mentions, 1)
        entities_embedding = model.encode(entities, convert_to_tensor=True)

        ##取对角线
        similarity = util.pytorch_cos_sim(question_embedding, entities_embedding)
        raw_data['similarities'] = similarity.diag().cpu().numpy().tolist()
    else:
        raw_data['similarities'] = []

    assert len(raw_data['filter_mentions']) == len(raw_data['similarities'])

with open(ouput_file, "w") as json_file:
    for raw_data in raw_datas:
        json.dump(raw_data, json_file)
        json_file.write('\n')