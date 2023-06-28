from src.models import RumiRoBERTa, RumiRobertaConfig
import torch
from transformers import RobertaConfig
import os, shutil


source_path = "../../models/roberta-large"
output_path = "./models/roberta-large"

config = RumiRobertaConfig.from_json_file(os.path.join(source_path, "config.json"))
# model = RumiRoBERTa.from_pretrained("./models/pytorch_model.bin", config=config)

# 跑finetune分类的时候不同的标签数量的数据集要设置不同的num_labels
config.num_labels=2

model = RumiRoBERTa(config)
state_dict =torch.load(os.path.join(source_path,"pytorch_model.bin"))
initial_dict = {}
for key in state_dict.keys():
    if 'roberta.' in key:
        initial_dict[key.replace('roberta.','rumi_model.')] = state_dict[key]
        initial_dict[key.replace('roberta.','answer_model.')] = state_dict[key]
model.load_state_dict(initial_dict,strict=False)
model.save_pretrained(output_path)
shutil.copy(os.path.join(source_path, "tokenizer.json"), output_path)
shutil.copy(os.path.join(source_path, "vocab.json"), output_path)
shutil.copy(os.path.join(source_path, "merges.txt"), output_path)
print(initial_dict.keys())