from model import RumiDeBERTa, RumiRoBERTa
import torch
from model import RumiRobertaConfig, RumiDebertaConfig
def initialize_deberta():
    config = RumiDebertaConfig.from_json_file("./models/vanilla_deberta/deberta-large/config.json")
    model = RumiDeBERTa(config)
    state_dict =torch.load("./models/vanilla_deberta/deberta-large/pytorch_model.bin")
    initial_dict = {}
    for key in state_dict.keys():
        if 'deberta.' in key:
            initial_dict[key.replace('deberta.','rumi_model.')] = state_dict[key]
            initial_dict[key.replace('deberta.','answer_model.')] = state_dict[key]
    model.load_state_dict(initial_dict,strict=False)
    model.save_pretrained('./models/deberta_large')

def initialize_roberta():
    config = RumiRobertaConfig.from_json_file("./models/vanilla_roberta/config.json")
    model = RumiRoBERTa(config)
    state_dict =torch.load("./models/vanilla_roberta/pytorch_model.bin")
    initial_dict = {}
    for key in state_dict.keys():
        if 'roberta.' in key:
            initial_dict[key.replace('roberta.','rumi_model.')] = state_dict[key]
            initial_dict[key.replace('roberta.','answer_model.')] = state_dict[key]
    model.load_state_dict(initial_dict,strict=False)
    model.save_pretrained('./models/roberta_large')

def initialize_t5():
    config = T5Config.from_json_file("./models/t5-small/config.json")
    model = RumiT5(config)
    state_dict =torch.load("./models/t5-small/pytorch_model.bin")
    # print(state_dict.keys())
    initial_dict = {}
    for key in state_dict.keys():
        initial_dict[key] = state_dict[key]
        initial_dict['rumi_'+key] = state_dict[key]
    model.load_state_dict(initial_dict,strict=False)
    model.save_pretrained('./models/t5-small')
# print(initial_dict.keys())
# print(model.state_dict)
initialize_deberta()