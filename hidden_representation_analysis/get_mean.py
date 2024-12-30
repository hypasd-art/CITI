import torch
import json
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

def draw():

    with open("./output/Meta-Llama-3-8B-Instruct/fea.json", 'r') as f:  
        fea_origin = json.load(f)

    with open("./output/apibank_llama3_instruct_full/fea.json", 'r') as f:  
        fea_funtune = json.load(f)

    with open("./output/code_alpaca_llama3_instruct_full/fea.json", 'r') as f:  
        fea_funtune_constrast = json.load(f)

    result = torch.zeros(3,32)
    results = torch.zeros(32, 4096)
    result_meta = torch.zeros(10, 32)

    def cosine_similarity(vec1, vec2):  
        dot_product = np.dot(vec1, vec2)  
        norm_vec1 = np.linalg.norm(vec1)  
        norm_vec2 = np.linalg.norm(vec2)  
        return dot_product / (norm_vec1 * norm_vec2) 
    
    for k,v in fea_origin.items():
        for name, para in v.items():
            print(name)
            para = torch.tensor(para)
            v[name] = para.mean(0).numpy().tolist()
    with open("./output/Meta-Llama-3-8B-Instruct/fea_mean.json", 'w') as f:  
        json.dump(fea_origin, f)
    
    # fea_funtune_mean = {}
    for k,v in fea_funtune.items():
        for name, para in v.items():
            para = torch.tensor(para)
            v[name] = para.mean(0).numpy().tolist()
    with open("./output/apibank_llama3_instruct_full/fea_mean.json", 'w') as f:  
        json.dump(fea_funtune, f)

    for k,v in fea_funtune_constrast.items():
        for name, para in v.items():
            para = torch.tensor(para)
            v[name] = para.mean(0).numpy().tolist()
    with open("./output/code_alpaca_llama3_instruct_full/fea_mean.json", 'w') as f:  
        json.dump(fea_funtune_constrast, f)



if __name__ == "__main__":
    draw()