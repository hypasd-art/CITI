import torch
import json
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

def draw(type_module = "mlp"):

    with open("./output/Meta-Llama-3-8B-Instruct/fea_mean.json", 'r') as f:  
        fea_origin = json.load(f)

    with open("./output/apibank_llama3_instruct_full/fea_mean.json", 'r') as f:  
        fea_funtune = json.load(f)

    with open("./output/code_alpaca_llama3_instruct_full/fea_mean.json", 'r') as f:  
        fea_funtune_constrast = json.load(f)

    result = torch.zeros(3,32)
    results = torch.zeros(32, 4096)
    result_meta = torch.zeros(10, 32)

    def cosine_similarity(vec1, vec2):  
        dot_product = np.dot(vec1, vec2)  
        norm_vec1 = np.linalg.norm(vec1)  
        norm_vec2 = np.linalg.norm(vec2)  
        return dot_product / (norm_vec1 * norm_vec2) 
    
    

    for k, v in fea_funtune.items():
        if k == "tool":
            for name, para in v.items():
                para_origin = fea_origin[k][name] # tool layer llama3
                para = torch.tensor(para)         # tool layer llama3_apibank
                para_origin = torch.tensor(para_origin)
                vector_tool = ((para - para_origin)) # .mean(0)
                    
                assert True not in (vector_tool == 0)
                if type_module in name:
                    l = int(name.split(".")[2])
                    results[l] = vector_tool
                
                    
    name_dict_other={"code":5, "math":6, "IF":7, "knowledge":8, "tool":9}
    for k, v in fea_funtune_constrast.items():
        print(k)
        for name, para in v.items():
            para_origin = fea_origin[k][name]
            para = torch.tensor(para)
            para_origin = torch.tensor(para_origin)
            vector = ((para - para_origin)) # .mean(0)
            
            assert True not in (vector==0)
            if type_module in name:
                l = int(name.split(".")[2])
                r = cosine_similarity(results[l].numpy(), vector.numpy())
                result_meta[name_dict_other[k]][l] = float(r)
                
    name_dict={"code":0, "math":1, "IF":2, "knowledge":3, "tool":4} # {"math":0, "code":1, "conversation":2, "tool":3}
    # calculate the increament of the vector
    for k, v in fea_funtune.items():
        print(k)
        for name, para in v.items():
            para_origin = fea_origin[k][name]
            para = torch.tensor(para)
            para_origin = torch.tensor(para_origin)
            vector = ((para - para_origin)) # .mean(0)
            
            assert True not in (vector==0)
            if type_module in name:
                l = int(name.split(".")[2])
                r = cosine_similarity(results[l].numpy(), vector.numpy())
                result_meta[name_dict[k]][l] = float(r)
            
    

    # results = result
    results_plot = result_meta
    print(results_plot)
    data_type, layers = results_plot.shape
    min_, max_ = torch.min(results_plot).item(), torch.max(results_plot).item() 
    x = list(range(0,32))
    DATASET_TO_OFFICIAL = {0:"HE - Tool",1:"GSM - Tool",2:"MT - Tool",3:"TQA - Tool", 4:"Tool - Tool",
                        5:"HE* - Tool",6:"GSM* - Tool",7:"MT* - Tool",8:"TQA* - Tool",9:"Tool* - Tool"}
    DATASET_TO_COLOR = {0:"royalblue", 1:"lime",2:"r",3:"cyan",4: 'fuchsia',5: 'darkorange',6: 'pink',7: 'black',8: "green", 9 :"purple"}
    for i,res in enumerate(results_plot):
        if "*" in DATASET_TO_OFFICIAL[i]:
            plt.plot(x, list(res.numpy()), label = DATASET_TO_OFFICIAL[i], alpha = 0.6, color=DATASET_TO_COLOR[i], linestyle='--')
        else:
            plt.plot(x, list(res.numpy()), label = DATASET_TO_OFFICIAL[i], alpha = 0.6, color=DATASET_TO_COLOR[i])


    plt.title(f'The Similarity of Incremental Change of Capabilities') 
    plt.xlabel("Layers") 
    plt.ylabel("Cosine Similarity") 
    plt.legend(loc="upper right")

    import os
    plt.savefig("./output/" + type_module + ".pdf")
    plt.close()

    

if __name__ == "__main__":
    draw("mlp")
    draw("attn")