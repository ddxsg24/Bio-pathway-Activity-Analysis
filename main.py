import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from utils import load_multi_nextwork, load_kegg, load_heterodata, get_kegg_name_list
from utils import seed_setting

from model import DMGI

import torch
from train_model import train_embedding, get_embedding
import torch.nn.functional as F

def calculate(kegg_name, dir_path='EnhancedPathway'):
    seed = 2024
    seed_setting(seed)

    multi_network_file_path = 'data/MultiSourceData'
    multi_network_name = ['Complexes', 'Kinase', 'Metabolic', 'ppi', 'Regulatory']

    kegg_file_path = 'data/KEGGPathways'

    multi_network = load_multi_nextwork(multi_network_file_path, multi_network_name)
    kegg = load_kegg(kegg_file_path, kegg_name)

    data, kegg_gene_set = load_heterodata(kegg, multi_network)

    node_types, edge_types = data.metadata()
    edge_types_num = len(edge_types)

    model_DMGI = DMGI(data['gene'].num_nodes, data['gene'].x.size(-1),
                      out_channels=128,
                      num_relations=edge_types_num)  # num_relations 这里要node_types, edge_types = data.metadata(),len(len(edge_types))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, model_DMGI = data.to(device), model_DMGI.to(device) #数据、模型都需要放到GPU
    optimizer_DMGI = torch.optim.RMSprop(model_DMGI.parameters(), lr=0.001, weight_decay=0.01)

    net_types = []
    for relation in edge_types:
        net_types.append(relation[1])
    # 训练模型
    print("Training...")
    train_embedding(model_DMGI, optimizer_DMGI, data, net_types)
    # 获取嵌入
    embedding = get_embedding(model_DMGI)

    kegg_temp = pd.DataFrame(kegg_gene_set, columns=['gene'])
    kegg_index_set = kegg_temp.merge(data.mapping, left_on='gene', right_on='gene', how='left')['index']

    kegg_embedding = embedding[kegg_index_set]

    print("Calculating similarity matrix...")
    similarity_matrix = F.cosine_similarity(kegg_embedding.unsqueeze(1), kegg_embedding.unsqueeze(0), dim=-1)
    similarity_matrix_np = similarity_matrix.cpu().numpy()
    similarity_matrix_df = pd.DataFrame(similarity_matrix_np, columns=kegg_gene_set, index=kegg_gene_set)

    print("Calculating k-nearest neighbors...")
    n_samples = similarity_matrix_df.shape[0]
    k = min(11, n_samples) # Number of neighbors，k必须比设定的多1，因为后续NearestNeighbors会把自己也算进去
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(similarity_matrix_df.values)

    # 根据嵌入结果增强网络
    distances, indices = nbrs.kneighbors(similarity_matrix_df.values)

    edgelist = []
    for node, neighbors in zip(similarity_matrix_df.index, indices):
        for neighbor in neighbors:
            if node == similarity_matrix_df.index[neighbor]:
                continue
            elif (similarity_matrix_df.index[neighbor], node) in edgelist:
                continue
            else:
                edgelist.append((node, similarity_matrix_df.index[neighbor]))

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    edgelist_df = pd.DataFrame(edgelist)
    enhanced_pathway_file_path = os.path.join(dir_path, f'enhanced_{kegg_name}.txt')
    edgelist_df.to_csv(enhanced_pathway_file_path, index=False, header=False)
    print("Edgelist saved")

if __name__ == '__main__':
    kegg_name_list = get_kegg_name_list()
    print(f"kegg_num: {len(kegg_name_list)}")
    for kegg_name in kegg_name_list:
        print("Calculating kegg: ", kegg_name)
        calculate(kegg_name=kegg_name)
        print("\nAll Done!\n")