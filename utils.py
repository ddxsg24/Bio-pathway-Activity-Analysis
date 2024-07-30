import os.path
import pandas as pd

import torch
from torch_geometric.data import Data, HeteroData

import numpy as np
import random
def load_multi_nextwork(multi_network_file_path, multi_network_name):
    multinetwork = {}

    for network in multi_network_name:
        network_path = os.path.join(multi_network_file_path, f"{network}.txt")
        multinetwork[network] = pd.read_csv(network_path, sep='\t', header=None).iloc[:, 0:2]

    return multinetwork

def load_kegg(kegg_file_path, kegg_name):
    kegg_path = os.path.join(kegg_file_path, f"{kegg_name}.txt")
    kegg = pd.read_csv(kegg_path, sep=' ', header=None)

    return kegg


def load_heterodata(kegg, multi_network): #kegg传入单个文件
    mapping = {}
    network_gene_set = []
    for network in multi_network:
        network_gene_set += list(set(multi_network[network][0]).union(set(multi_network[network][1])))
    network_gene_set = list(set(network_gene_set))

    kegg_gene_set = list(set(kegg[0]).union(set(kegg[1])))

    all_genes = list(set(network_gene_set + kegg_gene_set))

    for index, gene in enumerate(all_genes):
        mapping[gene] = index
    mapping_df = pd.DataFrame(list(mapping.items()), columns=['gene', 'index'])

    data = HeteroData()
    data['gene'].x = torch.ones(len(all_genes), 10)
    #接下来将所有网络的gene映射为index
    for network in multi_network:
        multi_network[network] = multi_network[network].merge(mapping_df, left_on=0, right_on='gene', how='left')
        multi_network[network] = multi_network[network].merge(mapping_df, left_on=1, right_on='gene', how='left')

        edge_index = torch.tensor([multi_network[network]['index_x'], multi_network[network]['index_y']], dtype=torch.long)
        data['gene', network, 'gene'].edge_index = edge_index

    kegg = kegg.merge(mapping_df, left_on=0, right_on='gene', how='left')
    kegg = kegg.merge(mapping_df, left_on=1, right_on='gene', how='left')
    edge_index = torch.tensor([kegg['index_x'], kegg['index_y']], dtype=torch.long)
    data['gene', 'kegg', 'gene'].edge_index = edge_index

    data.x = torch.ones(len(all_genes))
    data.mapping = mapping_df

    return data, kegg_gene_set

def get_kegg_name_list(keggs_file_dir='data/KEGGPathways'):
    #通过路径keggs_file_dir，遍历得到所有的kegg文件名
    kegg_name_list = []
    for root, dirs, files in os.walk(keggs_file_dir):
        for file in files:
            kegg_name_list.append(file.split('.')[0])

    return kegg_name_list

def seed_setting(seed_number):
    random.seed(seed_number)
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False