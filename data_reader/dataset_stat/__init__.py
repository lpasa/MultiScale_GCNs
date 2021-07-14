from torch_geometric.utils.convert import to_networkx
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.components import is_strongly_connected
from torch_geometric.datasets import TUDataset
from networkx.algorithms.distance_measures import diameter
import numpy as np
from torch_geometric.transforms import OneHotDegree
import torch


def get_stat(dataset,out_file=None):
    n_not_strong_connect=0
    n_pos=0
    n_neg=1
    n_nodes=[]
    n_edges=[]
    diameters=[]
    for graph in dataset:

        networkx_graph=to_networkx(graph)

        if not is_strongly_connected(networkx_graph):
            n_not_strong_connect+=1
        else:
            diameters.append(diameter(networkx_graph))

        # print(type(graph))
        if graph.y[0] == 1:
            n_pos += 1
        else:
            n_neg += 1

        n_nodes.append( graph.num_nodes)
        n_edges.append(graph.num_edges)
    print("---------------------------------",file=out_file)
    print(dataset.name, file=out_file)
    print("# graph: ",len(dataset), file=out_file)
    print("# not strong connected: ",n_not_strong_connect, file=out_file)
    print("# positive/negative graph: ",n_pos,"/",n_neg, file=out_file)
    print("average diameter: ",np.mean(np.asarray(diameters)), file=out_file)
    print("average # nodes",np.mean(np.asarray(n_nodes)) , file=out_file)
    print("average # edges: ",np.mean(np.asarray(n_edges)), file=out_file)
    print("dim input: ", dataset[0].x.shape[1], file=out_file)
    print("n_classes: ", max(dataset[0].y), file=out_file)
    print("---------------------------------",file=out_file)
    return len(dataset), n_not_strong_connect, n_pos, n_neg, np.mean(np.asarray(diameters)), np.mean(np.asarray(n_nodes)), np.mean(np.asarray(n_edges))


if __name__ == '__main__':
    f = open("dataset_stat.txt", "a")

    # dataset = TUDataset(root='~/Dataset/NCI1', name='NCI1')
    # n_graph, no_strong_conn, pos, neg, avg_diam, avg_nodes, avg_edges=get_stat(dataset)
    #
    #
    # dataset = TUDataset(root='~/Dataset/MUTAG', name='MUTAG')
    # n_graph, no_strong_conn, pos, neg, avg_diam, avg_nodes, avg_edges=get_stat(dataset,f)

    # dataset = TUDataset(root='~/Dataset/PROTEINS', name='PROTEINS')
    # dataset.data
    #
    # n_graph, no_strong_conn, pos, neg, avg_diam, avg_nodes, avg_edges = get_stat(dataset)

    # dataset = TUDataset(root='~/Dataset/Yeast', name='Yeast')
    # n_graph, no_strong_conn, pos, neg, avg_diam, avg_nodes, avg_edges = get_stat(dataset)
    #
    # dataset = TUDataset(root='~/Dataset/REDDIT-BINARY', name='REDDIT-BINARY')
    # n_graph, no_strong_conn, pos, neg, avg_diam, avg_nodes, avg_edges = get_stat(dataset)
    #
    # dataset = TUDataset(root='~/Dataset/DD', name='DD')
    # n_graph, no_strong_conn, pos, neg, avg_diam, avg_nodes, avg_edges = get_stat(dataset)
    #
    # dataset = TUDataset(root='/home/lpasa/Dataset/ENZYMES', name='ENZYMES',use_node_attr=True)
    # n_graph, no_strong_conn, pos, neg, avg_diam, avg_nodes, avg_edges=get_stat(dataset,f)

    dataset = TUDataset(root='~/Dataset/IMDB-BINARY/', name='IMDB-BINARY', transform=OneHotDegree(135))
    n_graph, no_strong_conn, pos, neg, avg_diam, avg_nodes, avg_edges=get_stat(dataset,f)

