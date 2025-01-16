import dgl
import copy
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
from dgl import DropNode, DropEdge, RemoveSelfLoop

#def ShowGraph(graph, nodeLabel, EdgeLabel):
def ShowGraph(graph, nodeLabel):
    plt.figure(figsize=(8, 8))
    #G = graph.to_networkx(node_attrs=nodeLabel.split(), edge_attrs=EdgeLabel.split()) #转换dgl graph to networks
    G = graph.to_networkx(node_attrs=nodeLabel.split())
    pos = nx.spring_layout(G)
    nx.draw(G, pos, edge_color="black", node_size=500, with_labels=True) #画图，设置节点大小

    node_data = nx.get_node_attributes(G, nodeLabel)
    node_labels = {index: "N:" + str(data) for index, data in
                   enumerate(node_data)}
    pos_higher = {}

    for k, v in pos.items():
        if (v[1] > 0):
            pos_higher[k] = (v[0]-0.04, v[1]+0.04)
        else:
            pos_higher[k] = (v[0]-0.04, v[1]-0.04)
    nx.draw_networkx_labels(G, pos_higher, labels=node_labels, font_color="brown", font_size=12)


    '''edge_labels = nx.get_edge_attributes(G, EdgeLabel)

    edge_labels = {(key[0], key[1]): "w:" + str(edge_labels[key].item()) for key in
                   edge_labels}
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

    print(G.edges.data())'''
    plt.show()

#detect_degree(list for graphs)
def detect_degree(graphs):
    for i in range(len(graphs)):

        buf = []
        for node in graphs[i].nodes():
            #undirected graph: in_degrees=out_degrees
            #if graphs[i].in_degrees(node) == 0:
            if graphs[i].in_degrees(node)==0 and graphs[i].out_degrees(node)==0:
               buf.append(int(node))
        if buf != []:
            #print(buf)
            graphs[i] = dgl.remove_nodes(graphs[i], th.tensor(buf))

'''
#detect_graph(list for graphs)
def detect_graph(graphs):
    dgl_graphs = []

    for i in range(len(graphs)):
        nx_graph = graphs[i].to_networkx(node_attrs=['x'], edge_attrs=['w']).to_undirected()

        subgraphs = []
        for nodes in nx.connected_components(nx_graph):
            subgraph = nx_graph.subgraph(nodes)
            if len(subgraph)==1:
                print(i)
                print(subgraph)

            subgraphs.append(subgraph)

        large_subgraphs = [subgraph for subgraph in subgraphs if len(subgraph)>2]
        merged_graph = nx.compose_all(large_subgraphs)
        dgl_graph = dgl.from_networkx(merged_graph)
        dgl_graphs.append(dgl_graph)

    return dgl_graphs
'''

'''
from dgl import save_graphs
edges = [(0, 1), (1, 2), (3, 4)]
g = dgl.graph(edges, num_nodes=5)
g.ndata['label'] = th.tensor([0, 1, 0, 1, 1])
g.ndata['h'] = th.randn((5, 6))

nx_graph = g.to_networkx(node_attrs=['h', 'label']).to_undirected()  #
print(nx_graph)

subgraphs = []
for nodes in nx.connected_components(nx_graph):
    subgraph = nx_graph.subgraph(nodes)
    subgraphs.append(subgraph)

large_subgraphs = [subgraph for subgraph in subgraphs if len(subgraph) > 2]
merged_graph = nx.compose_all(large_subgraphs)
print(merged_graph)

dgl_graph = dgl.from_networkx(merged_graph, node_attrs=['h', 'label'])  #
print(dgl_graph)
save_graphs('/your_path/graph.name.bin', dgl_graph)
'''

#detach features, e_features, e_adj, adj, t_mat from drug graph
'''def BDgraph_info(batch_graph):
    inc_mat = batch_graph.incidence_matrix(typestr='both')
    t_mat = th.abs(inc_mat)

    adj = batch_graph.adj()
    features = batch_graph.ndata['x']

    #convert to line graph
    line_graphs = dgl.line_graph(batch_graph)
    e_adj = line_graphs.adj()
    e_features = batch_graph.edata['w']

    return features, e_features, e_adj, adj, t_mat
'''

'''For GPU
#detach features, e_features, e_adj, adj, t_mat
def BDgraph_info(batch_graph):
    inc_mat = batch_graph.incidence_matrix(typestr='both')
    dense_inc_mat = inc_mat.to_dense()
    t_mat = th.abs(dense_inc_mat)
    sparse_t_mat = dense_to_coo(t_mat)

    adj = batch_graph.adj()
    features = batch_graph.ndata['x']

    #convert to line graph
    line_graphs = dgl.line_graph(batch_graph)
    e_adj = line_graphs.adj()
    e_features = batch_graph.edata['w']

    return features, e_features, e_adj, adj, 
'''

#readout features in terms of a PROTAC graph
def BPgraph_readout(batch_size, batch_graph, batch_features):
    tensor_list = []
    #set the size of last batch
    batch_size_buf = batch_size
    if (batch_graph.batch_size%batch_size) != 0:
        batch_size_buf = batch_graph.batch_size

    for i in range(batch_size_buf):
        '''print('\n')'''
        '''print("i: {}".format(i))'''
        start_row = 0
        #when i=0, unable to excute 'for j in range(i)'
        #'for j in range(i)': get start row
        for j in range(i):
            '''print("j: {}".format(j))'''
            row_num = dgl.slice_batch(batch_graph, j).num_nodes()
            start_row = start_row + row_num

        ith_rowNum = dgl.slice_batch(batch_graph, i).num_nodes()
        '''print("rowNum: {}".format(ith_rowNum))'''
        ith_graph = batch_features[start_row:(start_row+ith_rowNum), :]
        '''print("rowNum: {}".format(ith_graph.size()))'''
        #dim=0, calculate mean in terms of column
        readout_ith_Graph = th.mean(ith_graph, dim=0)
        tensor_list.append(readout_ith_Graph)

    tensor_graphs = th.stack(tensor_list)
    return tensor_graphs

#graph augmentations
def graph_augmentation(type_t, type_p, tgraphs, pgraphs, lgraphs):
    tls_aug1 = []
    tls_aug2 = []
    protacs_aug1 = []
    protacs_aug2 = []
    #augmentation methods
    transformDN = DropNode(0.2)
    transformDE = DropEdge(0.2)
    #augmentation for protein graphs
    tl_graphs = tgraphs + lgraphs
    tlgraphs_copy1 = copy.deepcopy(tl_graphs)
    tlgraphs_copy2 = copy.deepcopy(tl_graphs)
    for i in range(len(tl_graphs)):
        if type_t == 'DN':
            tls_aug1.append(transformDN(tlgraphs_copy1[i]))
            tls_aug2.append(transformDN(tlgraphs_copy2[i]))
        elif type_t == 'DE':
            tls_aug1.append(transformDE(tlgraphs_copy1[i]))
            tls_aug2.append(transformDE(tlgraphs_copy2[i]))
        elif type_t == 'RW':
            t_traces1, t_types1 = dgl.sampling.random_walk(tlgraphs_copy1[i], [0,1,2,3,4], length=20)
            t_traces2, t_types2 = dgl.sampling.random_walk(tlgraphs_copy2[i], [5,6,7,8,9], length=20)
            tconcat_vids1, _, _, _ = dgl.sampling.pack_traces(t_traces1, t_types1)
            tconcat_vids2, _, _, _ = dgl.sampling.pack_traces(t_traces2, t_types2)
            tsubgraph1 = dgl.node_subgraph(tlgraphs_copy1[i], tconcat_vids1, relabel_nodes=True, store_ids=False)
            tsubgraph2 = dgl.node_subgraph(tlgraphs_copy2[i], tconcat_vids2, relabel_nodes=True, store_ids=False)
            RselfLoop = RemoveSelfLoop()
            new_tsubgraph1 = RselfLoop(tsubgraph1)
            new_tsubgraph2 = RselfLoop(tsubgraph2)
            tls_aug1.append(new_tsubgraph1)
            tls_aug2.append(new_tsubgraph2)
        elif type_t == 'KH':
            tkh_subgraph1, _ = dgl.khop_in_subgraph(tlgraphs_copy1[i], 0, k=5)
            tkh_subgraph2, _ = dgl.khop_in_subgraph(tlgraphs_copy2[i], 5, k=4)
            tls_aug1.append(tkh_subgraph1)
            tls_aug2.append(tkh_subgraph2)
        else:
            print('please input augmentation type')

    #augmentation for protac graphs
    pgraphs_copy1 = copy.deepcopy(pgraphs)
    pgraphs_copy2 = copy.deepcopy(pgraphs)
    for j in range(len(pgraphs)):
        if type_p == 'DN':
            protacs_aug1.append(transformDN(pgraphs_copy1[j]))
            protacs_aug2.append(transformDN(pgraphs_copy2[j]))
        elif type_p == 'DE':
            protacs_aug1.append(transformDE(pgraphs_copy1[j]))
            protacs_aug2.append(transformDE(pgraphs_copy2[j]))
        elif type_p == 'RW':
            p_traces1, p_types1 = dgl.sampling.random_walk(pgraphs_copy1[j], [0,1,2,3,4], length=10)
            p_traces2, p_types2 = dgl.sampling.random_walk(pgraphs_copy2[j], [5,6,7,8,9], length=10)
            pconcat_vids1, _, _, _ = dgl.sampling.pack_traces(p_traces1, p_types1)
            pconcat_vids2, _, _, _ = dgl.sampling.pack_traces(p_traces2, p_types2)
            psubgraph1 = dgl.node_subgraph(pgraphs_copy1[j], pconcat_vids1, relabel_nodes=True, store_ids=False)
            psubgraph2 = dgl.node_subgraph(pgraphs_copy2[j], pconcat_vids2, relabel_nodes=True, store_ids=False)
            RselfLoop = RemoveSelfLoop()
            new_psubgraph1 = RselfLoop(psubgraph1)
            new_psubgraph2 = RselfLoop(psubgraph2)
            protacs_aug1.append(new_psubgraph1)
            protacs_aug2.append(new_psubgraph2)
        elif type_p == 'KH':
            pkh_subgraph1, _ = dgl.khop_in_subgraph(pgraphs_copy1[j], 0, k=10)
            pkh_subgraph2, _ = dgl.khop_in_subgraph(pgraphs_copy2[j], 5, k=9)
            protacs_aug1.append(pkh_subgraph1)
            protacs_aug2.append(pkh_subgraph2)
        else:
            print('please input augmentation type')

    #delete 0-degree nodes for graphs
    detect_degree(tls_aug1)
    detect_degree(tls_aug2)
    detect_degree(protacs_aug1)
    detect_degree(protacs_aug2)

    print(tl_graphs[0])
    print(tls_aug1[0])
    print(tls_aug2[0])
    print('##############################################')
    print(pgraphs[0])
    print(protacs_aug1[0])
    print(protacs_aug2[0])

    ShowGraph(tls_aug1[0],"x")

    return tls_aug1, tls_aug2, protacs_aug1, protacs_aug2
