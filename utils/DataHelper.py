import dgl
import numpy as np
import torch as th
import scipy.sparse as sp
from utils import DrugGraph as DG
from utils import TargetGraph as TG
from dgl.data.utils import save_graphs
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
#RDkit version no.
print(rdkit.__version__)

SMILES_CHAR =['[PAD]', 'C', '(', '=', 'O', ')', 'N', '[', '@', 'H', ']',
              '1', 'c', 'n', '/', '2', '#', 'S', 's', '+', '-', '\\', '3',
              '4', 'l', 'F', 'o', 'I', 'B', 'r', 'P', '5', '6', 'i', '7', '8',
              '9', '%', '0', 'p']
dru_dic = 40

#target and e3
proteinSeq_vocab = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
                   "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13,
                   "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19,
                   "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
pro_dic = 25

#Morgan/1024
def get_MorganFinger(protac_SMILES):
    protac_buf = []
    for i in range(len(protac_SMILES)):

        mol = Chem.MolFromSmiles(protac_SMILES[i])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_str = fp.ToBitString().split()
        buf = []
        for j in range(len(fp_str[0])):
            buf.append(int(fp_str[0][j]))
        protac_buf.append(buf)

    return protac_buf

#MACCS/167
def get_MaccsFinger(protac_SMILES):
    protac_buf = []
    for i in range(len(protac_SMILES)):

        mol = Chem.MolFromSmiles(protac_SMILES[i])
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_str = fp.ToBitString().split()
        buf = []
        for j in range(len(fp_str[0])):
            buf.append(int(fp_str[0][j]))
        protac_buf.append(buf)

    return protac_buf

#transfer token -> number
def LabelPro(protein_seqs):
    label_proteinSeqs = []
    for i in range(len(protein_seqs)):
        label_proteinSeqs.append([])
        for j in range(len(protein_seqs[i])):
            label_protein = proteinSeq_vocab[protein_seqs[i][j].split()[0]]
            label_proteinSeqs[i].append(label_protein)

    return label_proteinSeqs

#load id inforamtion
def LoadData(path):
    ldegradationID = np.loadtxt(path+'labeled degradation-ID.txt', dtype=int).tolist()

    return ldegradationID

#load id for DeepPROTACs
def LoadData1(path):
    ldegradationID = np.loadtxt(path+'PROTAC related to labeled samples.txt', dtype=int).tolist()

    return ldegradationID

#get samples
def GetSamples(id_info, targetGraphs, protacGraphs, ligaseGraphs):
    lgtarget_buf, lgprotac_buf, lgligase_bug, label_buf = [], [], [], []

    for i in range(len(id_info)):
        if (id_info[i][3]==0 or id_info[i][3]==1):

            lgtarget_buf.append(targetGraphs[id_info[i][1]-1])
            lgprotac_buf.append(protacGraphs[id_info[i][0]-1])
            lgligase_bug.append(ligaseGraphs[id_info[i][2]-1])
            label_buf.append(id_info[i][3])

    return lgtarget_buf, lgprotac_buf, lgligase_bug, label_buf

#get samples for DeepPROTACs
def GetSamples1(id_info, targetGraphs, protacSeq, warheadGraphs, linkerSeq, e3ligandGraphs, e3ligaseGraphs):
    target, warhead, linker, e3ligand, e3ligase, label = [], [], [], [], [], []

    for i in range(len(id_info)):
        #check if linker id is unknow
        if id_info[i][2] == 0:
            linker.append(protacSeq[id_info[i][0]-1])
        else:
            linker.append(linkerSeq[id_info[i][2]-1])

        target.append(targetGraphs[id_info[i][4]-1])
        warhead.append(warheadGraphs[id_info[i][1]-1])
        e3ligand.append(e3ligandGraphs[id_info[i][3]-1])
        e3ligase.append(e3ligaseGraphs[id_info[i][5]-1])
        label.append(id_info[i][6])

    return target, warhead, linker, e3ligand, e3ligase, label

#transfer smiles to integer
def trans_smiles(x):
    temp = list(x)
    temp = [SMILES_CHAR.index(i) if i in SMILES_CHAR else len(SMILES_CHAR) for i in temp]
    return temp

#shuttle
def Shuttle(target, protac, e3ligase, label, index):
    target = np.array(target, dtype=object)
    protac = np.array(protac, dtype=object)
    e3ligase = np.array(e3ligase, dtype=object)
    label = np.array(label, dtype=object)

    se_target = target[index]
    se_protac = protac[index]
    se_e3ligase = e3ligase[index]
    se_label = label[index]

    return se_target, se_protac, se_e3ligase, se_label

#shuttle for DeepPROTACs
def Shuttle1(target, warhead, linker, e3ligand, e3ligase, label, index):
    target = np.array(target, dtype=object)
    warhead = np.array(warhead, dtype=object)
    linker = np.array(linker, dtype=object)
    e3ligand = np.array(e3ligand, dtype=object)
    e3ligase = np.array(e3ligase, dtype=object)
    label = np.array(label, dtype=object)

    se_target = target[index]
    se_warhead = warhead[index]
    se_linker= linker[index]
    se_e3ligand = e3ligand[index]
    se_e3ligase = e3ligase[index]
    se_label = label[index]

    return se_target, se_warhead, se_linker, se_e3ligand, se_e3ligase, se_label

#row-normalize sparse matrix
def normalize(mx):
    #For GPU, mx_buff = mx.cpu().numpy()
    mx_buff = mx.numpy() #convert to numpy

    rowsum = np.array(mx_buff.sum(1)).astype("float")
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx_buff = r_mat_inv.dot(mx_buff)

    mx_buff = th.from_numpy(mx_buff).to(th.float32)
    return mx_buff #convert to tensor

#dense tensor to sparse tensor
def dense_to_coo(dense_tensor):
    dense_tensor_idx = th.nonzero(dense_tensor).T
    dense_tensor_data = dense_tensor[dense_tensor_idx[0], dense_tensor_idx[1]]
    coo_tensor = th.sparse_coo_tensor(dense_tensor_idx, dense_tensor_data, dense_tensor.size())

    return coo_tensor

#batch normalize features and adjacency matrix (node and edge)
def batch_normalize(batch_graph):
    #detach features, e_features, e_adj, adj, t_mat
    t_mat = th.abs(batch_graph.incidence_matrix(typestr='both'))
    adj = batch_graph.adj()
    features = batch_graph.ndata['x']
    #convert to line graph
    e_adj = dgl.line_graph(batch_graph).adj()
    e_features = batch_graph.edata['w']

    nor_features = normalize(features)
    nor_e_features = normalize(e_features)
    #adjacency matrix + identity matrix
    #nor_ai = normalize(th.eye(adj.size(0)) + adj)
    #nor_e_ai = normalize(th.eye(e_adj.size(0)) + e_adj)
    nor_adj = dense_to_coo(normalize(th.eye(adj.size(0))+adj))
    nor_e_adj = dense_to_coo(normalize(th.eye(e_adj.size(0))+e_adj))

    return nor_features, nor_e_features, nor_e_adj, nor_adj, t_mat


#split protac
##################################################################################
'''path = 'D:/code project/AiPROTACs/data/For DeepPROTACs/'
degradationID = np.loadtxt(path+'labeled degradation-ID.txt', dtype=int).tolist()
compositionID = np.loadtxt(path+'PROTAC composition-ID.txt', dtype=int).tolist()
warhead = np.loadtxt(path+'warhead.txt', dtype='str', comments='!').tolist()
linker = np.loadtxt(path+'linker.txt', dtype='str', comments='!').tolist()
e3ligand = np.loadtxt(path+'e3 ligand.txt', dtype='str', comments='!').tolist()
target_buf, ligase_bug, label_buf, protac_buf, warhead_buf, linker_buf, e3ligand_buf \
                                                          = [], [], [], [], [], [], []
for i in range(len(degradationID)):
    if (degradationID[i][3]==0 or degradationID[i][3]==1):
        target_buf.append(degradationID[i][1])
        ligase_bug.append(degradationID[i][2])
        label_buf.append(degradationID[i][3])
        protac_buf.append(degradationID[i][0])
        warhead_buf.append(compositionID[degradationID[i][0]-1][1])
        linker_buf.append(compositionID[degradationID[i][0]-1][2])
        e3ligand_buf.append(compositionID[degradationID[i][0]-1][3])

np.savetxt('./target_buf.csv', target_buf, delimiter=',')
np.savetxt('./ligase_bug.csv', ligase_bug, delimiter=',')
np.savetxt('./label_buf.csv', label_buf, delimiter=',')
np.savetxt('./protac_buf.csv', protac_buf, delimiter=',')
np.savetxt('./warhead_buf.csv', warhead_buf, delimiter=',')
np.savetxt('./linker_buf.csv', linker_buf, delimiter=',')
np.savetxt('./e3ligand_buf.csv', e3ligand_buf, delimiter=',')'''
##################################################################################
#construct graphs
#construct graphs for drugs
'''drug = np.loadtxt('D:/code project/AiPROTACs/data/wet-lab dataset/protac1.txt', dtype=str, comments='!').tolist()
save_graphs("D:/code project/AiPROTACs/ProtacGraphs.bin", [DG.SmileToGraph(smile) for smile in drug])'''
#construct graphs for targets
'''target = np.loadtxt('D:/code project/AiPROTACs/data/wet-lab dataset/e3 ligase.txt', dtype=str, comments='!').tolist()
target_graphs = []
for i in range(len(target)):
    target_map = np.loadtxt('../data/wet-lab dataset/target contact map 8A/{}.txt'.format(i), dtype=int)
    target_dis = np.loadtxt('../data/wet-lab dataset/target distance map/{}.txt'.format(i), dtype=float)
    target_ca = np.loadtxt('../data/wet-lab dataset/target ca coords/{}.txt'.format(i), dtype=float)
    target_seq3 = np.loadtxt('../data/wet-lab dataset/target seq for contact map/{}.txt'.format(i), dtype=str)

    print(i)
    print(target_map[i])
    print(target_dis[i])
    print(target_ca[i])
    print(target_seq3[i])
    target_graph = TG.TargetToGraph(target_map, target_dis, target_ca, target_seq3)

    #print(target_graph)
    target_graphs.append(target_graph)
save_graphs("D:/code project/AiPROTACs/TargetGraphs_8A.bin", target_graphs)'''