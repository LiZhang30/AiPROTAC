import dgl
import torch as th
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DatasetIterater(Data.Dataset):
    def __init__(self, texta, textb, textc, textd, label):
        self.texta = texta
        self.textb = textb
        self.textc = textc
        self.textd = textd
        self.label = label

    def __getitem__(self, item):
        return self.texta[item], self.textb[item], self.textc[item], self.textd[item], self.label[item]

    def __len__(self):
        return len(self.texta)


class DatasetIterater1(Data.Dataset):
    def __init__(self, texta):
        self.texta = texta

    def __getitem__(self, item):
        return self.texta[item]

    def __len__(self):
        return len(self.texta)


def Gcollate(samples):
    tgraphs, wgraphs, ldgraphs, legraphs, labels = map(list, zip(*samples))

    bd_tgraph = dgl.batch(tgraphs)
    bd_tgraph.ndata['x'] = th.as_tensor(bd_tgraph.ndata['x'], dtype=th.float32)

    bd_wgraph= dgl.batch(wgraphs)
    bd_wgraph.ndata['x'] = th.as_tensor(bd_wgraph.ndata['x'], dtype=th.float32)

    bd_ldgraph = dgl.batch(ldgraphs)
    bd_ldgraph.ndata['x'] = th.as_tensor(bd_ldgraph.ndata['x'], dtype=th.float32)

    bd_legraph = dgl.batch(legraphs)
    bd_legraph.ndata['x'] = th.as_tensor(bd_legraph.ndata['x'], dtype=th.float32)

    bd_labels = th.as_tensor(labels, dtype=th.long)
    return bd_tgraph, bd_wgraph, bd_ldgraph, bd_legraph, bd_labels


def BatchPad(batch_data):
    texta = list(batch_data)
    #max_len_a = max([len(seq_a) for seq_a in texta])
    #texta = [seq+[pad]*(max_len_a-len(seq)) for seq in texta]
    texta = th.LongTensor(texta)
    return texta


class GraphConv(nn.Module):
    def __init__(self, flag):
        super().__init__()
        if flag:
            self.layer1 = dglnn.GraphConv(69, 128, bias=False)
        else:
            self.layer1 = dglnn.GraphConv(33, 128, bias=False)
        self.layer2 = dglnn.GraphConv(128, 64, bias=False)

    def forward(self, graph):
        x1 = graph.ndata['x']

        x2 = self.layer1(graph, x1)
        x2 = F.relu(x2, inplace=True)

        x3 = self.layer2(graph, x2)
        x3 = F.relu(x3, inplace=True)

        with graph.local_scope():
            graph.ndata['x'] = x3
            readout = dgl.max_nodes(graph, 'x')

            return readout


class SmilesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(41, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        s = [len(x[0])]
        x = self.embed(x)
        x = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(x, None)
        out, _ = pad_packed_sequence(out, batch_first=True)
        y = self.fc(out[:, -1, :])
        return y


class ProtacModel(nn.Module):
    def __init__(self,
                 ligase_ligand_model,
                 ligase_pocket_model,
                 target_ligand_model,
                 target_pocket_model,
                 smiles_model):
        super().__init__()
        self.ligase_ligand_model = ligase_ligand_model
        self.ligase_pocket_model = ligase_pocket_model
        self.target_ligand_model = target_ligand_model
        self.target_pocket_model = target_pocket_model
        self.smiles_model = smiles_model
        self.fc1 = nn.Linear(64*5, 64)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(64, 2)

    def forward(self,
                ligase_ligand,
                ligase_pocket,
                target_ligand,
                target_pocket,
                smiles):
        v_0 = self.ligase_ligand_model(ligase_ligand)
        v_1 = self.ligase_pocket_model(ligase_pocket)
        v_2 = self.target_ligand_model(target_ligand)
        v_3 = self.target_pocket_model(target_pocket)
        v_4 = self.smiles_model(smiles)
        v_f = th.cat((v_0, v_1, v_2, v_3, v_4), 1)
        v_f = self.relu(self.fc1(v_f))
        v_f = self.fc2(v_f)
        return v_f