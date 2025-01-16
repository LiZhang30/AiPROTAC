import dgl
import torch as th
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import torch.utils.data as Data
from model.CensNet_model import CensNet
from utils.DglOperator import BPgraph_readout

class DatasetIterater(Data.Dataset):
    def __init__(self, texta, textb, textc, label):
        self.texta = texta
        self.textb = textb
        self.textc = textc
        self.label = label

    def __getitem__(self, item):
        return self.texta[item], self.textb[item], self.textc[item], self.label[item]

    def __len__(self):
        return len(self.texta)


def Gcollate(samples):
    tgraphs, pgraphs, lgraphs, labels = map(list, zip(*samples))

    batched_tgraph = dgl.batch(tgraphs)
    batched_tgraph.ndata['x'] = th.as_tensor(batched_tgraph.ndata['x'], dtype=th.float32)

    batched_pgraph = dgl.batch(pgraphs)
    batched_pgraph.ndata['x'] = th.as_tensor(batched_pgraph.ndata['x'], dtype=th.float32)
    batched_pgraph.edata['w'] = th.as_tensor(batched_pgraph.edata['w'], dtype=th.float32)

    batched_lgraph = dgl.batch(lgraphs)
    batched_lgraph.ndata['x'] = th.as_tensor(batched_lgraph.ndata['x'], dtype=th.float32)

    batched_labels = th.as_tensor(labels, dtype=th.long)
    return batched_tgraph, batched_pgraph, batched_lgraph, batched_labels


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.layer1 = dglnn.GraphConv(in_dim, hidden_dim*4, bias=False)
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim*4, bias=False),
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(inplace=True),
        )

        self.layer2 = dglnn.GraphConv(hidden_dim*8, hidden_dim*4, bias=False)
        self.layer3 = dglnn.GraphConv(hidden_dim*4, out_dim, bias=False)

    def forward(self, graph, x, w):
        x1 = self.layer1(graph, x, edge_weight=w)
        x1 = F.relu(x1, inplace=True)
        f1 = self.fc1(x)
        x1f1 = th.cat((x1, f1), 1)

        x2 = self.layer2(graph, x1f1, edge_weight=w)
        x2 = F.relu(x2, inplace=True)

        x3 = self.layer3(graph, x2, edge_weight=w)
        x3 = F.relu(x3, inplace=True)

        with graph.local_scope():
            graph.ndata['x'] = x3
            readout = dgl.sum_nodes(graph, 'x')

            return readout


class GraphBasedModel(nn.Module):
    def __init__(self, model_config):
        super(GraphBasedModel, self).__init__()
        #model config
        self.model_config = model_config
        self.pcn_dim = model_config['PROTAC_node_dim']
        self.pce_dim = model_config['PROTAC_edge_dim']
        self.pnn_dim = model_config['protein_node_dim']
        self.pne_dim = model_config['protein_edge_dim']
        self.hid_dim = model_config['node_hidden_dim']
        self.out_dim = model_config['node_out_dim']
        self.dp_ratio = model_config['dropout_ratio']
        self.cf_dim = model_config['classifier_dim']

        #model layers
        self.pCensNet = CensNet(self.pcn_dim, self.pce_dim, self.hid_dim, self.out_dim, self.dp_ratio)
        self.ew_gcn = GCN(self.pnn_dim, self.hid_dim, self.out_dim)
        ####self.lgcn = GCN(self.pnn_dim, self.hid_dim, self.out_dim)
        ####self.tgcn = GCN(self.pnn_dim, self.hid_dim, self.out_dim)

        #share weights
        self.projection = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim*4, bias=False),
            nn.LayerNorm(self.hid_dim*4),
            nn.Dropout(self.dp_ratio),
            nn.ReLU(inplace=True),

            nn.Linear(self.hid_dim*4, self.hid_dim, bias=False),
        )

        #classifier
        self.reg_fun = nn.Sequential(
            nn.Linear(self.hid_dim*3, self.hid_dim*8, bias=False),
            nn.LayerNorm(self.hid_dim*8),
            nn.Dropout(self.dp_ratio),
            nn.ReLU(inplace=True),

            nn.Linear(self.hid_dim*8, self.hid_dim*2, bias=False),
            nn.LayerNorm(self.hid_dim*2),
            nn.Dropout(self.dp_ratio),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(self.hid_dim*2, self.cf_dim, bias=False)
        )

    def forward(self, **kwargs):
        #GNN encoder for SL
        ####Glig = self.lgcn(lg, lx, lw)
        ####Gtar = self.tgcn(tg, tx, tw)
        Glig = self.ew_gcn(kwargs['lg'], kwargs['lx'], kwargs['lw'])
        Gtar = self.ew_gcn(kwargs['tg'], kwargs['tx'], kwargs['tw'])
        Gpro = self.pCensNet(kwargs['p_features'], kwargs['pe_features'],
                                           kwargs['pe_adj'], kwargs['p_adj'], kwargs['p_mat'])
        read_BGpro = BPgraph_readout(kwargs['bs'], kwargs['pg'], Gpro)

        #linear projection
        tout = self.projection(Gtar)
        pout = self.projection(read_BGpro)
        lout = self.projection(Glig)
        TPL_out = th.cat((tout, pout, lout), 1)
        #classifier
        affi = self.reg_fun(TPL_out)

        return affi