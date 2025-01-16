import dgl
import numpy as np
import torch as th
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import torch.utils.data as Data
from model.CensNet_model import CensNet
from utils.DglOperator import BPgraph_readout
#transformer parameters
d_model = 128 #embedding size
d_ff = 512 #feedforward dimension
d_k = d_v = 32 #dimension of K(=Q), V
n_layers = 1 #number of encoder
n_heads = 4 #number of heads in multi-head attention

class SSL_DatasetIterater(Data.Dataset):
    def __init__(self, texta, textb):
        self.texta = texta
        self.textb = textb

    def __getitem__(self, item):
        return self.texta[item], self.textb[item]

    def __len__(self):
        return len(self.texta)


def SSL_Gcollate(samples):
    tlgraphs, pgraphs = map(list, zip(*samples))

    batched_tlgraph = dgl.batch(tlgraphs)
    batched_tlgraph.ndata['x'] = th.as_tensor(batched_tlgraph.ndata['x'], dtype=th.float32)

    batched_pgraph = dgl.batch(pgraphs)
    batched_pgraph.ndata['x'] = th.as_tensor(batched_pgraph.ndata['x'], dtype=th.float32)
    batched_pgraph.edata['w'] = th.as_tensor(batched_pgraph.edata['w'], dtype=th.float32)

    return batched_tlgraph, batched_pgraph


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


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        #Q: [batch_size, n_heads, len_q, d_k]
        #K: [batch_size, n_heads, len_k, d_k]
        #V: [batch_size, n_heads, len_v(=len_k), d_v]
        #attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = th.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  #scores:[batch_size, n_heads, len_q, len_k]

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  #Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = th.matmul(attn, V)  #[batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.fc0 = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v*n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads*d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        #input_Q: [batch_size, len_q, d_model]
        #input_K: [batch_size, len_k, d_model]
        #input_V: [batch_size, len_v(=len_k), d_model]
        #attn_mask: [batch_size, seq_len, seq_len]

        #batch_size, seq_len, model_len = input_Q.size()
        if attn_mask is not None:
            batch_size, seq_len, model_len = input_Q.size()
            residual_2D = input_Q.view(batch_size*seq_len, model_len)
            residual = self.fc0(residual_2D).view(batch_size, seq_len, model_len)
        else:
            residual, batch_size = input_Q, input_Q.size(0)

        #(B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  #Q:[bs, heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  #K:[bs, heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  #V:[bs, heads, len_v(=len_k), d_v]

        if attn_mask is not None:
            #attn_mask:[batch_size, n_heads, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        #context:[batch_size, n_heads, len_q, d_v]
        #attn:[batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads*d_v)  #context:[bs, len_q, heads*d_v]
        output = self.fc(context)  #[batch_size, len_q, d_model]
        output = th.squeeze(output)  #delete dimension=1
        '''return nn.LayerNorm(d_model).to('cuda:0')(output + residual), attn'''
        return nn.LayerNorm(d_model)(output+residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        #inputs:[batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        '''return nn.LayerNorm(d_model).to('cuda:0')(output + residual) #[batch_size, seq_len, d_model]'''
        return nn.LayerNorm(d_model)(output+residual)  #[batch_size, seq_len, d_model]


class CrossAttention_EncoderLayer(nn.Module):
    def __init__(self):
        super(CrossAttention_EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, inputs_Q, inputs_K, inputs_V, enc_self_attn_mask):
        #enc_inputs:[batch_size, src_len, d_model]
        #enc_self_attn_mask:[batch_size, src_len, src_len]

        #enc_outputs:[batch_size, src_len, d_model]
        #attn:[batch_size, n_heads, src_len, src_len]
        #enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(inputs_Q, inputs_K, inputs_V, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  #enc_outputs:[batch_size, src_len, d_model]
        return enc_outputs, attn


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.layer1 = dglnn.GraphConv(in_dim, hidden_dim*4, bias=False, allow_zero_in_degree=True)
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim*4, bias=False),
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(inplace=True),
        )

        self.layer2 = dglnn.GraphConv(hidden_dim*8, hidden_dim*4, bias=False, allow_zero_in_degree=True)
        self.layer3 = dglnn.GraphConv(hidden_dim*4, out_dim, bias=False, allow_zero_in_degree=True)

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

        self.pp_cross_atten = CrossAttention_EncoderLayer()
        self.cc_cross_atten = CrossAttention_EncoderLayer()

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
            nn.Linear(self.hid_dim*2, self.hid_dim*8, bias=False),
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

        #interatcion between PROTACs and proteins
        Inner_outPL, _ = self.pp_cross_atten(Glig, read_BGpro, read_BGpro, None)
        Inner_outPT, _ = self.pp_cross_atten(Gtar, read_BGpro, read_BGpro, None)
        #interatcion between two complexs
        LPT_out, _ = self.cc_cross_atten(Inner_outPL, Inner_outPT, Inner_outPT, None)
        TPL_out, _ = self.cc_cross_atten(Inner_outPT, Inner_outPL, Inner_outPL, None)

        #linear projection
        dout = self.projection(LPT_out)
        tout = self.projection(TPL_out)
        DT_out = th.cat((dout, tout), 1)
        #classifier
        affi = self.reg_fun(DT_out)

        if kwargs['flag'] is not None:
            'GNN encoder for SSL'
            #ssl1_aug1
            SSL1_aug1_Gtl = self.ew_gcn(kwargs['self1Aug1_tlg'], kwargs['self1Aug1_tlx'], kwargs['self1Aug1_tlw'])
            #ssl1_aug2
            SSL1_aug2_Gtl = self.ew_gcn(kwargs['self1Aug2_tlg'], kwargs['self1Aug2_tlx'], kwargs['self1Aug2_tlw'])

            #ssl2_aug1
            SSL2_aug1_Gpro = self.pCensNet(kwargs['self2Aug1_p_features'], kwargs['self2Aug1_pe_features'],
                                           kwargs['self2Aug1_pe_adj'], kwargs['self2Aug1_p_adj'],
                                           kwargs['self2Aug1_p_mat'])
            SSL2_readAug1_BGpro = BPgraph_readout(kwargs['ssl2_bs'], kwargs['self2Aug1_pg'], SSL2_aug1_Gpro)
            #ssl2_aug2
            SSL2_aug2_Gpro = self.pCensNet(kwargs['self2Aug2_p_features'], kwargs['self2Aug2_pe_features'],
                                           kwargs['self2Aug2_pe_adj'], kwargs['self2Aug2_p_adj'],
                                           kwargs['self2Aug2_p_mat'])
            SSL2_readAug2_BGpro = BPgraph_readout(kwargs['ssl2_bs'], kwargs['self2Aug2_pg'], SSL2_aug2_Gpro)

            return affi, SSL1_aug1_Gtl, SSL1_aug2_Gtl, SSL2_readAug1_BGpro, SSL2_readAug2_BGpro

        else:
            return affi