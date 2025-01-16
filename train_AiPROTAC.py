import os
import copy
import json
import logging
import torch as th
import numpy as np
from sklearn import metrics
from datetime import datetime
from dgl.data.utils import load_graphs
from utils.CLloss import ContrastiveLoss
from dgl.dataloading import GraphDataLoader
from utils.DataHelper import LoadData, GetSamples
from utils.DataHelper import Shuttle, batch_normalize
from utils.DglOperator import detect_degree, ShowGraph, graph_augmentation
from cpu import ConfigArgumentParser, save_args, set_random_seed, setup_logger
from model.AiPROTAC import SSL_DatasetIterater, SSL_Gcollate, DatasetIterater, Gcollate, GraphBasedModel

logger = logging.getLogger(__name__)
def parse_args():
    parser = ConfigArgumentParser(description="Degradation")

    parser.add_argument("--log_dir", type=str, default="./log_out", help="Directory to save checkpoints and logs.")
    parser.add_argument("--work_dir", type=str, default="./", help="Directory to save runtime_config.yaml.")
    parser.add_argument("--data_dir", type=str, default="./data/PROTAC-DB 2.0/", help="Directory to save dataset.")

    parser.add_argument("--seed", type=int, default=1, help="Random seed, set to negative to randomize everything.")
    parser.add_argument("--deterministic", action="store_true", help="Turn on the CUDNN deterministic setting.")
    parser.add_argument("--ssl1_train_num", type=int, default=336, help="Number of (target+ligase).")
    parser.add_argument("--ssl2_train_num", type=int, default=3270, help="Number of protacs.")
    parser.add_argument("--train_num", type=int, default=829, help="Number of train set.")
    parser.add_argument("--test_num", type=int, default=207, help="Number of test set.")

    parser.add_argument("--log_iter", type=int, default=5, help="Interval for logging to console and tensorboard")
    parser.add_argument("--min_train_loss", type=float, default=1e6, help="Initial minimum epoch train loss")
    parser.add_argument("--ssl1_batch_size", type=int, default=16, help="Input batch size for training.")
    parser.add_argument("--ssl2_batch_size", type=int, default=32, help="Input batch size for training.")
    parser.add_argument("--temperature", type=float, default=0.05, help="temperature of contrastive loss.")

    parser.add_argument("--batch_size", type=int, default=32, help="Input batch size for training.")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to train on cuda:0.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial Learning rate")
    parser.add_argument("--grad_step", type=int, default=1, help="Gradient accumulate")
    parser.add_argument("--batch_num", type=int, default=26, help="Number of batch")
    #model config
    parser.add_argument("--model_config", type=str, default="./model/config.json", help="Initialize model parameters")

    return parser.parse_args()


def build_dataset(dir, train_num, test_num):
    targetGraphs, _ = load_graphs(dir+"TargetGraphs_8A.bin")
    protacGraphs, _ = load_graphs(dir+"ProtacGraphs.bin")
    ligaseGraphs, _ = load_graphs(dir+"LigaseGraphs_8A.bin")
    #delete 0-degree nodes for graphs
    detect_degree(targetGraphs)
    detect_degree(protacGraphs)
    detect_degree(ligaseGraphs)
    '''ShowGraph(targetGraphs[4],"x")'''
    #data preparation for SSL
    tls_aug1, tls_aug2, protacs_aug1, protacs_aug2 = graph_augmentation('KH', 'KH',
                                                                  targetGraphs, protacGraphs, ligaseGraphs)
    id_info = LoadData(dir)
    LGtargets, LGprotacs, LGligases, Labels = GetSamples(id_info, targetGraphs, protacGraphs, ligaseGraphs)
    #shuttle samples
    index = [i for i in range(len(Labels))]
    np.random.shuffle(index)
    SLGtargets, SLGprotacs, SLGligases, Slabels = Shuttle(LGtargets, LGprotacs, LGligases, Labels, index)
    SSL1_Gtrain_iter = SSL_DatasetIterater(tls_aug1[:], tls_aug2[:])
    SSL2_Gtrain_iter = SSL_DatasetIterater(protacs_aug1[:], protacs_aug2[:])
    Gtrain_iter = DatasetIterater(SLGtargets[0:train_num], SLGprotacs[0:train_num],
                                                                      SLGligases[0:train_num], Slabels[0:train_num])
    Gtest_iter = DatasetIterater(SLGtargets[train_num:train_num+test_num], SLGprotacs[train_num:train_num+test_num],
                                    SLGligases[train_num:train_num+test_num], Slabels[train_num:train_num+test_num])

    return SSL1_Gtrain_iter, SSL2_Gtrain_iter, Gtrain_iter, Gtest_iter


def build_dataloader(args):
    SSL1_Gtrain_iter, SSL2_Gtrain_iter, Gtrain_iter, Gtest_iter = build_dataset(args.data_dir,
                                                                                     args.train_num, args.test_num)
    #GDloader params:  drop_last=False, shuffle=False
    SSL1_Gtrain_loader = GraphDataLoader(SSL1_Gtrain_iter, batch_size=args.ssl1_batch_size, collate_fn=SSL_Gcollate)
    SSL2_Gtrain_loader = GraphDataLoader(SSL2_Gtrain_iter, batch_size=args.ssl2_batch_size, collate_fn=SSL_Gcollate,
                                                                                                     drop_last=True)
    Gtrain_loader = GraphDataLoader(Gtrain_iter, batch_size=args.batch_size, collate_fn=Gcollate)
    Gtest_loader = GraphDataLoader(Gtest_iter, batch_size=args.batch_size, collate_fn=Gcollate)

    return SSL1_Gtrain_loader, SSL2_Gtrain_loader, Gtrain_loader, Gtest_loader


def valids(model, test_loader, device):
    with th.no_grad():
        sl_criterion = th.nn.CrossEntropyLoss()
        model.eval()
        y_true = []
        y_pred = []
        y_score = []
        loss = []
        iteration = 0
        for batch_id, BGdata in enumerate(test_loader):
            BTgraph, BPgraph, BLgraph, Tlabels = BGdata[0], BGdata[1], BGdata[2], BGdata[3]
            nor_p_features, nor_pe_features, nor_pe_adj, nor_p_adj, p_mat = batch_normalize(BPgraph)

            pred = model(flag=None,
                         bs=args.batch_size,
                         pg=BPgraph.to(args.device),
                         p_features=nor_p_features.to(args.device),
                         pe_features=nor_pe_features.to(args.device),
                         pe_adj=nor_pe_adj.to(args.device),
                         p_adj=nor_p_adj.to(args.device),
                         p_mat=p_mat.to(args.device),

                         tg=BTgraph.to(args.device),
                         tx=BTgraph.ndata['x'].to(args.device),
                         tw=BTgraph.edata['w'].to(args.device),
                         lg=BLgraph.to(args.device),
                         lx=BLgraph.ndata['x'].to(args.device),
                         lw=BLgraph.edata['w'].to(args.device),
                        )

            loss_val = sl_criterion(pred, Tlabels.to(device))
            loss.append(loss_val.item())

            y_score = y_score + th.nn.functional.softmax(pred,1)[:,1].cpu().tolist()
            y_pred = y_pred + th.max(pred,1)[1].cpu().tolist()
            y_true = y_true + Tlabels.cpu().tolist()

            iteration += 1

        np.savetxt('y_score.csv', y_score, delimiter=',')
        np.savetxt('y_pred.csv', y_pred, delimiter=',')
        np.savetxt('y_true.csv', y_true, delimiter=',')

        model.train()
    return sum(loss) / iteration, metrics.accuracy_score(y_true, y_pred), \
                                  metrics.average_precision_score(y_true, y_score), \
                                  metrics.recall_score(y_true, y_pred), \
                                  metrics.f1_score(y_true, y_pred), \
                                  metrics.roc_auc_score(y_true, y_score)


def main(args):
    #2. Basic setup
    setup_logger(output_dir=args.log_dir)
    save_args(args, os.path.join(args.work_dir, "runtime_config.yaml"))
    #If args.seed is negative or None, will use a randomly generated seed
    set_random_seed(args.seed, args.deterministic)

    #3. Create data_loader, model, optimizer, lr_scheduler
    SSL1_Gtrain, SSL2_Gtrain, Gtrain, Gtest = build_dataloader(args)
    model_condig = json.load(open(args.model_config, 'r'))
    Gmodel = GraphBasedModel(model_condig).to(args.device)
    optim = th.optim.Adam(params=Gmodel.parameters(), lr=args.lr)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=1e-5)
    ssl1_criterion = ContrastiveLoss(args.ssl1_batch_size, args.device, args.temperature)
    ssl2_criterion = ContrastiveLoss(args.ssl2_batch_size, args.device, args.temperature)
    sl_criterion = th.nn.CrossEntropyLoss()

    epoach_loss = []
    lSSL1_Gtrain = copy.deepcopy(list(enumerate(SSL1_Gtrain)))
    lSSL2_Gtrain = copy.deepcopy(list(enumerate(SSL2_Gtrain)))
    #here,let ssl1_max_index = ssl2_max_index
    ssl1_MaxIndex = lSSL1_Gtrain.index(max(lSSL1_Gtrain))
    ssl2_MaxIndex = lSSL2_Gtrain.index(max(lSSL2_Gtrain))
    ssl1_index = 0
    ssl2_index = 0
    for epoch in range(args.epochs):
        print("===========================Go for Training=============================================================")
        #start batch normalization and dropout
        Gmodel.train()
        for batch_id, BGdata in enumerate(Gtrain):
            BTgraph, BPgraph, BLgraph, Tlabels = BGdata[0], BGdata[1], BGdata[2], BGdata[3]
            nor_p_features, nor_pe_features, nor_pe_adj, nor_p_adj, p_mat = batch_normalize(BPgraph)

            ssl1_batch_id, ssl1_BGdata = lSSL1_Gtrain[ssl1_index]
            ssl2_batch_id, ssl2_BGdata = lSSL2_Gtrain[ssl2_index]
            ssl1_aug1, ssl1_aug2 = ssl1_BGdata[0], ssl1_BGdata[1]
            ssl2_aug1, ssl2_aug2 = ssl2_BGdata[0], ssl2_BGdata[1]
            self2Aug1_nor_p_features, self2Aug1_nor_pe_features, self2Aug1_nor_pe_adj, \
                                                  self2Aug1_nor_p_adj, self2Aug1_p_mat = batch_normalize(ssl2_aug1)
            self2Aug2_nor_p_features, self2Aug2_nor_pe_features, self2Aug2_nor_pe_adj, \
                                                  self2Aug2_nor_p_adj, self2Aug2_p_mat = batch_normalize(ssl2_aug2)
            print(ssl1_batch_id)
            print(ssl2_batch_id)
            print(ssl1_index)
            print(ssl2_index)

            ssl1_index += 1
            ssl2_index += 1
            #start from scratch
            if ssl1_index > ssl1_MaxIndex:
                ssl1_index = 0
            if ssl2_index > ssl2_MaxIndex:
                ssl2_index = 0

            pred, l1_e1, l1_e2, l2_e1, l2_e2 = Gmodel(flag = 1,
                                      bs=args.batch_size,
                                      pg = BPgraph.to(args.device),
                                      p_features = nor_p_features.to(args.device),
                                      pe_features = nor_pe_features.to(args.device),
                                      pe_adj = nor_pe_adj.to(args.device),
                                      p_adj = nor_p_adj.to(args.device),
                                      p_mat = p_mat.to(args.device),

                                      tg = BTgraph.to(args.device),
                                      tx = BTgraph.ndata['x'].to(args.device),
                                      tw = BTgraph.edata['w'].to(args.device),
                                      lg = BLgraph.to(args.device),
                                      lx = BLgraph.ndata['x'].to(args.device),
                                      lw = BLgraph.edata['w'].to(args.device),

                                      ssl1_bs = args.ssl1_batch_size,
                                      self1Aug1_tlg = ssl1_aug1.to(args.device),
                                      self1Aug1_tlx = ssl1_aug1.ndata['x'].to(args.device),
                                      self1Aug1_tlw = ssl1_aug1.edata['w'].to(args.device),
                                      self1Aug2_tlg = ssl1_aug2.to(args.device),
                                      self1Aug2_tlx = ssl1_aug2.ndata['x'].to(args.device),
                                      self1Aug2_tlw = ssl1_aug2.edata['w'].to(args.device),

                                      ssl2_bs = args.ssl2_batch_size,
                                      self2Aug1_pg = ssl2_aug1.to(args.device),
                                      self2Aug1_p_features = self2Aug1_nor_p_features.to(args.device),
                                      self2Aug1_pe_features= self2Aug1_nor_pe_features.to(args.device),
                                      self2Aug1_pe_adj = self2Aug1_nor_pe_adj.to(args.device),
                                      self2Aug1_p_adj = self2Aug1_nor_p_adj.to(args.device),
                                      self2Aug1_p_mat = self2Aug1_p_mat.to(args.device),

                                      self2Aug2_pg = ssl2_aug2.to(args.device).to(args.device),
                                      self2Aug2_p_features = self2Aug2_nor_p_features.to(args.device),
                                      self2Aug2_pe_features = self2Aug2_nor_pe_features.to(args.device),
                                      self2Aug2_pe_adj = self2Aug2_nor_pe_adj.to(args.device),
                                      self2Aug2_p_adj = self2Aug2_nor_p_adj.to(args.device),
                                      self2Aug2_p_mat = self2Aug2_p_mat.to(args.device),
                                     )

            ssl1_loss = ssl1_criterion(l1_e1, l1_e2)
            ssl2_loss = ssl2_criterion(l2_e1, l2_e2)
            loss = sl_criterion(pred, Tlabels.to(args.device))
            total_loss = loss+0.05*(ssl1_loss+ssl2_loss)

            #loss = loss/args.grad_step
            optim.zero_grad()
            total_loss.backward()
            optim.step()

            ####if (batch_id+1) % args.grad_step == 0:
            ####    optim.step()
            ####    optim.zero_grad()

            if (batch_id+1) % args.log_iter == 0:
                print("Training at epoch: {}, step: {}, loss is: {}".format(epoch+1, batch_id,
                                                                              total_loss.cpu().detach().numpy()))
            if (batch_id+1) == args.batch_num:
                epoach_loss.append(total_loss.cpu().detach().numpy())
                print('\n')
                print("Training after epoch: {}, loss is: {}".format(epoch+1, total_loss.cpu().detach().numpy()))

        if epoach_loss[epoch] < args.min_train_loss:
            args.min_train_loss = epoach_loss[epoch]
            th.save(Gmodel.state_dict(), './final_model.pth')
            print("Save best model after epoch: {}".format(epoch+1))

            print("===========================Go for emetrics=========================================================")
            val_loss, acc, pre, rec, f1, auroc = valids(Gmodel, Gtest, args.device)
            print("epoch:{}, val_loss:{:.5f}, acc:{:.5f}, pre:{:.5f}, rec:{:.5f}, f1:{:.5f}, auroc:{:.5f}"
                                                     .format(epoch+1, val_loss, acc, pre, rec, f1, auroc))

        scheduler.step()

    np.savetxt('./epoach_loss.csv', epoach_loss, delimiter=',')
    #th.save(Gmodel.state_dict(),'./final_model.pth')


if __name__ == '__main__':
    #1. Create an argument parser supporting loading YAML configuration file
    args = parse_args()
    beginT = datetime.now()
    print("Starting Time: {}".format(beginT.strftime(r'%m-%d-%H:%M:%S')))
    main(args)
    endT = datetime.now()
    print("Ending Time: {}".format(endT.strftime(r'%m-%d-%H:%M:%S')))
    interval = endT - beginT
    m, s = divmod(interval.total_seconds(), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print("Duration is {}d {}h {}m {}s".format(d, h, m, s))
