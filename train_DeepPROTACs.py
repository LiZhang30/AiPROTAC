import os
import logging
import torch as th
import numpy as np
from sklearn import metrics
from datetime import datetime
from utils.DataHelper import Shuttle1
from dgl.data.utils import load_graphs
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from utils.DglOperator import detect_degree
from utils.DataHelper import LoadData1, GetSamples1, trans_smiles
from model.DeepPROTACs_Sinput import GraphConv, SmilesNet, ProtacModel
from cpu import ConfigArgumentParser, save_args, set_random_seed, setup_logger
from model.DeepPROTACs_Sinput import DatasetIterater, DatasetIterater1, Gcollate, BatchPad


logger = logging.getLogger(__name__)
def parse_args():
    parser = ConfigArgumentParser(description="Degradation")

    parser.add_argument("--log_dir", type=str, default="./log_out", help="Directory to save checkpoints and logs.")
    parser.add_argument("--work_dir", type=str, default="./", help="Directory to save runtime_config.yaml.")
    parser.add_argument("--data_dir", type=str, default="./data/For DeepPROTACs PROTAC-DB 2.0/", help="Directory to save dataset.")

    parser.add_argument("--seed", type=int, default=1, help="Random seed, set to negative to randomize everything.")
    parser.add_argument("--deterministic", action="store_true", help="Turn on the CUDNN deterministic setting.")
    parser.add_argument("--train_num", type=int, default=829, help="Number of train set.")
    parser.add_argument("--test_num", type=int, default=207, help="Number of test set.")

    parser.add_argument("--log_iter", type=int, default=165, help="Interval for logging to console and tensorboard")
    parser.add_argument("--min_train_loss", type=float, default=1e6, help="Initial minimum epoch train loss")

    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size for training.")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to train on cuda:0.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial Learning rate")
    parser.add_argument("--grad_step", type=int, default=1, help="Gradient accumulate")
    parser.add_argument("--batch_num", type=int, default=829, help="Number of batch")

    return parser.parse_args()


def build_dataset(dir, train_num, test_num):
    targetGraphs, _ = load_graphs(dir+"TargetGraphs_8A.bin")
    warheadGraphs, _ = load_graphs(dir+"WarheadGraphs.bin")
    e3ligandGraphs, _ = load_graphs(dir+"E3ligandGraphs.bin")
    e3ligaseGraphs, _ = load_graphs(dir+"LigaseGraphs_8A.bin")
    protacSeq = np.loadtxt(dir+'protac.txt', dtype=str).tolist()
    linkerSeq = np.loadtxt(dir+'linker.txt', dtype=str).tolist()
    #delete 0-degree nodes for graphs
    detect_degree(targetGraphs)
    detect_degree(warheadGraphs)
    detect_degree(e3ligandGraphs)
    detect_degree(e3ligaseGraphs)
    #construct samples
    id_info = LoadData1(dir)
    LGtargets, LGwarheads, linker, LGe3ligand, LGe3ligases, Labels = GetSamples1(id_info, targetGraphs,
                                   protacSeq, warheadGraphs, linkerSeq, e3ligandGraphs, e3ligaseGraphs)
    #encode smiles
    Llinker = []
    for i in range(len(linker)):
         Llinker.append(trans_smiles(linker[i]))
    #shuttle samples
    index = [i for i in range(len(Labels))]
    np.random.shuffle(index)
    SLGtargets, SLGwarheads, SLlinker, SLGe3ligand, SLGe3ligases, Slabels = \
                       Shuttle1(LGtargets, LGwarheads, Llinker, LGe3ligand, LGe3ligases, Labels, index)

    Gtrain_iter = DatasetIterater(SLGtargets[0:train_num], SLGwarheads[0:train_num], SLGe3ligand[0:train_num],
                                                                    SLGe3ligases[0:train_num], Slabels[0:train_num])
    Strain_iter = DatasetIterater1(SLlinker[0:train_num])
    Gtest_iter = DatasetIterater(SLGtargets[train_num:train_num+test_num], SLGwarheads[train_num:train_num+test_num],
                             SLGe3ligand[train_num:train_num + test_num], SLGe3ligases[train_num:train_num+test_num],
                                                                              Slabels[train_num:train_num+test_num])
    Stest_iter = DatasetIterater1(SLlinker[train_num:train_num+test_num])

    return Gtrain_iter, Strain_iter, Gtest_iter, Stest_iter


def build_dataloader(args):
    Gtrain_iter, Strain_iter, Gtest_iter, Stest_iter = build_dataset(args.data_dir, args.train_num, args.test_num)
    #GDloader params:  drop_last=False, shuffle=False
    Gtrain_loader = GraphDataLoader(Gtrain_iter, batch_size=args.batch_size, collate_fn=Gcollate)
    Strain_loader = DataLoader(Strain_iter, batch_size=args.batch_size, collate_fn=BatchPad)
    Gtest_loader = GraphDataLoader(Gtest_iter, batch_size=args.batch_size, collate_fn=Gcollate)
    Stest_loader = DataLoader(Stest_iter, batch_size=args.batch_size, collate_fn=BatchPad)

    return Gtrain_loader, Strain_loader, Gtest_loader, Stest_loader


def valids(model, Gtest_loader, Stest_loader, device):
    with th.no_grad():
        sl_criterion = th.nn.CrossEntropyLoss()
        model.eval()
        y_true = []
        y_pred = []
        y_score = []
        loss = []
        iteration = 0
        for batch_id, (BGdata, BSdata) in enumerate(zip(Gtest_loader, Stest_loader)):
            BTgraph, BWgraph, BLdgraph, BLegraph, Tlabels = \
                    BGdata[0], BGdata[1], BGdata[2], BGdata[3], BGdata[4]

            pred = model(BLdgraph.to(args.device),
                         BLegraph.to(args.device),
                         BWgraph.to(args.device),
                         BTgraph.to(args.device),
                         BSdata.to(args.device),)

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
    return sum(loss)/iteration, metrics.accuracy_score(y_true, y_pred), \
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
    Gtrain, Strain, Gtest, Stest = build_dataloader(args)

    ligase_ligand_model = GraphConv(flag=1)
    ligase_pocket_model = GraphConv(flag=0)
    target_ligand_model = GraphConv(flag=1)
    target_pocket_model = GraphConv(flag=0)
    smiles_model = SmilesNet()
    Gmodel = ProtacModel(ligase_ligand_model,
                         ligase_pocket_model,
                         target_ligand_model,
                         target_pocket_model,
                         smiles_model).to(args.device)
    optim = th.optim.Adam(params=Gmodel.parameters(), lr=args.lr)
    sl_criterion = th.nn.CrossEntropyLoss()

    epoach_loss = []
    for epoch in range(args.epochs):
        print("===========================Go for Training=============================================================")
        #start batch normalization and dropout
        Gmodel.train()
        for batch_id, (BGdata, BSdata) in enumerate(zip(Gtrain, Strain)):
            BTgraph, BWgraph, BLdgraph, BLegraph, Tlabels = \
                    BGdata[0], BGdata[1], BGdata[2], BGdata[3], BGdata[4]

            pred = Gmodel(BLdgraph.to(args.device),
                          BLegraph.to(args.device),
                          BWgraph.to(args.device),
                          BTgraph.to(args.device),
                          BSdata.to(args.device),)

            loss = sl_criterion(pred, Tlabels.to(args.device))
            #loss = loss/args.grad_step
            optim.zero_grad()
            loss.backward()
            optim.step()

            ####if (batch_id+1) % args.grad_step == 0:
            ####    optim.step()
            ####    optim.zero_grad()

            if (batch_id+1) % args.log_iter == 0:
                print("Training at epoch: {}, step: {}, loss is: {}".format(epoch+1, batch_id,
                                                                              loss.cpu().detach().numpy()))
            if (batch_id+1) == args.batch_num:
                epoach_loss.append(loss.cpu().detach().numpy())
                print('\n')
                print("Training after epoch: {}, loss is: {}".format(epoch+1, loss.cpu().detach().numpy()))

        if epoach_loss[epoch] < args.min_train_loss:
            args.min_train_loss = epoach_loss[epoch]
            th.save(Gmodel.state_dict(), './final_model.pth')
            print("Save best model after epoch: {}".format(epoch+1))

            print("===========================Go for emetrics=========================================================")
            val_loss, acc, pre, rec, f1, auroc = valids(Gmodel, Gtest, Stest, args.device)
            print("epoch:{}, val_loss:{:.5f}, acc:{:.5f}, pre:{:.5f}, rec:{:.5f}, f1:{:.5f}, auroc:{:.5f}"
                                                     .format(epoch+1, val_loss, acc, pre, rec, f1, auroc))

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