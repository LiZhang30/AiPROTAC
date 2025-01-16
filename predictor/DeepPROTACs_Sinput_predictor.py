import os
import logging
import torch as th
import numpy as np
from sklearn import metrics
from datetime import datetime
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

    parser.add_argument("--log_dir", type=str, default="../log_out", help="Directory to save checkpoints and logs.")
    parser.add_argument("--work_dir", type=str, default="../", help="Directory to save runtime_config.yaml.")
    parser.add_argument("--data_dir", type=str, default="../data/For DeepPROTACs self-built dataset/")
    parser.add_argument("--seed", type=int, default=1, help="Random seed, set to negative to randomize everything.")
    parser.add_argument("--deterministic", action="store_true", help="Turn on the CUDNN deterministic setting.")
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size for training.")
    parser.add_argument("--device", type=str, default='cpu', help="Device to train on cuda:0.")

    return parser.parse_args()


def build_dataset(dir):
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

    Gtest_iter = DatasetIterater(LGtargets[:], LGwarheads[:], LGe3ligand[:], LGe3ligases[:], Labels[:])
    Stest_iter = DatasetIterater1(Llinker[:])

    return Gtest_iter, Stest_iter


def build_dataloader(args):
    Gtest_iter, Stest_iter = build_dataset(args.data_dir)
    #GDloader params:  drop_last=False, shuffle=False
    Gtest_loader = GraphDataLoader(Gtest_iter, batch_size=args.batch_size, collate_fn=Gcollate)
    Stest_loader = DataLoader(Stest_iter, batch_size=args.batch_size, collate_fn=BatchPad)

    return Gtest_loader, Stest_loader


def main(args):
    #2. Basic setup
    setup_logger(output_dir=args.log_dir)
    save_args(args, os.path.join(args.work_dir, "runtime_config.yaml"))
    #If args.seed is negative or None, will use a randomly generated seed
    set_random_seed(args.seed, args.deterministic)

    #3. Create data_loader
    Gtest_loader, Stest_loader = build_dataloader(args)

    with th.no_grad():
        ligase_ligand_model = GraphConv(flag=1)
        ligase_pocket_model = GraphConv(flag=0)
        target_ligand_model = GraphConv(flag=1)
        target_pocket_model = GraphConv(flag=0)
        smiles_model = SmilesNet()
        predictor = ProtacModel(ligase_ligand_model,
                            ligase_pocket_model,
                            target_ligand_model,
                            target_pocket_model,
                            smiles_model).to(args.device)
        predictor.eval()
        predictor.load_state_dict(th.load('1 DeepPROTACs-Sinput-seed1 final_model.pth'))

        y_true = []
        y_pred = []
        y_score = []
        iteration = 0
        for batch_id, (BGdata, BSdata) in enumerate(zip(Gtest_loader, Stest_loader)):
            BTgraph, BWgraph, BLdgraph, BLegraph, Tlabels = \
                BGdata[0], BGdata[1], BGdata[2], BGdata[3], BGdata[4]

            pred = predictor(BLdgraph.to(args.device),
                             BLegraph.to(args.device),
                             BWgraph.to(args.device),
                             BTgraph.to(args.device),
                             BSdata.to(args.device), )

            # y_score是实际预测概率
            y_score = y_score + th.nn.functional.softmax(pred, 1)[:, 1].cpu().tolist()
            # y_pred是实际预测类别
            y_pred = y_pred + th.max(pred, 1)[1].cpu().tolist()
            # y_true是先验类别
            y_true = y_true + Tlabels.cpu().tolist()

            iteration += 1

        np.savetxt('y_score.csv', y_score, delimiter=',')
        np.savetxt('y_pred.csv', y_pred, delimiter=',')
        np.savetxt('y_true.csv', y_true, delimiter=',')

        acc = metrics.accuracy_score(y_true, y_pred)
        pre = metrics.average_precision_score(y_true, y_score)
        rec = metrics.recall_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        auroc = metrics.roc_auc_score(y_true, y_score)
        print("acc:{:.5f}, pre:{:.5f}, rec:{:.5f}, f1:{:.5f}, auroc:{:.5f}".format(acc, pre, rec, f1, auroc))


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