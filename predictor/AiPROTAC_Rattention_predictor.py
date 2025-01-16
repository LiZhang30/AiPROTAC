import os
import json
import logging
import torch as th
import numpy as np
from sklearn import metrics
from datetime import datetime
from dgl.data.utils import load_graphs
from dgl.dataloading import GraphDataLoader
from utils.DglOperator import detect_degree
from utils.DataHelper import LoadData, GetSamples
from utils.DataHelper import batch_normalize
from cpu import ConfigArgumentParser, save_args, set_random_seed, setup_logger
from model.AiPROTAC_Rattention import DatasetIterater, Gcollate, GraphBasedModel


logger = logging.getLogger(__name__)
def parse_args():
    parser = ConfigArgumentParser(description="Degradation")

    parser.add_argument("--log_dir", type=str, default="../log_out", help="Directory to save checkpoints and logs.")
    parser.add_argument("--work_dir", type=str, default="../", help="Directory to save runtime_config.yaml.")
    parser.add_argument("--data_dir", type=str, default="../data/self-built dataset/")
    parser.add_argument("--seed", type=int, default=1, help="Random seed, set to negative to randomize everything.")
    parser.add_argument("--deterministic", action="store_true", help="Turn on the CUDNN deterministic setting.")
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size for training.")
    parser.add_argument("--device", type=str, default='cpu', help="Device to train on cuda:0.")
    #model config
    parser.add_argument("--model_config", type=str, default="../model/config.json", help="Initialize model parameters")

    return parser.parse_args()


def build_dataset(dir):
    targetGraphs, _ = load_graphs(dir+"TargetGraphs_8A.bin")
    protacGraphs, _ = load_graphs(dir+"ProtacGraphs.bin")
    ligaseGraphs, _ = load_graphs(dir+"LigaseGraphs_8A.bin")
    #delete 0-degree nodes for graphs
    detect_degree(targetGraphs)
    detect_degree(protacGraphs)
    detect_degree(ligaseGraphs)

    id_info = LoadData(dir)
    LGtargets, LGprotacs, LGligases, Labels = GetSamples(id_info, targetGraphs, protacGraphs, ligaseGraphs)
    test_iter = DatasetIterater(LGtargets[:], LGprotacs[:], LGligases[:], Labels[:])

    return test_iter


def build_dataloader(args):
    test_iter = build_dataset(args.data_dir)
    #GDloader params:  drop_last=False, shuffle=False
    test_loader = GraphDataLoader(test_iter, batch_size=args.batch_size, collate_fn=Gcollate)

    return test_loader


def main(args):
    #2. Basic setup
    setup_logger(output_dir=args.log_dir)
    save_args(args, os.path.join(args.work_dir, "runtime_config.yaml"))
    #If args.seed is negative or None, will use a randomly generated seed
    set_random_seed(args.seed, args.deterministic)

    #3. Create data_loader, model, optimizer, lr_scheduler
    test_loader = build_dataloader(args)
    model_condig = json.load(open(args.model_config, 'r'))

    with th.no_grad():
        predictor = GraphBasedModel(model_condig).to(args.device)
        predictor.eval()
        predictor.load_state_dict(th.load('1 AiPROTACs-Rattention-seed1 final_model.pth'))

        y_true = []
        y_pred = []
        y_score = []
        iteration = 0
        for batch_id, BGdata in enumerate(test_loader):
            BTgraph, BPgraph, BLgraph, Tlabels = BGdata[0], BGdata[1], BGdata[2], BGdata[3]
            nor_p_features, nor_pe_features, nor_pe_adj, nor_p_adj, p_mat = batch_normalize(BPgraph)

            pred = predictor(bs=args.batch_size,
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

            #y_score是实际预测概率
            y_score = y_score + th.nn.functional.softmax(pred,1)[:,1].cpu().tolist()
            #y_pred是实际预测类别
            y_pred = y_pred + th.max(pred,1)[1].cpu().tolist()
            #y_true是先验类别
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