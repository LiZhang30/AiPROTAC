import torch as th
import numpy as np
import torch.nn as nn
from sklearn import metrics
from datetime import datetime
from cpu import set_random_seed
from utils import DataHelper as DH
from sklearn.ensemble import RandomForestClassifier
from utils.DataHelper import get_MorganFinger, get_MaccsFinger
from utils.DataHelper import LabelPro, LoadData, GetSamples, Shuttle
emb_size = 512
seed = 9
train_num = 829
test_num = 207

def emb_function(seq_list, dic_size, emb_size):
    buf = []
    src_emb = nn.Embedding(dic_size, emb_size)

    for i in range(len(seq_list)):
        emb = src_emb(th.tensor(seq_list[i]))
        squeezed_emb = th.mean(emb, dim=0).squeeze()
        numpy_semb = squeezed_emb.detach().numpy()
        buf.append(numpy_semb)
    return buf


def main():
    set_random_seed(seed)
    #load data source
    target = np.loadtxt('./data/PROTAC-DB 2.0/target.txt', dtype=str, comments='!').tolist()
    protac = np.loadtxt('./data/PROTAC-DB 2.0/protac.txt', dtype=str, comments='!').tolist()
    e3ligase = np.loadtxt('./data/PROTAC-DB 2.0/e3 ligase.txt', dtype=str, comments='!').tolist()
    id_info = LoadData('./data/PROTAC-DB 2.0/')
    #transfer into vector
    ltarget = LabelPro(target)
    lprotac = get_MaccsFinger(protac)
    le3ligase = LabelPro(e3ligase)
    #construct dataset
    ltarget_vec = emb_function(ltarget, DH.pro_dic, emb_size)
    lprotac_vec = emb_function(lprotac, DH.dru_dic, emb_size)
    le3ligase_vec = emb_function(le3ligase, DH.pro_dic, emb_size)
    lvtarget, lvGprotac, lve3ligase, labels = GetSamples(id_info, ltarget_vec, lprotac_vec, le3ligase_vec)
    index = [i for i in range(len(labels))]
    np.random.shuffle(index)
    starget, sprotac, se3ligase, slabels = Shuttle(lvtarget, lvGprotac, lve3ligase, labels, index)
    samples = np.concatenate((starget, sprotac, se3ligase), axis=1)
    #define train and test dataset
    samples_list = samples.tolist()
    slabels_list = slabels.tolist()
    x_train = th.tensor(samples_list[0:train_num])
    y_train = th.tensor(slabels_list[0:train_num])
    x_test = th.tensor(samples_list[train_num:train_num+test_num])
    y_test = th.tensor(slabels_list[train_num:train_num+test_num])

    #define RF model
    model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                   class_weight='balanced', criterion='entropy',
                                   max_depth=6, max_features=0.7602967141008143,
                                   max_leaf_nodes=None, max_samples=None,
                                   min_impurity_decrease=1.922617919965002e-02,
                                   min_samples_leaf=5, min_samples_split=5,
                                   min_weight_fraction_leaf=0.0, n_estimators=5,
                                   n_jobs=-1, oob_score=False, random_state=2934,
                                   verbose=0, warm_start=False)
    #train model
    model.fit(x_train, y_train)
    print(model)

    #test model
    y_true = y_test
    outputs = model.predict_proba(x_test)
    y_pred = model.predict(x_test)
    y_score = []
    y_score = y_score+th.nn.functional.softmax(th.from_numpy(outputs), 1)[:, 1].cpu().tolist()

    np.savetxt('y_score.csv', y_score, delimiter=',')
    np.savetxt('y_pred.csv', y_pred, delimiter=',')
    np.savetxt('y_true.csv', y_true, delimiter=',')
    acc = metrics.accuracy_score(y_true, y_pred)
    pre = metrics.average_precision_score(y_true, y_score)
    rec = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    auroc = metrics.roc_auc_score(y_true, y_score)
    print("acc:{:.5f}, pre:{:.5f}, rec:{:.5f}, f1:{:.5f}, auroc:{:.5f}".format(acc, pre, rec, f1, auroc))


    #For self_buit dataset
    sb_target = np.loadtxt('./data/self-built dataset/target.txt', dtype=str, comments='!').tolist()
    sb_protac = np.loadtxt('./data/self-built dataset/protac.txt', dtype=str, comments='!').tolist()
    sb_e3ligase = np.loadtxt('./data/self-built dataset/e3 ligase.txt', dtype=str, comments='!').tolist()
    sb_id_info = LoadData('./data/self-built dataset/')
    #transfer into vector
    sb_ltarget = LabelPro(sb_target)
    sb_lprotac = get_MaccsFinger(sb_protac)
    sb_le3ligase = LabelPro(sb_e3ligase)
    #construct dataset
    sb_ltarget_vec = emb_function(sb_ltarget, DH.pro_dic, emb_size)
    sb_lprotac_vec = emb_function(sb_lprotac, DH.dru_dic, emb_size)
    sb_le3ligase_vec = emb_function(sb_le3ligase, DH.pro_dic, emb_size)
    sb_lvtarget, sb_lvGprotac, sb_lve3ligase, sb_labels = \
                            GetSamples(sb_id_info, sb_ltarget_vec, sb_lprotac_vec, sb_le3ligase_vec)

    sb_samples = np.concatenate((sb_lvtarget, sb_lvGprotac, sb_lve3ligase), axis=1)
    #define sb_test dataset
    sb_samples_list = sb_samples.tolist()
    sb_x_test = th.tensor(sb_samples_list[:])
    sb_y_test = th.tensor(sb_labels[:])

    sb_y_true = sb_y_test
    sb_outputs = model.predict_proba(sb_x_test)
    sb_y_pred = model.predict(sb_x_test)
    sb_y_score = []
    sb_y_score = sb_y_score+th.nn.functional.softmax(th.from_numpy(sb_outputs), 1)[:, 1].cpu().tolist()

    np.savetxt('sb_y_score.csv', sb_y_score, delimiter=',')
    np.savetxt('sb_y_pred.csv', sb_y_pred, delimiter=',')
    np.savetxt('sb_y_true.csv', sb_y_true, delimiter=',')
    acc = metrics.accuracy_score(sb_y_true, sb_y_pred)
    pre = metrics.average_precision_score(sb_y_true, sb_y_score)
    rec = metrics.recall_score(sb_y_true, sb_y_pred)
    f1 = metrics.f1_score(sb_y_true, sb_y_pred)
    auroc = metrics.roc_auc_score(sb_y_true, sb_y_score)
    print("acc:{:.5f}, pre:{:.5f}, rec:{:.5f}, f1:{:.5f}, auroc:{:.5f}".format(acc, pre, rec, f1, auroc))


if __name__ == '__main__':
    # 1. Create an argument parser supporting loading YAML configuration file
    beginT = datetime.now()
    print("Starting Time: {}".format(beginT.strftime(r'%m-%d-%H:%M:%S')))
    main()
    endT = datetime.now()
    print("Ending Time: {}".format(endT.strftime(r'%m-%d-%H:%M:%S')))
    interval = endT - beginT
    m, s = divmod(interval.total_seconds(), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print("Duration is {}d {}h {}m {}s".format(d, h, m, s))



