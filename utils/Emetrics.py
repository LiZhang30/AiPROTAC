import subprocess
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def get_aupr(Y, P):
    if hasattr(Y, 'A'): Y = Y.A
    if hasattr(P, 'A'): P = P.A
    Y = np.where(Y>0, 1, 0)
    Y = Y.ravel()
    P = P.ravel()
    f = open("temp.txt", 'w')
    for i in range(Y.shape[0]):
        f.write("%f %d\n" %(P[i], Y[i]))
    f.close()
    f = open("foo.txt", 'w')
    subprocess.call(["java", "-jar", "auc.jar", "temp.txt", "list"], stdout=f)
    f.close()
    f = open("foo.txt")
    lines = f.readlines()
    aucpr = float(lines[-2].split()[-1])
    f.close()
    return aucpr

def get_cindex(Y, P):
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair is not 0:
        return summ/pair
    else:
        return 0

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred-y_pred_mean) * (y_obs-y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs-y_obs_mean) * (y_obs-y_obs_mean))
    y_pred_sq = sum((y_pred-y_pred_mean) * (y_pred-y_pred_mean))

    return mult / float(y_obs_sq*y_pred_sq)

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs-(k*y_pred)) * (y_obs-(k*y_pred)))
    down= sum((y_obs-y_obs_mean) * (y_obs-y_obs_mean))

    return 1 - (upp/float(down))

def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2*(1-np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def get_MSE(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum((y_obs-y_pred)*(y_obs-y_pred))/len(y_obs)

def get_RMSE(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return np.sqrt(sum((y_obs-y_pred) * (y_obs-y_pred)) / len(y_obs))

def get_spearmanr(y_obs, y_pred):
    diff_pred, diff_obs = y_pred-np.mean(y_pred), y_obs-np.mean(y_obs)
    return np.sum(diff_pred*diff_obs) / np.sqrt(np.sum(diff_pred**2) * np.sum(diff_obs**2))

def get_PCC(X, Y):
    XMean = np.mean(X)
    YMean = np.mean(Y)

    XSD = np.std(X)
    YSD = np.std(Y)

    ZX = (X-XMean) / XSD
    ZY = (Y-YMean) / YSD

    r = np.sum(ZX*ZY) / (len(X))

    return r


y_true1 = np.loadtxt('./y_true1.txt', dtype=int).tolist()
'''y_true2 = np.loadtxt('./y_true2.txt', dtype=int).tolist()
y_true3 = np.loadtxt('./y_true3.txt', dtype=int).tolist()
y_true4 = np.loadtxt('./y_true4.txt', dtype=int).tolist()
y_true5 = np.loadtxt('./y_true5.txt', dtype=int).tolist()'''


y_RF_MACCS = np.loadtxt('./y_score1_RF-MACCS.txt', dtype=float).tolist()
y_RF_Morgan = np.loadtxt('./y_score1_RF-Morgan.txt', dtype=float).tolist()
y_SVM_MACCS = np.loadtxt('./y_score1_SVM-MACCS.txt', dtype=float).tolist()
y_SVM_Morgan = np.loadtxt('./y_score1_SVM-Morgan.txt', dtype=float).tolist()
y_DeepPROTACs = np.loadtxt('./y_score1_DeepPROTACs.txt', dtype=float).tolist()
y_DeepPROTACs_Sinput = np.loadtxt('./y_score1_DeepPROTACs-Sinput.txt', dtype=float).tolist()
y_AiPROTACs_GCN = np.loadtxt('./y_score1_AiPROTACs-GCN.txt', dtype=float).tolist()
y_AiPROTACs_GAT = np.loadtxt('./y_score1_AiPROTACs-GAT.txt', dtype=float).tolist()
y_AiPROTACs_Rattention = np.loadtxt('./y_score1_AiPROTACs-RgraphCL.txt', dtype=float).tolist()
y_AiPROTACs_RgraphCL = np.loadtxt('./y_score1_AiPROTACs-Rattention.txt', dtype=float).tolist()
y_AiPROTACs = np.loadtxt('./y_score1_AiPROTACs.txt', dtype=float).tolist()


fpr1, tpr1, thread1 = metrics.roc_curve(y_true1, y_RF_MACCS)
roc_auc1 = metrics.auc(fpr1, tpr1)

fpr2, tpr2, thread2 = metrics.roc_curve(y_true1, y_RF_Morgan)
roc_auc2 = metrics.auc(fpr2, tpr2)

fpr3, tpr3, thread3 = metrics.roc_curve(y_true1, y_SVM_MACCS)
roc_auc3 = metrics.auc(fpr3, tpr3)

fpr4, tpr4, thread4 = metrics.roc_curve(y_true1, y_SVM_Morgan)
roc_auc4 = metrics.auc(fpr4, tpr4)

fpr5, tpr5, thread5 = metrics.roc_curve(y_true1, y_DeepPROTACs)
roc_auc5 = metrics.auc(fpr5, tpr5)

fpr6, tpr6, thread6 = metrics.roc_curve(y_true1, y_DeepPROTACs_Sinput)
roc_auc6 = metrics.auc(fpr6, tpr6)

fpr7, tpr7, thread7 = metrics.roc_curve(y_true1, y_AiPROTACs_GCN)
roc_auc7 = metrics.auc(fpr7, tpr7)

fpr8, tpr8, thread8 = metrics.roc_curve(y_true1, y_AiPROTACs_GAT)
roc_auc8 = metrics.auc(fpr8, tpr8)

fpr9, tpr9, thread9 = metrics.roc_curve(y_true1, y_AiPROTACs_Rattention)
roc_auc9 = metrics.auc(fpr9, tpr9)

fpr10, tpr10, thread10 = metrics.roc_curve(y_true1, y_AiPROTACs_RgraphCL)
roc_auc10 = metrics.auc(fpr10, tpr10)

fpr11, tpr11, thread11 = metrics.roc_curve(y_true1, y_AiPROTACs)
roc_auc11 = metrics.auc(fpr11, tpr11)

'''
plt.figure()
lw = 1
plt.plot(fpr1, tpr1, color='olive', lw=2, label='RF-MACCS (area=%0.4f)'%roc_auc1)
plt.plot(fpr2, tpr2, color='lightgreen', lw=2, label='RF-Morgan (area=%0.4f)'%roc_auc2)
plt.plot(fpr3, tpr3, color='skyblue', lw=2, label='SVM-MACCS (area=%0.4f)'%roc_auc3)
plt.plot(fpr4, tpr4, color='brown', lw=2, label='SVM-Morgan (area=%0.4f)'%roc_auc4)
plt.plot(fpr5, tpr5, color='grey', lw=2, label='DeepPROTACs (area=%0.4f)' % roc_auc5)
plt.plot(fpr6, tpr6, color='gold', lw=2, label='DeepPROTACs-Sinput (area=%0.4f)'%roc_auc6)
plt.plot(fpr7, tpr7, color='cyan', lw=2, label='AiPROTACs-GCN (area=%0.4f)'%roc_auc7)
plt.plot(fpr8, tpr8, color='lime', lw=2, label='AiPROTACs-GAT (area=%0.4f)'%roc_auc8)
plt.plot(fpr9, tpr9, color='darkviolet', lw=2, label='AiPROTACs-Rattention (area=%0.4f)'%roc_auc9)
plt.plot(fpr10, tpr10, color='blue', lw=2, label='AiPROTACs-RgraphCL (area=%0.4f)'%roc_auc10)
plt.plot(fpr11, tpr11, color='red', lw=2, label='AiPROTACs (area=%0.4f)'%roc_auc11)

#plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('(The ROC Curve)')
plt.legend(loc="lower right", prop={'size':10})
plt.savefig('auc-roc.pdf', )
plt.show()

'''
#y_true = np.loadtxt('./y_true.txt', dtype=int).tolist()
#y_score1 = np.loadtxt('./y_score1.txt', dtype=float).tolist()
#y_score2 = np.loadtxt('./y_score2.txt', dtype=float).tolist()

precision1, recall1, _ = metrics.precision_recall_curve(y_true1, y_RF_MACCS)
roc_pr1 = metrics.auc(recall1, precision1)

precision2, recall2, _ = metrics.precision_recall_curve(y_true1, y_RF_Morgan)
roc_pr2 = metrics.auc(recall2, precision2)

precision3, recall3, _ = metrics.precision_recall_curve(y_true1, y_SVM_MACCS)
roc_pr3 = metrics.auc(recall3, precision3)

precision4, recall4, _ = metrics.precision_recall_curve(y_true1, y_SVM_Morgan)
roc_pr4 = metrics.auc(recall4, precision4)

precision5, recall5, _ = metrics.precision_recall_curve(y_true1, y_DeepPROTACs)
roc_pr5 = metrics.auc(recall5, precision5)

precision6, recall6, _ = metrics.precision_recall_curve(y_true1, y_DeepPROTACs_Sinput)
roc_pr6 = metrics.auc(recall6, precision6)

precision7, recall7, _ = metrics.precision_recall_curve(y_true1, y_AiPROTACs_GCN)
roc_pr7 = metrics.auc(recall7, precision7)

precision8, recall8, _ = metrics.precision_recall_curve(y_true1, y_AiPROTACs_GAT)
roc_pr8 = metrics.auc(recall8, precision8)

precision9, recall9, _ = metrics.precision_recall_curve(y_true1, y_AiPROTACs_Rattention)
roc_pr9 = metrics.auc(recall9, precision9)

precision10, recall10, _ = metrics.precision_recall_curve(y_true1, y_AiPROTACs_RgraphCL)
roc_pr10 = metrics.auc(recall10, precision10)

precision11, recall11, _ = metrics.precision_recall_curve(y_true1, y_AiPROTACs)
roc_pr11 = metrics.auc(recall11, precision11)

plt.figure()
lw = 1
plt.plot(recall1, precision1, color='olive', lw=2, label='RF-MACCS (area=%0.4f)'%roc_pr1)
plt.plot(recall2, precision2, color='lightgreen', lw=2, label='RF-Morgan (area=%0.4f)'%roc_pr2)
plt.plot(recall3, precision3, color='skyblue', lw=2, label='SVM-MACCS (area=%0.4f)'%roc_pr3)
plt.plot(recall4, precision4, color='brown', lw=2, label='SVM-Morgan (area=%0.4f)'%roc_pr4)
plt.plot(recall5, precision5, color='grey', lw=2, label='DeepPROTACs (area=%0.4f)'%roc_pr5)
plt.plot(recall6, precision6, color='gold', lw=2, label='DeepPROTACs-Sinput (area=%0.4f)'%roc_pr6)
plt.plot(recall7, precision7, color='cyan', lw=2, label='AiPROTACs-GCN (area=%0.4f)'%roc_pr7)
plt.plot(recall8, precision8, color='lime', lw=2, label='AiPROTACs-GAT (area=%0.4f)'%roc_pr8)
plt.plot(recall9, precision9, color='darkviolet', lw=2, label='AiPROTACs-Rattention (area=%0.4f)'%roc_pr9)
plt.plot(recall10, precision10, color='blue', lw=2, label='AiPROTACs-RgraphCL (area=%0.4f)'%roc_pr10)
plt.plot(recall11, precision11, color='red', lw=2, label='AiPROTACs (area=%0.4f)'%roc_pr11)

#plt.plot([1, 0], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('(The PR Curve)')
plt.legend(loc="lower right", prop={'size':10})
plt.savefig('auc-pr.pdf', )
plt.show()






