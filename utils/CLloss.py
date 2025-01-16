import torch as th
import torch.nn as nn
import torch.nn.functional as F

'''
class SimCSELoss(torch.nn.Module):
    def __init__(self, temperature=0.05, unique=False):
        super(SimCSELoss, self).__init__()
        self.temperature = temperature
        self.unique = unique

    def forward(self, embeddings):
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        y_true = torch.arange(embeddings.size(0)).cuda()

        y_true = (y_true - y_true % 2 * 2) + 1

        # Exclude diagonal elements from similarity matrix
        mask = torch.eye(sim_matrix.shape[0], dtype=torch.bool).cuda()
        sim_matrix.masked_fill_(mask, -1e9)

        sim = sim_matrix / self.temperature
        loss = F.cross_entropy(sim, y_true)
        return torch.mean(loss)
'''
'''
def simcse_sup_loss(y_pred, t=0.05):
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    # [ 0,  1,  3,  4,  6,  7,  9, 10, 12, 13, 15
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    # [ 1,  0,  4,  3,  7,  6, 10,  9, 13, 12,
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 选取有效(use_row中对应的)的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / t
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)
'''
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device, temperature):
        super().__init__()
        self.batch_size = batch_size
        #超参数-温度
        self.register_buffer("temperature", th.tensor(temperature).to(device))
        #主对角线为0，其余位置全为1的mask矩阵
        self.register_buffer("negatives_mask", (~th.eye(batch_size*2, batch_size*2, dtype=bool).to(device)).float())

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1) #(bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1) #(bs, dim)  --->  (bs, dim)

        #repre:(2*bs, dim)
        representations = th.cat([z_i, z_j], dim=0)
        #simi_mat:(2*bs, 2*bs)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = th.diag(similarity_matrix, self.batch_size) #bs
        sim_ji = th.diag(similarity_matrix, -self.batch_size) #bs
        positives = th.cat([sim_ij, sim_ji], dim=0) #2*bs

        nominator = th.exp(positives/self.temperature) #2*bs
        denominator = self.negatives_mask*th.exp(similarity_matrix/self.temperature) #2*bs, 2*bs

        loss_partial = -th.log(nominator/th.sum(denominator, dim=1)) #2*bs
        loss = th.sum(loss_partial)/(2*self.batch_size)

        return loss







