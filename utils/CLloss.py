import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", th.tensor(temperature).to(device))
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







