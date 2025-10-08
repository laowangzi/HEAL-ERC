import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import random
from sklearn.metrics import f1_score

def init_seed(seed, reproducibility):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss    

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor

    
class HEAL(nn.Module):
    def __init__(self, D_m, D_e, n_layers, heads, dropout, n_classes, pool_type, n_layers_rnn, rnn_type='lstm', feedforward=2):
        super(HEAL, self).__init__()
        n_layers = n_layers
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.n_classes = n_classes
        self.rnn_D_h = D_m
        self.pool_type = pool_type

        # multi-layers transformer blocks, deep network
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=D_m, nhead=self.heads,dim_feedforward=int(D_m*feedforward),batch_first=True, dropout=dropout)
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        # RNNs
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=D_e, hidden_size=D_m, num_layers=n_layers_rnn, bidirectional=True, dropout=dropout, batch_first=True)
            self.rnn_s = nn.LSTM(input_size=D_e, hidden_size=D_m, num_layers=n_layers_rnn, bidirectional=True, dropout=dropout, batch_first=True)
        # GRU
        else:
            self.rnn = nn.GRU(input_size=D_e, hidden_size=D_m, num_layers=n_layers_rnn, bidirectional=True, dropout=dropout, batch_first=True)
            self.rnn_s = nn.GRU(input_size=D_e, hidden_size=D_m, num_layers=n_layers_rnn, bidirectional=True, dropout=dropout, batch_first=True)
        # Latent Dependency Generator    
        self.generator = LLDG(D_m)
        
        # Fusion module
        if self.pool_type == 'cat':
            D = 7
        elif self.pool_type == 'mean':
            D = 1
        elif self.pool_type == 'att':
            D = 1
            self.attention = SimpleAttention(D_m)
        #Prediction    
        self.linear = nn.Linear(D*D_m, int(D*D_m/2))
        self.smax_fc = nn.Linear(int(D*D_m/2), self.n_classes)
        #project to hidden dim
        self.proj = nn.Linear(D_e, D_m)
        
        self.layer_norm = nn.LayerNorm(D_e)
        
        self.init_parameters()
        
    def _init_rnn(self, weight):
        for w in weight.chunk(4, 0):
            nn.init.xavier_uniform_(w)
    def init_parameters(self):
        self._init_rnn(self.rnn.weight_ih_l0)
        self._init_rnn(self.rnn.weight_hh_l0)
        self.rnn.bias_ih_l0.data.zero_()
        self.rnn.bias_hh_l0.data.zero_()
        self._init_rnn(self.rnn_s.weight_ih_l0)
        self._init_rnn(self.rnn_s.weight_hh_l0)
        self.rnn_s.bias_ih_l0.data.zero_()
        self.rnn_s.bias_hh_l0.data.zero_()

        nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.smax_fc.weight, mode='fan_in', nonlinearity='relu')

    def rnn_speaker(self, U, umask, qmask):
        u_num = int(sum(umask).detach().cpu())
        U = U[:u_num, :]
        qmask = qmask[:u_num]
        N, D = U.size()
        unique_labels = qmask.unique()
        output = torch.zeros((N, 2*self.rnn_D_h)).cuda()
        for label in unique_labels:
            mask = (qmask == label).nonzero().squeeze()
            if mask.numel() > 0:
                U_label = U[mask]
                output[mask], _ = self.rnn_s(U_label.unsqueeze(0))
        pad = torch.zeros((umask.shape[0]-N, 2*self.rnn_D_h)).cuda()
        return torch.concat((output, pad))

    def forward(self, U, qmask, umask):         
        #Linear Project to Hidden Dim
        U_proj = self.proj(U)
        
        qmask = qmask.transpose(0, 1)
        U = U.transpose(0, 1)    #bs*seqlen*D
        U_proj = U_proj.transpose(0, 1)
        mask = (torch.sum(U_proj, -1).squeeze() == 0.0)   #bs*seqlen
        bs, seqlen = U.shape[0], U.shape[1]
        #RNN global
        emotions_rnn, _ = self.rnn(U)
        #RNN speaker
        emotions_rnn_s = []
        for u, q, um in zip(U, qmask, umask):
            e_rnn_s = self.rnn_speaker(u, um, q)
            emotions_rnn_s.append(e_rnn_s)
        emotions_rnn_s = torch.stack(emotions_rnn_s)

        #LLDG module
        gmask = self.generator(U_proj)
        emotions_gan = []
        for u, m, g in zip(U_proj, mask, gmask):
            e_g = self.transformer_blocks(u.unsqueeze(0), src_key_padding_mask=m.unsqueeze(0), mask=g).squeeze()   #seqlen*D
            emotions_gan.append(e_g)
        emotions_gan = torch.stack(emotions_gan)   #bs*seqlen*D

        #Transformer Global
        emotions = self.transformer_blocks(U_proj, src_key_padding_mask=mask)
        
        #Transformer Speaker
        qmask = speaker_mask(qmask)  #bs*seqlen*seqlen
        emotions_speaker = []
        for u, m, q in zip(U_proj, mask, qmask):
            e_q = self.transformer_blocks(u.unsqueeze(0), src_key_padding_mask=m.unsqueeze(0), mask=q).squeeze()   #seqlen*D
            emotions_speaker.append(e_q)
        emotions_speaker = torch.stack(emotions_speaker)   #bs*seqlen*D
        
        # Fusion
        alpha = []
        
        if self.pool_type == 'cat':
            emotions = torch.cat((emotions, emotions_speaker, emotions_gan, emotions_rnn, emotions_rnn_s), -1)
        elif self.pool_type == 'mean':
            t1, t2 = torch.chunk(emotions_rnn, 2, -1)
            t3, t4 = torch.chunk(emotions_rnn_s, 2, -1)
            emotions = (emotions + emotions_speaker + emotions_gan + t1 + t2 + t3 + t4)/7.0
        elif self.pool_type == 'att':
            emotions = torch.cat((emotions, emotions_speaker, emotions_gan, emotions_rnn, emotions_rnn_s), -1)
            emotions, alpha = self.attention(emotions)  #bs*seqlen*D

        #Prediction
        hidden = F.relu(self.linear(emotions)).transpose(0, 1)   #seqlen*bs*2D
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), -1) 
        
        return log_prob, alpha    

    
class LLDG(nn.Module):
    def __init__(self, input_dim):
        super(LLDG, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.tanh = nn.Tanh()
#         self.dropout = nn.Dropout(dropout)
    def forward(self, noise):
        x = self.fc1(noise)
#         x = self.tanh(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = self.tanh(x)
        
        output = torch.matmul(x, x.transpose(1, 2))        
        output = self.tanh(output)        
        output = torch.where(output > 0, torch.zeros_like(output), torch.ones_like(output) * float('-inf'))
        return output
    


#Generate speaker-specific attention mask matrix, where every speaker can only aware utterences him/her-self
def speaker_mask(input_tensor):
    expanded_tensor = input_tensor.unsqueeze(2).expand(-1, -1, input_tensor.shape[1])
    comparison_tensor = input_tensor.unsqueeze(1).expand(-1, input_tensor.shape[1], -1)
    bool_tensor = expanded_tensor == comparison_tensor
    float_tensor = bool_tensor.float()
    float_tensor[bool_tensor == False] = float('-inf')
    float_tensor[bool_tensor == True] = float(0)
    return float_tensor


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'best model save at {path}')
    
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f'load best model from {path}')
    return model
    
class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super(SimpleAttention, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        # x shape: (BS, SL, 7*D)
        bs, sl, _ = x.size()
        x = x.view(bs*sl, -1, self.dim)  # reshape to (BS*SL, 7, D)
        scores = self.linear(x).squeeze()  # compute scores (BS*SL, 7)
        weights = F.softmax(scores, dim=-1)  # compute weights (BS*SL, 7)
        x = x.view(bs*sl, -1, self.dim)  # reshape x to (BS*SL, 7, D)
        out = (weights.unsqueeze(2) * x).sum(dim=1)  # weighted sum (BS*SL, D)
        out = out.view(bs, sl, self.dim)  # reshape to (BS, SL, D)
        return out, weights