import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from model import HEAL, MaskedNLLLoss, init_seed
from dataloader import MyDataset
import warnings
warnings.filterwarnings("ignore")

def get_loaders(path, batch_size=32, num_workers=0, pin_memory=False):
    trainset = MyDataset(path, 'train')
    validset = MyDataset(path, 'dev')
    testset = MyDataset(path, 'test')
    
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):    
    losses = []
    preds = []
    labels = []
    masks = []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()           
        textf, qmask, umask, label = [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]        
        log_prob, _ = model(textf, qmask, umask)
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1) 
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1) 
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()            

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore
    
def weight_calculate():
    #The correspongding weight is the sample num in the training set (For IEMOCAP, it's train+val set)
    lst = [1324, 1468, 933, 839, 452, 742]
    m = max(lst)
    res = []
    for i in lst:
        res.append(round(m/i, 4))
    print(f'loss weights:{res}')
    return res
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-6, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--D_m', type=int, default=128, metavar='hidden', help='hidden dimension')
    parser.add_argument('--n_layers', type=int, default=3, metavar='n_layers', help='Transformer layers num')
    parser.add_argument('--n_layers_rnn', type=int, default=3, metavar='n_layers_rnn', help='RNN layers num')
    parser.add_argument('--heads', type=int, default=64, metavar='heads', help='attention heads')
    parser.add_argument('--batch-size', type=int, default=512, metavar='BS', help='batch size')
    parser.add_argument('--pool_type', type=str, default='cat', help='cat/att/mean')
    parser.add_argument('--rnn_type', type=str, default='lstm', help='lstm/gru')
    parser.add_argument('--dim_feedforward', type=float, default=0.5, help='dim_feedforward of Transformer FFN')
    parser.add_argument('--epochs', type=int, default=120, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weight')
    args = parser.parse_args()
    init_seed(args.seed, True)

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')
        
        
    batch_size = args.batch_size
    # IEMOCAP has 6 classes
    n_classes  = 6
    # use roberta-large, which input dim is 1024
    D_e = 1024    

    dataset = './data/IEMOCAP/iemocap_features_roberta.pkl'
    print(f'dataset: {dataset}')
    model = HEAL(args.D_m, D_e, args.n_layers, args.heads, args.dropout, n_classes, args.pool_type, args.n_layers_rnn, args.rnn_type, args.dim_feedforward)
    if args.cuda:
        model.cuda()

    if args.class_weight:
        loss_weights = torch.FloatTensor(weight_calculate())
        loss_function  = MaskedNLLLoss(loss_weights.cuda() if args.cuda else loss_weights)
        print('use loss weight')
    else:
        loss_function = MaskedNLLLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    train_loader, valid_loader, test_loader = get_loaders(dataset, batch_size=args.batch_size)

    best_loss, best_label, best_pred, best_mask, best_f1, best_val_f1 = None, None, None, None, None, None    
    best_test_acc, best_test_f1 = None, None
    for e in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function,
                                               train_loader, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, val_fscore = train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, test_loader, e)

    #   Save the best performance according to valid set f1 score
        if best_val_f1 == None or val_fscore > best_val_f1:
            best_loss, best_label, best_pred, best_mask, best_f1 =\
                    test_loss, test_label, test_pred, test_mask, test_fscore
            best_val_f1 = val_fscore
        if best_test_f1 == None or (test_fscore+test_acc) > (best_test_acc+best_test_f1):
            best_test_acc, best_test_f1 = test_acc, test_fscore
        #print current performance
        print('epoch {} train_loss {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e+1, train_loss, valid_loss, valid_acc, val_fscore,\
                        test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
    #Final Result Selected by Valid set F1 score
    print('-'*100)
    print('Validation Set Select Best Test Performance:')
    #best_mask is all 1, don't have other weights.
    print('weighted-f1-score {}'.format(round(f1_score(best_label, best_pred, sample_weight=best_mask, average='weighted')*100, 2)), end='; ')
    print('acc:', round(accuracy_score(best_label, best_pred, sample_weight=best_mask)*100, 2))
    print('-'*100)
    print('Model\'s Best Performance on Test Set:')
    print(f'weighted-f1-score {best_test_f1}; acc: {best_test_acc}')
    print('-'*100)