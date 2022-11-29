import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import argparse
import time
import torch
from torch import nn
from torch_geometric.data import HeteroData
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from Dataset import MGTAB
from models import HGT
from utils import sample_mask
import numpy as np


parser = argparse.ArgumentParser(description='HGT')
parser.add_argument('--task', type=str, default='bot', help='detection task of stance or bot')
parser.add_argument('--relation_select', type=int, default=[0,1], nargs='+', help='selection of relations in the graph (0-6).')
parser.add_argument('--random_seed', type=int, default=[1,2,3,4,5], nargs='+', help='selection of random seeds')
parser.add_argument('--hidden_dimension', type=int, default=256, help="linear channels")
parser.add_argument('--linear_channels', type=int, default=128, help="linear channels")
parser.add_argument('--out_channel', type=int, default=32, help='output channels')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate (1 - keep probability)')
parser.add_argument('--epochs', type=int, default=200, help='description channel')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for optimizer')
args = parser.parse_args()


def main(seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MGTAB('./Dataset/MGTAB')
    origin_data = dataset[0]

    relation_dict = {
        0:'followers',
        1:'friends',
        2:'mention',
        3:'reply',
        4:'quoted',
        5:'url',
        6:'hashtag'
    }

    sample_number = origin_data.x.shape[0]
    args.features_num = origin_data.x.shape[1]

    shuffled_idx = shuffle(np.array(range(sample_number)), random_state=seed)
    train_idx = shuffled_idx[:int(0.7 * sample_number)]
    val_idx = shuffled_idx[int(0.7 * sample_number):int(0.9 * sample_number)]
    test_idx = shuffled_idx[int(0.9 * sample_number):]

    origin_data.train_mask = sample_mask(train_idx, sample_number)
    origin_data.val_mask = sample_mask(val_idx, sample_number)
    origin_data.test_mask = sample_mask(test_idx, sample_number)

    test_mask = origin_data.test_mask
    train_mask = origin_data.train_mask
    val_mask = origin_data.val_mask

    origin_data.to(device)
    data = HeteroData().to(device)
    data.x = origin_data.x

    if args.task == 'stance':
        args.out_dim = 3
        data.y = origin_data.y1
    else:
        args.out_dim = 2
        data.y = origin_data.y2

    data.edge_index = {}
    if len(args.relation_select) > 0:
        for index in args.relation_select:
            data.edge_index[("user", relation_dict[index], "user")] = origin_data.edge_index[:, origin_data.edge_type==index]
            print('{}'.format(relation_dict[index]), end='  ')
    print('\n')

    data.train_idx = torch.from_numpy(np.array([i for i in range(len(data.y.cpu()))])[origin_data.train_mask.cpu()])
    data.valid_idx = torch.from_numpy(np.array([i for i in range(len(data.y.cpu()))])[origin_data.val_mask.cpu()])
    data.test_idx = torch.from_numpy(np.array([i for i in range(len(data.y.cpu()))])[origin_data.test_mask.cpu()])

    model = HGT(args, data.edge_index.keys()).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    def train(epoch):
        model.train()
        output = model(data.x, data.edge_index)
        loss_train = loss(output[origin_data.train_mask], data.y[origin_data.train_mask])
        out = output.max(1)[1].to('cpu').detach().numpy()
        label = data.y.to('cpu').detach().numpy()
        acc_train = accuracy_score(out[train_mask], label[train_mask])
        acc_val = accuracy_score(out[val_mask], label[val_mask])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'acc_val: {:.4f}'.format(acc_val.item()), )
        return acc_train, loss_train


    def test():
        model.eval()
        output = model(data.x, data.edge_index)
        loss_test = loss(output[origin_data.test_mask], data.y[origin_data.test_mask])
        output = output.max(1)[1].to('cpu').detach().numpy()
        label = data.y.to('cpu').detach().numpy()
        acc_test = accuracy_score(label[test_mask], output[test_mask])
        f1 = f1_score(label[test_mask], output[test_mask], average='macro')
        precision = precision_score(label[test_mask], output[test_mask], average='macro')
        recall = recall_score(label[test_mask], output[test_mask], average='macro')
        return acc_test, loss_test, f1, precision, recall


    max_acc = 0
    for epoch in range(args.epochs):
        train(epoch)
        acc_test, loss_test, f1, precision, recall = test()
        if acc_test > max_acc:
            max_acc = acc_test
            max_epoch = epoch
            max_f1 = f1
            max_precision = precision
            max_recall = recall

    print("Test set results:",
          "epoch= {:}".format(max_epoch),
          "test_accuracy= {:.4f}".format(max_acc),
          "precision= {:.4f}".format(max_precision),
          "recall= {:.4f}".format(max_recall),
          "f1_score= {:.4f}".format(max_f1)
          )
    return max_acc, max_precision, max_recall, max_f1



if __name__ == "__main__":

    t = time.time()
    acc_list =[]
    precision_list = []
    recall_list = []
    f1_list = []

    for i, seed in enumerate(args.random_seed):
        print('traning {}th model\n'.format(i+1))
        acc, precision, recall, f1 = main(seed)
        acc_list.append(acc*100)
        precision_list.append(precision*100)
        recall_list.append(recall*100)
        f1_list.append(f1*100)

    print('acc:       {:.2f} + {:.2f}'.format(np.array(acc_list).mean(), np.std(acc_list)))
    print('precision: {:.2f} + {:.2f}'.format(np.array(precision_list).mean(), np.std(precision_list)))
    print('recall:    {:.2f} + {:.2f}'.format(np.array(recall_list).mean(), np.std(recall_list)))
    print('f1:        {:.2f} + {:.2f}'.format(np.array(f1_list).mean(), np.std(f1_list)))
    print('total time:', time.time() - t)
