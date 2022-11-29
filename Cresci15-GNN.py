import torch
import torch.nn as nn
import argparse
import time
from Dataset import Cresci15
from models import RGCN, GAT, GCN, SAGE, BotRGCN
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser()
parser.add_argument('--relation_select', type=int, default=[0,1], nargs='+', help='selection of relations in the graph (0 1)')
parser.add_argument('--model', type=str, default='GCN', help='GCN, GAT, GraphSage, RGCN, BotRGCN')
parser.add_argument('--hidden_dimension', type=int, default=256, help='number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.3, help='number of hidden units.')
parser.add_argument('--epochs', type=int, default=200, help='training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay for optimizer.')
args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)


def main():
    dataset = Cresci15('./Dataset/Cresci-15')
    data = dataset[0]

    test_mask = data.test_mask
    train_mask = data.train_mask
    val_mask = data.val_mask
    out_dim = 2
    data = data.to(device)
    embedding_size = data.x.shape[1]
    relation_num = len(args.relation_select)

    index_select_list = (data.edge_type == 100)
    relation_dict = {
        0:'followers',
        1:'friends'
    }


    print('relation used:', end=' ')
    for features_index in args.relation_select:
            index_select_list = index_select_list + (features_index == data.edge_type)
            print('{}'.format(relation_dict[features_index]), end='  ')
    edge_index = data.edge_index[:, index_select_list]
    edge_type = data.edge_type[index_select_list]


    if args.model == 'RGCN':
        model = RGCN(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'GCN':
        model = GCN(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'GAT':
        model = GAT(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'GraphSage':
        model = SAGE(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'BotRGCN':
        model = BotRGCN(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)

    def train(epoch):
        model.train()
        output = model(data.x, edge_index, edge_type)
        loss_train = loss(output[data.train_mask], data.y[data.train_mask])
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
        return acc_val


    def test():
        model.eval()
        output = model(data.x, edge_index, edge_type)
        loss_test = loss(output[data.test_mask], data.y[data.test_mask])
        out = output.max(1)[1].to('cpu').detach().numpy()
        label = data.y.to('cpu').detach().numpy()
        acc_test = accuracy_score(out[test_mask], label[test_mask])
        f1 = f1_score(out[test_mask], label[test_mask], average='macro')
        precision = precision_score(out[test_mask], label[test_mask], average='macro')
        recall = recall_score(out[test_mask], label[test_mask], average='macro')
        return acc_test, loss_test, f1, precision, recall

    model.apply(init_weights)

    max_val_acc = 0
    for epoch in range(args.epochs):
        acc_val = train(epoch)
        acc_test, loss_test, f1, precision, recall = test()
        if acc_val > max_val_acc:
            max_val_acc = acc_val
            max_acc = acc_test
            max_epoch = epoch + 1
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


if __name__ == "__main__":

    t = time.time()
    main()
    print('total time:', time.time() - t)



