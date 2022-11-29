import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import pytorch_lightning as pl
import torch
import argparse
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from Dataset import MGTABlarge
from torch_geometric.loader import NeighborLoader
from samplemodel import BotRGCN, GCN, GAT, SHGN, RGT
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='stance', help='detection task of stance or bot')
parser.add_argument('--relation_select', type=int, default=[0,1], nargs='+', help='Selection of relations in the graph (0-6).')
parser.add_argument('--model', type=str, default='RGT', help='BotRGCN, GCN, GAT, SHGN')
parser.add_argument('--GPU_num', type=int, default=1, help='numbers of GPUs used')
parser.add_argument("--linear_channels", type=int, default=256, help="linear channels")
parser.add_argument("--out_channel", type=int, default=128, help="description channel")
parser.add_argument("--dropout", type=float, default=0.3, help="description channel")
parser.add_argument("--batch_size", type=int, default=512, help="description channel")
parser.add_argument("--epochs", type=int, default=200, help="training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="description channel")
parser.add_argument("--l2_reg", type=float, default=5e-4, help="description channel")
parser.add_argument("--random_seed", type=int, default=None, help="random")
parser.add_argument("--rel_dim", type=int, default=200, help="catgorical features")
parser.add_argument("--beta", type=float, default=0.05, help="description channel")
parser.add_argument("--trans_head", type=int, default=4, help="description channel")
parser.add_argument("--semantic_head", type=int, default=4, help="description channel")


if __name__ == "__main__":
    global args, pred_test, pred_test_prob, label_test
    args = parser.parse_args()

    if args.random_seed != None:
        pl.seed_everything(args.random_seed)


    dataset = MGTABlarge('Dataset/MGTAB-large')
    data = dataset[0]
    args.features_num = data.x.shape[1]
    if args.task == 'stance':
        args.out_dim = 3
        data.y = data.y1
    else:
        args.out_dim = 2
        data.y = data.y2

    data.y = torch.cat([data.y, -1 * torch.ones(data.x.shape[0] - len(data.y))]).type(torch.int64)
    index_select_list = (data.edge_type == 100)
    relation_dict = {
        0: 'followers',
        1: 'friends',
        2: 'mention',
        3: 'reply',
        4: 'quoted',
        5: 'url',
        6: 'hashtag'
    }

    args.relation_num = len(args.relation_select)
    print('relation used:', end=' ')
    for features_index in args.relation_select:
        index_select_list = index_select_list + (features_index == data.edge_type)
        print('{}'.format(relation_dict[features_index]), end='  ')
    data.edge_index = data.edge_index[:, index_select_list]
    data.edge_type = data.edge_type[index_select_list]
    data.edge_weight = data.edge_weight[index_select_list]

    train_idx = torch.from_numpy(np.array([i for i in range(data.x.shape[0])])[data.train_mask])
    valid_idx = torch.from_numpy(np.array([i for i in range(data.x.shape[0])])[data.val_mask])
    test_idx = torch.from_numpy(np.array([i for i in range(data.x.shape[0])])[data.test_mask])

    args.test_batch_size = math.ceil(sum(data.test_mask))
    train_loader = NeighborLoader(data, num_neighbors=[15, 25], input_nodes=train_idx, batch_size=args.batch_size,
                                  shuffle=True)
    valid_loader = NeighborLoader(data, num_neighbors=[15, 25], input_nodes=valid_idx, batch_size=args.batch_size)
    test_loader = NeighborLoader(data, num_neighbors=[15, 25], input_nodes=test_idx, batch_size=args.test_batch_size)

    if args.model == 'BotRGCN':
        model = BotRGCN(args)
    elif args.model == 'GCN':
        model = GCN(args)
    elif args.model == 'GAT':
        model = GAT(args)
    elif args.model == 'SHGN':
        model = SHGN(args)
    elif args.model == 'RGT':
        model = RGT(args)

    trainer = pl.Trainer(gpus=args.GPU_num, num_nodes=1, max_epochs=args.epochs, precision=16, log_every_n_steps=1)
    trainer.fit(model, train_loader, valid_loader)
    trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=args.epochs, precision=16, log_every_n_steps=1)
    trainer.test(model, test_loader, verbose=True)
