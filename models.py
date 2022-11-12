import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import HGTConv, RGCNConv
from layer import RGTLayer, SimpleHGNLayer
import torch.nn.functional as F


class BotRGCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(BotRGCN, self).__init__()
        self.dropout = dropout

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(768, int(hidden_dimension*3/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(10, int(hidden_dimension/8)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(10, int(hidden_dimension/8)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.rgcn = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        t = self.linear_relu_tweet(feature[:, -768:].to(torch.float32))
        n = self.linear_relu_num_prop(feature[:, [4,6,7,8,10,11,12,13,14,15]].to(torch.float32))
        b = self.linear_relu_cat_prop(feature[:, [1,2,3,5,9,16,17,18,19,20]].to(torch.float32))
        x = torch.cat((t, n, b), dim=1)
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class RGCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(RGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.rgcn1 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)
        self.rgcn2 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn1(x, edge_index, edge_type)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class GAT(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.gat1 = GATConv(hidden_dimension, int(hidden_dimension / 8), heads=8)
        self.gat2 = GATConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.gat1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class GCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.gcn1 = GCNConv(hidden_dimension, hidden_dimension)
        self.gcn2 = GCNConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.gcn1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class SAGE(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(SAGE, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.sage1 = SAGEConv(hidden_dimension, hidden_dimension)
        self.sage2 = SAGEConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.sage1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class HGT(nn.Module):
    def __init__(self, args, relation_list):
        super(HGT, self).__init__()

        self.relation_list = list(relation_list)
        self.linear1 = nn.Linear(args.features_num, args.hidden_dimension)

        self.HGT_layer1 = HGTConv(in_channels=args.hidden_dimension, out_channels=args.hidden_dimension,
                                  metadata=(['user'], self.relation_list))
        self.HGT_layer2 = HGTConv(in_channels=args.hidden_dimension, out_channels=args.linear_channels,
                                  metadata=(['user'], self.relation_list))
        self.out1 = torch.nn.Linear(args.linear_channels, args.out_channel)
        self.out2 = torch.nn.Linear(args.out_channel, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, features, edge_index_dict):

        user_features = self.drop(self.ReLU(self.linear1(features)))
        x_dict = {"user": user_features}
        x_dict = self.HGT_layer1(x_dict, edge_index_dict)
        x_dict = self.HGT_layer1(x_dict, edge_index_dict)
        user_features = self.ReLU(self.out1(x_dict["user"]))
        x = self.out2(user_features)

        return x


class RGT(nn.Module):
    def __init__(self, args):
        super(RGT, self).__init__()

        self.linear1 = nn.Linear(args.features_num, args.hidden_dimension)
        self.RGT_layer1 = RGTLayer(num_edge_type=len(args.relation_select), in_channel=args.hidden_dimension, out_channel=args.hidden_dimension,
                                   trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)
        # self.RGT_layer2 = RGTLayer(num_edge_type=len(args.relation_select), in_channel=args.hidden_dimension, out_channel=args.hidden_dimension, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)

        self.out1 = torch.nn.Linear(args.hidden_dimension, args.out_channel)
        self.out2 = torch.nn.Linear(args.out_channel, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, features, edge_index, edge_type):

        user_features = self.drop(self.ReLU(self.linear1(features)))
        user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
        user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
        user_features = self.drop(self.ReLU(self.out1(user_features)))
        x = self.out2(user_features)

        return x



class SHGN(nn.Module):
    def __init__(self, args):
        super(SHGN, self).__init__()

        self.linear1 = nn.Linear(args.features_num, args.hidden_dimension)
        self.HGN_layer1 = SimpleHGNLayer(num_edge_type=args.num_edge_type, in_channels=args.hidden_dimension,
                                         out_channels=args.hidden_dimension, rel_dim=args.rel_dim, beta=args.beta)
        self.HGN_layer2 = SimpleHGNLayer(num_edge_type=args.num_edge_type, in_channels=args.hidden_dimension,
                                         out_channels=args.linear_channels, rel_dim=args.rel_dim, beta=args.beta,
                                         final_layer=True)

        self.out1 = torch.nn.Linear(args.linear_channels, args.out_channel)
        self.out2 = torch.nn.Linear(args.out_channel, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.ReLU = nn.LeakyReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, feature, edge_index, edge_type):

        user_features = self.drop(self.ReLU(self.linear1(feature)))
        user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
        user_features, _ = self.HGN_layer1(user_features, edge_index, edge_type, alpha)
        user_features = self.drop(self.ReLU(self.out1(user_features)))
        x = self.out2(user_features)
        return x