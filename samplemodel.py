import torch
from torch import nn
from torch_geometric.nn import RGCNConv,GCNConv,GATConv
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from layer import SimpleHGNLayer, RGTLayer
import pytorch_lightning as pl


class SHGN(pl.LightningModule):
    def __init__(self, args):
        super(SHGN, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.test_batch_size = args.test_batch_size
        self.features_num = args.features_num
        self.pred_test = []
        self.pred_test_prob = []
        self.label_test = []

        self.linear1 = nn.Linear(args.features_num, args.linear_channels)
        self.HGN_layer1 = SimpleHGNLayer(num_edge_type=args.relation_num,
                                         in_channels=args.linear_channels,
                                         out_channels=args.linear_channels,
                                         rel_dim=args.rel_dim,
                                         beta=args.beta)
        self.HGN_layer2 = SimpleHGNLayer(num_edge_type=args.relation_num,
                                         in_channels=args.linear_channels,
                                         out_channels=args.out_channel,
                                         rel_dim=args.rel_dim,
                                         beta=args.beta,
                                         final_layer=True)

        self.out1 = torch.nn.Linear(args.out_channel, 64)
        self.out2 = torch.nn.Linear(64, args.out_dim)

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

    def training_step(self, train_batch, batch_idx):
        user_features = train_batch.x[:, :self.features_num]
        label = train_batch.y

        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)

        user_features = self.drop(self.ReLU(self.linear1(user_features)))
        user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
        user_features, _ = self.HGN_layer2(user_features, edge_index, edge_type, alpha)

        user_features = self.drop(self.ReLU(self.out1(user_features)))
        pred = self.out2(user_features)
        train_pred = pred[train_batch.train_mask][:train_batch.batch_size]
        train_label = label[train_batch.train_mask][:train_batch.batch_size]
        loss = self.CELoss(train_pred, train_label)

        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = val_batch.x[:, :self.features_num]
            label = val_batch.y

            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)

            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
            user_features, _ = self.HGN_layer2(user_features, edge_index, edge_type, alpha)

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features)
            pred_binary = torch.argmax(pred, dim=1)
            val_pred_binary = pred_binary[val_batch.val_mask][:val_batch.batch_size]
            val_label = label[val_batch.val_mask][:val_batch.batch_size]

            acc = accuracy_score(val_label.cpu(), val_pred_binary.cpu())
            f1 = f1_score(val_label.cpu(), val_pred_binary.cpu(), average='macro')

            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = test_batch.x
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)

            user_features = self.drop(self.ReLU(self.linear1(user_features)))
            user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
            user_features, _ = self.HGN_layer2(user_features, edge_index, edge_type, alpha)

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features)
            pred_binary = torch.argmax(pred, dim=1)
            test_pred = pred[test_batch.test_mask][:test_batch.batch_size]
            test_pred_binary = pred_binary[test_batch.test_mask][:test_batch.batch_size]
            test_label = label[test_batch.test_mask][:test_batch.batch_size]

            self.pred_test.append(test_pred_binary.squeeze().cpu())
            self.pred_test_prob.append(test_pred[:, 1].squeeze().cpu())
            self.label_test.append(test_label.squeeze().cpu())

            pred_test = torch.cat(self.pred_test).cpu()
            pred_test_prob = torch.cat(self.pred_test_prob).cpu()
            label_test = torch.cat(self.label_test).cpu()

            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1 = f1_score(label_test.cpu(), pred_test.cpu(), average='macro')
            precision = precision_score(label_test.cpu(), pred_test.cpu(), average='macro')
            recall = recall_score(label_test.cpu(), pred_test.cpu(), average='macro')

            print("\nacc: {}".format(acc),
                  "f1: {}".format(f1),
                  "precision: {}".format(precision),
                  "recall: {}".format(recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }



class BotRGCN(pl.LightningModule):
    def __init__(self, args):
        super(BotRGCN, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.test_batch_size = args.test_batch_size
        self.features_num = args.features_num
        self.pred_test = []
        self.pred_test_prob = []
        self.label_test = []

        self.linear_relu_input = nn.Sequential(
            nn.Linear(args.features_num, args.linear_channels),
            nn.LeakyReLU()
        )
        self.rgcn1 = RGCNConv(args.linear_channels, args.linear_channels, num_relations=args.relation_num)
        self.rgcn2 = RGCNConv(args.linear_channels, args.out_channel, num_relations=args.relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(args.out_channel, 64),
            nn.LeakyReLU()
        )

        self.out2 = torch.nn.Linear(64, args.out_dim)

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

    def training_step(self, train_batch, batch_idx):
        user_features = train_batch.x[:, :self.features_num]
        label = train_batch.y

        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)

        user_features = self.linear_relu_input(user_features)
        user_features = self.drop(self.rgcn1(user_features, edge_index, edge_type))
        user_features = self.rgcn2(user_features, edge_index, edge_type)
        user_features = self.linear_relu_output1(user_features)
        pred = self.out2(user_features)
        train_pred = pred[train_batch.train_mask][:train_batch.batch_size]
        train_label = label[train_batch.train_mask][:train_batch.batch_size]
        loss = self.CELoss(train_pred, train_label)

        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = val_batch.x[:, :self.features_num]
            label = val_batch.y

            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.rgcn1(user_features, edge_index, edge_type))
            user_features = self.rgcn2(user_features, edge_index, edge_type)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            val_pred_binary = pred_binary[val_batch.val_mask][:val_batch.batch_size]
            val_label = label[val_batch.val_mask][:val_batch.batch_size]

            acc = accuracy_score(val_label.cpu(), val_pred_binary.cpu())
            f1 = f1_score(val_label.cpu(), val_pred_binary.cpu(), average='macro')

            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = test_batch.x
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.rgcn1(user_features, edge_index, edge_type))
            user_features = self.rgcn2(user_features, edge_index, edge_type)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            test_pred = pred[test_batch.test_mask][:test_batch.batch_size]
            test_pred_binary = pred_binary[test_batch.test_mask][:test_batch.batch_size]
            test_label = label[test_batch.test_mask][:test_batch.batch_size]

            self.pred_test.append(test_pred_binary.squeeze().cpu())
            self.pred_test_prob.append(test_pred[:, 1].squeeze().cpu())
            self.label_test.append(test_label.squeeze().cpu())

            pred_test = torch.cat(self.pred_test).cpu()
            pred_test_prob = torch.cat(self.pred_test_prob).cpu()
            label_test = torch.cat(self.label_test).cpu()

            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1 = f1_score(label_test.cpu(), pred_test.cpu(), average='macro')
            precision = precision_score(label_test.cpu(), pred_test.cpu(), average='macro')
            recall = recall_score(label_test.cpu(), pred_test.cpu(), average='macro')

            print("\nacc: {}".format(acc),
                "f1: {}".format(f1),
                "precision: {}".format(precision),
                "recall: {}".format(recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }



class GAT(pl.LightningModule):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.test_batch_size = args.test_batch_size
        self.features_num = args.features_num
        self.pred_test = []
        self.pred_test_prob = []
        self.label_test = []

        self.linear_relu_input = nn.Sequential(
            nn.Linear(args.features_num, args.linear_channels),
            nn.LeakyReLU()
        )

        self.gat1 = GATConv(args.linear_channels, int(args.linear_channels / 4), heads=4)
        self.gat2 = GATConv(args.linear_channels, args.out_channel)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(args.out_channel, 64),
            nn.LeakyReLU()
        )

        self.out2 = torch.nn.Linear(64, args.out_dim)

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

    def training_step(self, train_batch, batch_idx):
        user_features = train_batch.x[:, :self.features_num]
        label = train_batch.y

        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)

        user_features = self.linear_relu_input(user_features)
        user_features = self.drop(self.gat1(user_features, edge_index))
        user_features = self.gat2(user_features, edge_index)
        user_features = self.drop(self.linear_relu_output1(user_features))
        pred = self.out2(user_features)
        train_pred = pred[train_batch.train_mask][:train_batch.batch_size]
        train_label = label[train_batch.train_mask][:train_batch.batch_size]
        loss = self.CELoss(train_pred, train_label)

        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = val_batch.x[:, :self.features_num]
            label = val_batch.y

            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.gat1(user_features, edge_index))
            user_features = self.gat2(user_features, edge_index)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            val_pred_binary = pred_binary[val_batch.val_mask][:val_batch.batch_size]
            val_label = label[val_batch.val_mask][:val_batch.batch_size]

            acc = accuracy_score(val_label.cpu(), val_pred_binary.cpu())
            f1 = f1_score(val_label.cpu(), val_pred_binary.cpu(), average='macro')

            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = test_batch.x
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.gat1(user_features, edge_index))
            user_features = self.gat2(user_features, edge_index)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            test_pred = pred[test_batch.test_mask][:test_batch.batch_size]
            test_pred_binary = pred_binary[test_batch.test_mask][:test_batch.batch_size]
            test_label = label[test_batch.test_mask][:test_batch.batch_size]

            self.pred_test.append(test_pred_binary.squeeze().cpu())
            self.pred_test_prob.append(test_pred[:, 1].squeeze().cpu())
            self.label_test.append(test_label.squeeze().cpu())

            pred_test = torch.cat(self.pred_test).cpu()
            pred_test_prob = torch.cat(self.pred_test_prob).cpu()
            label_test = torch.cat(self.label_test).cpu()

            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1 = f1_score(label_test.cpu(), pred_test.cpu(), average='macro')
            precision = precision_score(label_test.cpu(), pred_test.cpu(), average='macro')
            recall = recall_score(label_test.cpu(), pred_test.cpu(), average='macro')

            print("\nacc: {}".format(acc),
                  "f1: {}".format(f1),
                  "precision: {}".format(precision),
                  "recall: {}".format(recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }



class GCN(pl.LightningModule):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.test_batch_size = args.test_batch_size
        self.features_num = args.features_num
        self.pred_test = []
        self.pred_test_prob = []
        self.label_test = []

        self.linear_relu_input = nn.Sequential(
            nn.Linear(args.features_num, args.linear_channels),
            nn.LeakyReLU()
        )
        self.gcn1 = GCNConv(args.linear_channels, args.linear_channels)
        self.gcn2 = GCNConv(args.linear_channels, args.out_channel)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(args.out_channel, 64),
            nn.LeakyReLU()
        )

        self.out2 = torch.nn.Linear(64, args.out_dim)

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

    def training_step(self, train_batch, batch_idx):
        user_features = train_batch.x[:, :self.features_num]
        label = train_batch.y

        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)

        user_features = self.linear_relu_input(user_features)
        user_features = self.drop(self.gcn1(user_features, edge_index))
        user_features = self.gcn2(user_features, edge_index,)
        user_features = self.drop(self.linear_relu_output1(user_features))
        pred = self.out2(user_features)
        train_pred = pred[train_batch.train_mask][:train_batch.batch_size]
        train_label = label[train_batch.train_mask][:train_batch.batch_size]
        loss = self.CELoss(train_pred, train_label)

        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = val_batch.x[:, :self.features_num]
            label = val_batch.y

            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.gcn1(user_features, edge_index))
            user_features = self.gcn2(user_features, edge_index)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            val_pred_binary = pred_binary[val_batch.val_mask][:val_batch.batch_size]
            val_label = label[val_batch.val_mask][:val_batch.batch_size]

            acc = accuracy_score(val_label.cpu(), val_pred_binary.cpu())
            f1 = f1_score(val_label.cpu(), val_pred_binary.cpu(), average='macro')

            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = test_batch.x
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.drop(self.gcn1(user_features, edge_index))
            user_features = self.gcn2(user_features, edge_index)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            test_pred = pred[test_batch.test_mask][:test_batch.batch_size]
            test_pred_binary = pred_binary[test_batch.test_mask][:test_batch.batch_size]
            test_label = label[test_batch.test_mask][:test_batch.batch_size]

            self.pred_test.append(test_pred_binary.squeeze().cpu())
            self.pred_test_prob.append(test_pred[:, 1].squeeze().cpu())
            self.label_test.append(test_label.squeeze().cpu())

            pred_test = torch.cat(self.pred_test).cpu()
            pred_test_prob = torch.cat(self.pred_test_prob).cpu()
            label_test = torch.cat(self.label_test).cpu()

            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1 = f1_score(label_test.cpu(), pred_test.cpu(), average='macro')
            precision = precision_score(label_test.cpu(), pred_test.cpu(), average='macro')
            recall = recall_score(label_test.cpu(), pred_test.cpu(), average='macro')


            print("\nacc: {}".format(acc),
                  "f1: {}".format(f1),
                  "precision: {}".format(precision),
                  "recall: {}".format(recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }


class RGT(pl.LightningModule):
    def __init__(self, args):
        super(RGT, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.test_batch_size = args.test_batch_size
        self.features_num = args.features_num
        self.pred_test = []
        self.pred_test_prob = []
        self.label_test = []

        self.linear_relu_input = nn.Sequential(
            nn.Linear(args.features_num, args.linear_channels),
            nn.LeakyReLU()
        )
        self.RGT_layer1 = RGTLayer(num_edge_type=args.relation_num,
                                   in_channel=args.linear_channels,
                                   trans_heads=args.trans_head,
                                   semantic_head=args.semantic_head,
                                   out_channel=args.linear_channels,
                                   dropout=args.dropout)
        self.RGT_layer2 = RGTLayer(num_edge_type=args.relation_num,
                                   in_channel=args.linear_channels,
                                   trans_heads=args.trans_head,
                                   semantic_head=args.semantic_head,
                                   out_channel=args.linear_channels,
                                   dropout=args.dropout)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(args.linear_channels, 64),
            nn.LeakyReLU()
        )

        self.out2 = torch.nn.Linear(64, args.out_dim)

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

    def training_step(self, train_batch, batch_idx):
        user_features = train_batch.x[:, :self.features_num]
        label = train_batch.y

        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)

        user_features = self.linear_relu_input(user_features)
        user_features = self.RGT_layer1(user_features, edge_index, edge_type)
        user_features = self.RGT_layer2(user_features, edge_index, edge_type)
        user_features = self.drop(self.linear_relu_output1(user_features))
        pred = self.out2(user_features)
        train_pred = pred[train_batch.train_mask][:train_batch.batch_size]
        train_label = label[train_batch.train_mask][:train_batch.batch_size]
        loss = self.CELoss(train_pred, train_label)

        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = val_batch.x[:, :self.features_num]
            label = val_batch.y

            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.RGT_layer1(user_features, edge_index, edge_type)
            user_features = self.RGT_layer2(user_features, edge_index, edge_type)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            val_pred_binary = pred_binary[val_batch.val_mask][:val_batch.batch_size]
            val_label = label[val_batch.val_mask][:val_batch.batch_size]

            acc = accuracy_score(val_label.cpu(), val_pred_binary.cpu())
            f1 = f1_score(val_label.cpu(), val_pred_binary.cpu(), average='macro')

            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_features = test_batch.x
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)

            user_features = self.linear_relu_input(user_features)
            user_features = self.RGT_layer1(user_features, edge_index, edge_type)
            user_features = self.RGT_layer2(user_features, edge_index, edge_type)
            user_features = self.drop(self.linear_relu_output1(user_features))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            test_pred = pred[test_batch.test_mask][:test_batch.batch_size]
            test_pred_binary = pred_binary[test_batch.test_mask][:test_batch.batch_size]
            test_label = label[test_batch.test_mask][:test_batch.batch_size]

            self.pred_test.append(test_pred_binary.squeeze().cpu())
            self.pred_test_prob.append(test_pred[:, 1].squeeze().cpu())
            self.label_test.append(test_label.squeeze().cpu())

            pred_test = torch.cat(self.pred_test).cpu()
            pred_test_prob = torch.cat(self.pred_test_prob).cpu()
            label_test = torch.cat(self.label_test).cpu()

            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1 = f1_score(label_test.cpu(), pred_test.cpu(), average='macro')
            precision = precision_score(label_test.cpu(), pred_test.cpu(), average='macro')
            recall = recall_score(label_test.cpu(), pred_test.cpu(), average='macro')


            print("\nacc: {}".format(acc),
                  "f1: {}".format(f1),
                  "precision: {}".format(precision),
                  "recall: {}".format(recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }