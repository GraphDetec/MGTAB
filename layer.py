import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class SimpleHGNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edge_type, rel_dim, beta=None, final_layer=False):
        super(SimpleHGNLayer, self).__init__(aggr = "add", node_dim=0)
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.W_r = torch.nn.Linear(rel_dim, out_channels, bias=False)
        self.a = torch.nn.Linear(3*out_channels, 1, bias=False)
        self.W_res = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.rel_emb = torch.nn.Embedding(num_edge_type, rel_dim)
        self.beta = beta
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.ELU = torch.nn.ELU()
        self.final = final_layer
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                    
    def forward(self, x, edge_index, edge_type, pre_alpha=None):
        
        node_emb = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, pre_alpha=pre_alpha)
        output = node_emb + self.W_res(x)
        output = self.ELU(output)
        if self.final:
            output = F.normalize(output, dim=1)
            
        return output, self.alpha.detach()
      
    def message(self, x_i, x_j, edge_type, pre_alpha, index, ptr, size_i):
        out = self.W(x_j)
        rel_emb = self.rel_emb(edge_type)
        alpha = self.leaky_relu(self.a(torch.cat((self.W(x_i), self.W(x_j), self.W_r(rel_emb)), dim=1)))
        alpha = softmax(alpha, index, ptr, size_i)
        if pre_alpha is not None and self.beta is not None:
            self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
        else:
            self.alpha = alpha
        out = out * alpha.view(-1,1)
        return out

    def update(self, aggr_out):
        return aggr_out


def masked_edge_index(edge_index, edge_mask):
    return edge_index[:, edge_mask]


class SemanticAttention(torch.nn.Module):
    def __init__(self, in_channel, num_head, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.num_head = num_head
        self.att_layers = torch.nn.ModuleList()
        # multi-head attention
        for i in range(num_head):
            self.att_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(in_channel, hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_size, 1, bias=False))
            )

    def forward(self, z):
        w = self.att_layers[0](z).mean(0)
        beta = torch.softmax(w, dim=0)

        beta = beta.expand((z.shape[0],) + beta.shape)
        output = (beta * z).sum(1)

        for i in range(1, self.num_head):
            w = self.att_layers[i](z).mean(0)
            beta = torch.softmax(w, dim=0)

            beta = beta.expand((z.shape[0],) + beta.shape)
            temp = (beta * z).sum(1)
            output += temp

        return output / self.num_head


class RGTLayer(torch.nn.Module):
    def __init__(self, num_edge_type, in_channel, out_channel, trans_heads, semantic_head, dropout):
        super(RGTLayer, self).__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_channel + out_channel, in_channel),
            torch.nn.Sigmoid()
        )

        self.activation = torch.nn.ELU()
        self.transformer_list = torch.nn.ModuleList()
        for i in range(int(num_edge_type)):
            self.transformer_list.append(
                TransformerConv(in_channels=in_channel, out_channels=out_channel, heads=trans_heads, dropout=dropout,
                                concat=False))

        self.num_edge_type = num_edge_type
        self.semantic_attention = SemanticAttention(in_channel=out_channel, num_head=semantic_head)

    def forward(self, features, edge_index, edge_type):
        r"""
        feature: input node features
        edge_index: all edge index, shape (2, num_edges)
        edge_type: same as RGCNconv in torch_geometric
        num_rel: number of relations
        beta: return cross relation attention weight
        agg: aggregation type across relation embedding
        """

        edge_index_list = []
        for i in range(self.num_edge_type):
            tmp = masked_edge_index(edge_index, edge_type == i)
            edge_index_list.append(tmp)

        u = self.transformer_list[0](features, edge_index_list[0].squeeze(0)).flatten(1)  # .unsqueeze(1)
        a = self.gate(torch.cat((u, features), dim=1))

        semantic_embeddings = (torch.mul(torch.tanh(u), a) + torch.mul(features, (1 - a))).unsqueeze(1)

        for i in range(1, len(edge_index_list)):
            u = self.transformer_list[i](features, edge_index_list[i].squeeze(0)).flatten(1)
            a = self.gate(torch.cat((u, features), dim=1))
            output = torch.mul(torch.tanh(u), a) + torch.mul(features, (1 - a))
            semantic_embeddings = torch.cat((semantic_embeddings, output.unsqueeze(1)), dim=1)

            return self.semantic_attention(semantic_embeddings)
