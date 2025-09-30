import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import BCEWithLogitsLoss
from .model import HMSG
from models_GAT_GAT.layers import StructuralAttentionLayer, TemporalAttentionLayer, GCNLayer
from utils.utilities import fixed_unigram_candidate_sampler

class MulVDH(nn.Module):
    def __init__(self, args, meta_paths, in_size, aggre_type):
        super(MulVDH, self).__init__()
        self.args = args
        if args.window < 0:
            self.num_time_steps = args.time_steps
        else:
            self.num_time_steps = min(args.time_steps, args.window + 1)  # window = 0 => only self.
        self.meta_paths = meta_paths
        self.in_size = in_size
        self.aggre_type = aggre_type
        self.hidden_units = args.hidden_units
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        self.temporal_drop = args.temporal_drop
        self.embModel, self.temporal_attn = self.build_model()
        self.bceloss = BCEWithLogitsLoss()

    def forward(self, args, hg_user_item, feat_hmsg):

        emb_layers_output = []
        for i in range(len(hg_user_item)):
            emb_layers_output.append(self.embModel([hg_user_item[i], feat_hmsg[i]]))

        emb_layers_outputs = emb_layers_output
        emb_layers_outputs_u = []
        emb_layers_outputs_i = []
        user_num = args.user_num


        for out in emb_layers_outputs:
            u_feat = out['user'].view(out['user'].shape[0], 1, out['user'].shape[-1])
            i_feat = out['item'].view(out['item'].shape[0], 1, out['item'].shape[-1])

            emb_layers_outputs_u.append(u_feat)
            emb_layers_outputs_i.append(i_feat)

        # padding outputs along with Ni
        # user
        maximum_user_node_num = emb_layers_outputs[-1]['user'].shape[0]
        out_user_dim = emb_layers_outputs[-1]['user'].shape[-1]
        emb_layers_outputs_user_padded = []
        for out in emb_layers_outputs_u:
            zero_padding = torch.zeros(maximum_user_node_num - out.shape[0], 1, out_user_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            emb_layers_outputs_user_padded.append(padded)
        emb_layers_outputs_user_padded = torch.cat(emb_layers_outputs_user_padded, dim=1)

        # item
        maximum_item_node_num = emb_layers_outputs[-1]['item'].shape[0]
        out_item_dim = emb_layers_outputs[-1]['item'].shape[-1]
        emb_layers_outputs_item_padded = []
        for out in emb_layers_outputs_i:
            zero_padding = torch.zeros(maximum_item_node_num - out.shape[0], 1, out_item_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            emb_layers_outputs_item_padded.append(padded)
        emb_layers_outputs_item_padded = torch.cat(emb_layers_outputs_item_padded, dim=1)

        # Temporal Attention forward
        # user
        temporal_user_out = self.temporal_attn(emb_layers_outputs_user_padded)
        # item
        temporal_item_out = self.temporal_attn(emb_layers_outputs_item_padded)
        out = torch.cat([temporal_user_out, temporal_item_out], dim=0)

        return out


    def build_model(self):
        # 1: graph embedding Layers
        emb_layers = nn.Sequential()
        layer = HMSG(meta_paths=self.meta_paths,
                     in_size=self.in_size,
                     hidden_size=self.hidden_units,
                     aggre_type=self.aggre_type,
                     num_heads=self.num_heads,
                     dropout=self.dropout)
        emb_layers.add_module(name="emb_layer_{}".format(0), module=layer)

        # 2: Temporal Attention Layers
        input_dim = self.hidden_units * self.num_heads #manual set, equal to
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)

        return emb_layers, temporal_attention_layers



    def get_loss(self, args, edge_sample_pos_4snap, edge_sample_neg_4snap, hg_user_item, feat_hmsg):

        emb = self.forward(args, hg_user_item, feat_hmsg)
        final_user_emb, final_item_emb = emb.split([args.user_num, emb.shape[0] - args.user_num], dim=0)
        user_num = args.user_num
        self.graph_loss = 0
        for t in range(self.num_time_steps - 1):
            user_embed = final_user_emb[:, t, :]
            item_embed = final_item_emb[:, t, :]

            # print(type(edge_sample_pos_4snap[t]))
            #
            # print(type(edge_sample_pos_4snap[t][0]))
            # a = np.array(edge_sample_pos_4snap[t])
            # print(type(a))
            # print(a)
            # print(a[:, 0])
            # b = torch.tensor(edge_sample_pos_4snap[t])
            # print(type(b))
            # print(b)
            # print(b[:, 0])


            pos_embedding_user = user_embed[np.array(edge_sample_pos_4snap[t])[:, 0]].view(-1, user_embed.shape[1])
            pos_embedding_item = item_embed[np.array(edge_sample_pos_4snap[t])[:, 1]].view(-1, item_embed.shape[1])
            neg_embedding_user = user_embed[np.array(edge_sample_neg_4snap[t])[:, 0]].view(-1, user_embed.shape[1])
            neg_embedding_item = item_embed[np.array(edge_sample_neg_4snap[t])[:, 1]].view(-1, item_embed.shape[1])
            # pos_out = torch.bmm(pos_embedding_user, pos_embedding_item)  # .view(-1, 5)
            # neg_out = -torch.bmm(neg_embedding_user, neg_embedding_item)  # .view(-1, 5)
            # train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
            pos_score = torch.sum(pos_embedding_user * pos_embedding_item, dim=1)
            neg_score = -torch.sum(neg_embedding_user * neg_embedding_item, dim=1)
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score))
            graphloss = pos_loss + self.args.neg_weight * neg_loss

            self.graph_loss += graphloss

        return self.graph_loss





