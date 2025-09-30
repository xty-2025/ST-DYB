import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
import numpy as np

from models_GAT_fourier_and_diffusion.utils import SpaSeqNetLast, seq2str
from models_GAT_fourier_and_diffusion.misc import DummyArgs
from torch.fft import rfft, irfft, fft, ifft

from sklearn.metrics import f1_score, accuracy_score
from models_GAT_fourier_and_diffusion.misc import EarlyStopping, move_to, cal_metric
# from ablation_experiment.utils import COLLECTOR
from libwon.utils.collector import *
from tqdm import tqdm, trange
from models_GAT_fourier_and_diffusion.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionPooling(nn.Module):
    def __init__(self, K_len, hid_dim):
        super(SelfAttentionPooling, self).__init__()
        self.K_len = K_len
        self.hid_dim = hid_dim

        # 定义查询、键、值的映射
        self.query_layer = nn.Linear(K_len * hid_dim * 2, hid_dim)
        self.key_layer = nn.Linear(K_len * hid_dim * 2, hid_dim)
        self.value_layer = nn.Linear(K_len * hid_dim * 2, hid_dim)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, K_len)
        )

    def forward(self, x):
        """
        输入:
            x: 形状为 [Batch Size, K_len * hid_dim * 2]

        输出:
            output: 形状为 [Batch Size, K_len]
        """
        # 计算 Query, Key, Value
        queries = self.query_layer(x)  # [Batch Size, hid_dim]
        keys = self.key_layer(x)  # [Batch Size, hid_dim]
        values = self.value_layer(x)  # [Batch Size, hid_dim]

        # 自注意力得分计算 (缩放点积注意力机制)
        attention_scores = torch.bmm(queries.unsqueeze(1), keys.unsqueeze(2))  # [Batch Size, 1, 1]
        attention_scores = attention_scores / (self.hid_dim ** 0.5)  # 缩放
        attention_weights = F.softmax(attention_scores, dim=-1)  # 归一化为概率分布

        # 通过注意力权重加权求和
        weighted_sum = torch.bmm(attention_weights, values.unsqueeze(1))  # [Batch Size, 1, hid_dim]

        # 展平并通过输出层映射到最终维度
        weighted_sum = weighted_sum.view(weighted_sum.size(0), -1)  # [Batch Size, hid_dim]
        output = self.output_layer(weighted_sum)  # [Batch Size, K_len]
        return output



class SpecMask(nn.Module):
    def __init__(self, hid_dim, temporature, K_len) -> None:
        super().__init__()
        self.K_len = K_len
        #self.node_spec_map = nn.Sequential(nn.Linear(self.K_len * hid_dim * 2, hid_dim * 2), nn.ReLU(),
        #                                   nn.Linear(hid_dim * 2, self.K_len))  # N, K x d x 2 - > N x f

        self.node_spec_map = SelfAttentionPooling(K_len, hid_dim)
        self.temporature = temporature
        self.K_len = K_len
        self.hid_dim = hid_dim
        self.spec_lin = nn.Sequential(
            nn.Linear(K_len * hid_dim * 2, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim)
        )

    def forward(self, specs):
        # specs [N, T, d]
        # learn causal and spurious masks
        spec_real, spec_imag = specs.real, specs.imag  # [N, K, d]
        spec_real_imag = torch.stack([spec_real, spec_imag], dim=-1)  # [N, K, d, 2]
        node_choice = self.node_spec_map(
            spec_real_imag.view(-1, self.K_len * self.hid_dim * 2))  # [N, K * d * 2] -> [N, K]

        cmask_ = torch.sigmoid(node_choice / self.temporature)
        smask_ = torch.sigmoid(- node_choice / self.temporature)  # [N, K]
        if len(COLLECTOR.cache.get('cmask0', [])) == len(COLLECTOR.cache.get('loss', [])):
            COLLECTOR.add('cmask0', seq2str(cmask_[0].detach().cpu().numpy()))
            COLLECTOR.add('smask0', seq2str(smask_[0].detach().cpu().numpy()))

        cmask = cmask_.unsqueeze(-1).expand_as(spec_imag)
        smask = smask_.unsqueeze(-1).expand_as(spec_imag)

        # filter in the spectral domain
        c_spec_real = spec_real * cmask  # [N, K, d] * [N, K, d]
        c_spec_imag = spec_imag * cmask  # [N, K, d] * [N, K, d]

        s_spec_real = spec_real * smask  # [N, K, d] * [N, K, d]
        s_spec_imag = spec_imag * smask  # [N, K, d] * [N, K, d]

        c_spec = torch.cat([c_spec_real, c_spec_imag], dim=-1).flatten(-2, -1)
        s_spec = torch.cat([s_spec_real, s_spec_imag], dim=-1).flatten(-2, -1)

        c_spec = self.spec_lin(c_spec)  # [N, d]
        s_spec = self.spec_lin(s_spec)  # [N, d]

        return c_spec, s_spec


class TimeNet(torch.nn.Module):
    def __init__(self, args):
        super(TimeNet, self).__init__()
        in_dim = args.nfeat
        hid_dim = 2 * args.nhid
        out_dim = 2 * args.nhid
        num_layers = args.n_layers
        time_length = args.time_steps

        self.zc_zs_weight = nn.Parameter(torch.rand(1, requires_grad=True))

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.time_length = time_length
        self.spec_len = time_length
        self.K_len = 1 + self.spec_len // 2
        self.args = args
        self.len_train = self.time_length - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        self.len = time_length
        self.earlystop = EarlyStopping('val_auc', mode="max", patience=args.early_stop)
        self.device = args.device
        self.metric = args.metric
        self.args = args
        self.linear = nn.Linear(self.args.total_node * self.args.output_dim, hid_dim, bias=False)
        self.timenet_dim = list(map(int, args.temporal_layer_config.split(",")))



        # spatio model
        if args.backbone in 'GCN GIN GAT'.split():
            args.static_conv = args.backbone
            self.backbone = SpaSeqNetLast(args)
        else:
            raise NotImplementedError()

        # spectral model
        if args.spec_filt == "mask":
            self.spec_filt = SpecMask(hid_dim, args.temporature, self.K_len)
        else:
            raise NotImplementedError()

        # post gnn
        self.post_gnn = args.post_gnn
        # if args.post_gnn:
        #     self.nodeconv = NodeFormer(hid_dim, hid_dim, hid_dim, num_layers = args.post_gnn)

        # decoder
        self.cs_decoder = self.backbone.cs_decoder
        self.ss_decoder = self.backbone.ss_decoder

        self.ctype = args.ctype



    def forward(self, graph_feature):
        # dataset
        x = graph_feature # the shape should be 7*37791*128
        self.x = [x for _ in range(self.time_length)] if len(x.shape) <= 2 else x
        feature = self.x #16*610*32
        #cs, ss = self.get_final_emb(feature) #get_final_emb() need edge_list and feature, but now we have got node feature from HGSL, so the edge_list is not need
        H = self.get_final_emb(feature)#16 800 16
        return H


    def spectral_filter(self, z):
        if not self.args.use_filt:
            return [z[-1], z[-1]]
        # z [T, N, d]
        ctype = self.ctype
        time_len = z.shape[0]
        # transform into spectral domain
        z = torch.permute(z, (1, 0, 2))  # [N, T, d]
        specs = rfft(z, n=self.spec_len, norm="ortho", dim=1)  # [N, K, d]spec_len=time_step

        # learn causal and spurious masks
        c_spec, s_spec = self.spec_filt(specs)  # [N, d] node_num*dim

        if self.post_gnn:
            c_spec = self.nodeconv(c_spec)
            s_spec = self.nodeconv(s_spec)
        #out = [c_spec, s_spec]

        input_dim = self.args.output_dim * self.args.total_node
        #irfft
        zc_recon = irfft(c_spec, n=self.timenet_dim[-1], norm="ortho", dim=1)#node_num * timestep 190*16->190*15
        zs_recon = irfft(s_spec, n=self.timenet_dim[-1], norm="ortho", dim=1)#node_num * timestep
        h_rec = self.zc_zs_weight * zc_recon + (1-self.zc_zs_weight) * zs_recon
        # h_rec = 0.8 * zc_recon + 0.2 * zs_recon

        out = h_rec
        return out

    def get_final_emb(self, feature):
        # cs, ss = [], []
        H = []
        for t in range(self.time_length):
            # jump the GCN/GAT
            # the shape must is T*node_num*nhid (need bianhuan)
            x_list = feature[:t + 1] #type is list,len is timestep T, shape is node_num * dim(128)
            x_list = [self.linear(x) for x in x_list] #128 to 16, type is list,len is timestep T, shape is node_num * nhid(16)
            x_list = torch.stack(x_list)#type is tensor, shape is T*node_num*nhid
            # cz, sz = self.spectral_filter(x_list)  # [N, d] is node_num * nhid(16)
            # cs.append(cz)
            # ss.append(sz)
            h = self.spectral_filter(x_list)  # [N, d] is node_num * nhid(16) 800 16
            H.append(h)

        # cs = torch.stack(cs, dim=0)  # [T, N, d]
        # ss = torch.stack(ss, dim=0)  # [T, N, d]
        H = torch.stack(H, dim=0)# [T, N, d] 16 800 16 time_step*node_num*dim
        return H


class TemporalAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 n_heads,
                 num_time_steps,
                 attn_drop,
                 residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))  # [128, 128]; W*Q
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):


        position_inputs = torch.arange(0, self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs]

        # 2: Query, Key based multi-head self attention. [143, 16, 128]
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1] / self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]

        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)  # Q*K
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)  # output:[2288, 16, 16]
        self.attn_wts_all = outputs  # [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)  # dropout
        outputs = torch.matmul(outputs, v_)
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
                            dim=2)  # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)


'''
    def train_epoch(self, data):
        self.train()

        cz, sz = self.get_final_emb(data['edge_index'][:self.len_train])
        loss = self.cal_loss([cz, sz], data)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    @torch.no_grad()
    def test_epoch(self, data):
        model = self
        model.eval()

        cz, sz = self.get_final_emb(data['edge_index'])
        embeddings = cz

        auc_list = []
        for t in range(self.len):
            node_mask = data['node_masks'][t]
            preds = model.cs_decoder(embeddings[t])
            preds = preds[node_mask]
            preds = preds.argmax(dim=-1).squeeze()

            target = data['y'][node_mask].squeeze()
            auc = cal_metric(preds, target, self.args)
            auc_list.append(auc)
        train = np.mean(auc_list[:self.len_train])
        val = np.mean(auc_list[self.len_train:self.len_train + self.len_val])
        test = np.mean(auc_list[self.len_train + self.len_val:])

        COLLECTOR.add(key='auc_list', value=auc_list)

        return train, val, test
'''


