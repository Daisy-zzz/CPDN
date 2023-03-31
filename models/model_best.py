import logging
import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F
from models.LSTM_Layernorm import LayerNormLSTM
from global_configs import TEXT_DIM, VISUAL_DIM, USER_DIM, USER_EMB, CAT_EMB, SUBCAT_EMB, CONCEPT_EMB, DEVICE


class MODEL(torch.nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.d_l = 256
        self.alpha = 0.6
        self.att_lstm = att_lstm(self.d_l)
        self.proj_v = nn.Sequential(
            nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )
        self.proj_l = nn.Sequential(
            nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )
        self.v_1conv = nn.Sequential(
            nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=1, stride=1, bias=False),
            nn.ReLU()
        )
        self.l_1conv = nn.Sequential(
            nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=1, stride=1, bias=False),
            nn.ReLU()
        )
        self.proj_u = nn.Sequential(
            nn.Linear(USER_DIM, self.d_l),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.user_emb = nn.Sequential(
            nn.Embedding(USER_EMB, self.d_l),
            nn.Dropout(0.25)
        )
        self.cat_emb = nn.Sequential(
            nn.Embedding(CAT_EMB, self.d_l),
            nn.Dropout(0.25)
        )
        self.subcat_emb = nn.Sequential(
            nn.Embedding(SUBCAT_EMB, self.d_l),
            nn.Dropout(0.25)
        )
        self.concept_emb = nn.Sequential(
            nn.Embedding(CONCEPT_EMB, self.d_l),
            nn.Dropout(0.5)
        )
        self.fusion_category_1 = fusion_category(self.d_l, self.d_l)
        self.fusion_category_2 = fusion_category(self.d_l * 2, self.d_l)
        self.fusion_category = nn.Sequential(
            nn.Linear(self.d_l * 3, self.d_l),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(
            self,
            visual,
            text,
            category,
            user,
            label
    ):
        # N * L * DIM -> N * L * H
        visual_emb = self.proj_v(visual.transpose(1, 2)).transpose(1, 2)
        text_emb = self.proj_l(text.transpose(1, 2)).transpose(1, 2)
        visual_res = self.v_1conv(visual.transpose(1, 2)).transpose(1, 2)
        text_res = self.l_1conv(text.transpose(1, 2)).transpose(1, 2)
        visual_emb = (1 - self.alpha) * visual_emb + self.alpha * visual_res
        text_emb = (1 - self.alpha) * text_emb + self.alpha * text_res
        user_id = self.user_emb(user.squeeze(2)[:, :, 0].long())
        user_des = self.proj_u(user.squeeze(2)[:, :, 1: USER_DIM + 1])
        cat = self.cat_emb(category[:, :, 0])
        subcat = self.subcat_emb(category[:, :, 1])
        concept = self.concept_emb(category[:, :, 2])
        category_emb = self.fusion_category_2(self.fusion_category_1(cat, subcat), concept)
        #category_emb = torch.cat([cat, subcat, concept], dim=-1)
        loss, output = self.att_lstm(visual_emb, text_emb, user_id, user_des, category_emb, label)

        return loss, output

    def test(
            self,
            visual,
            text,
            category,
            user
    ):
        visual_emb = self.proj_v(visual.transpose(1, 2)).transpose(1, 2)
        text_emb = self.proj_l(text.transpose(1, 2)).transpose(1, 2)
        visual_res = self.v_1conv(visual.transpose(1, 2)).transpose(1, 2)
        text_res = self.l_1conv(text.transpose(1, 2)).transpose(1, 2)
        visual_emb = (1 - self.alpha) * visual_emb + self.alpha * visual_res
        text_emb = (1 - self.alpha) * text_emb + self.alpha * text_res
        user_id = self.user_emb(user.squeeze(2)[:, :, 0].long())
        user_des = self.proj_u(user.squeeze(2)[:, :, 1: USER_DIM + 1])
        cat = self.cat_emb(category[:, :, 0])
        subcat = self.subcat_emb(category[:, :, 1])
        concept = self.concept_emb(category[:, :, 2])
        category_emb = self.fusion_category_2(self.fusion_category_1(cat, subcat), concept)
        #category_emb = torch.cat([cat, subcat, concept], dim=-1)
        output = self.att_lstm.test(visual_emb, text_emb, user_id, user_des, category_emb)

        return output


class fusion_category(nn.Module):
    def __init__(self, last_dim, this_dim):
        super(fusion_category, self).__init__()
        self.last_dim = last_dim
        self.this_dim = this_dim
        self.linear_last = nn.Linear(self.last_dim, self.last_dim)
        self.linear_this = nn.Linear(self.this_dim, self.this_dim)
        self.linear_last_o = nn.Linear(self.last_dim, self.last_dim)
        self.linear_this_o = nn.Linear(self.this_dim, self.this_dim)

    def forward(self, f_last, f_this):
        f_this_trans = self.linear_this(f_this)
        f_last_trans = self.linear_last(f_last)
        if f_this_trans.shape[2] < f_last_trans.shape[2]:
            f_this_expand = torch.cat([f_this_trans, f_this_trans], dim=-1)
        else:
            f_this_expand = f_this_trans
        f_this_gate = torch.sigmoid(f_last_trans + f_this_expand)
        f_last_gate = torch.mul(f_last, f_this_gate)
        #f_out_gate = torch.sigmoid(self.linear_last_o(f_last) + self.linear_this_o(f_this))
        #f_this_fusion = torch.mul(f_out_gate, torch.tanh(f_this + f_last_gate))
        f_this_fusion = torch.cat([f_this, f_last_gate], dim=-1)
        #f_this_fusion = torch.tanh(f_this + f_last_gate)
        return f_this_fusion

import math
class att_lstm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d_l = dim
        self.hidden = dim // 2
        self.dim_k = dim // 2
        self.fusion = concat(self.d_l, 7, self.d_l)
        #self.fusion = addition(self.d_l, self.d_l)
        self.lstm = lstm(self.d_l, self.hidden)
        self.output_layer = nn.Linear(self.hidden, 1)
        #self.encoder_layer = nn.TransformerEncoderLayer(self.d_l, nhead=4, batch_first=True, norm_first=True)
        #self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.num_heads = 2
        self.w_q = nn.Linear(self.hidden, self.dim_k)
        self.w_k = nn.Linear(self.hidden, self.dim_k)
        self.w_v = nn.ModuleList([nn.Linear(self.hidden, self.dim_k) for _ in range(self.num_heads)])
        self.w_out = nn.Linear(self.dim_k * self.num_heads, self.dim_k * self.num_heads // 2)
        #self.w_k_Mul = nn.Linear(self.dim_k // self.num_heads, 1)
        self.att_type = "att"

    def loss_function(self, y_pred, y):
        loss_fct = MSELoss()
        MSE = loss_fct(y_pred.view(-1, ), y.view(-1, ))
        return MSE

    def att(self, lstm_output, h_t, type):
        # # lstm_output [N, L, H_out], h_t [N, L, H_out]
        # # h_t [N, L, H_out] -> [N, H_out, L]
        # h_t = h_t.permute(0, 2, 1)
        # # print(lstm_output.shape, h_t.shape)
        # # lstm_output [N, L, H_out], h_t [N, H_out, L]
        # # att [N, L, L]
        # att_weights = torch.bmm(lstm_output, h_t)
        # # 添加mask,下三角形矩阵,上三角元素都是0
        # padding_num = -2 ** 32 + 1
        # diag_vals = torch.ones_like(att_weights[0, :, :])  # (L, L)
        # low_tril = torch.tril(diag_vals, diagonal=0)  # (L, L)
        # masks = torch.tile(low_tril.unsqueeze(0), [att_weights.shape[0], 1, 1])  # (N, L, L)
        # paddings = torch.ones_like(masks) * padding_num
        # mask_att = torch.where(torch.eq(masks, 0), paddings, att_weights)
        #
        # attention = F.softmax(mask_att, 2)
        # # bmm: [N, L, L] [N, L, H_out]
        # attn_out = torch.bmm(attention, lstm_output)
        '''
        attention
        '''
        if type == "dot_satt":
            b_s, seq_len, dim = lstm_output.shape
            d_k = self.dim_k // self.num_heads
            q_n = self.w_q(h_t.squeeze(0)).reshape(b_s, self.num_heads, d_k).unsqueeze(dim=-1) # [N, k, d_k, 1]
            key = self.w_k(lstm_output).reshape(b_s, seq_len, self.num_heads, d_k).transpose(1, 2) # [N, k, L, d_k]
            value = [w_v(lstm_output).transpose(1, 2) for w_v in self.w_v] # k * [N, d_v, L]
            weights = torch.matmul(key, q_n).transpose(0, 1) / math.sqrt(d_k) # [N, k, L, 1]
            soft_weights = F.softmax(weights, 2)
            out = [torch.matmul(v, w).squeeze() for v, w in zip(value, soft_weights)]
            # out[i]:[N, d_v, L] × [N, L, 1] -> [N, d_v]
            # out: [N, d_v] * k
            out = torch.cat(out, dim=-1)
            out = self.w_out(out) # [batch_size, dim_v]
            return out
        elif type == "att":
            # lstm_output [N, L, H_out], h_t [1, N, H_out]
            # h_t [1, N, H_out] -> [N, H_out, 1]
            # lstm_output [N, L, H_out], h_t [N, H_out, 1]
            # att [N, L]
            h_t = h_t.permute(1, 2, 0)
            att = torch.bmm(lstm_output, h_t)#.squeeze(2)
            #N, L, H = lstm_output.shape
            #h_t = h_t.permute(1, 0, 2).expand(N, L, H)
            #att = torch.cosine_similarity(h_t, lstm_output, dim=2).unsqueeze(2)
            # # Generate masks:判断是否是0向量，只针对0向量进行mask
            #padding_num = -2 ** 32 + 1
            #masks = torch.sign(torch.sum(torch.abs(tool), dim=-1).squeeze(-1))  # (N, L, H) -> (N, L)
            # Apply masks to inputs
            #paddings = torch.ones_like(masks) * padding_num
            #att = torch.where(torch.eq(masks, 0), paddings, att)  # (N, L)
            attention = F.softmax(att, 1)
            # bmm: [N, H_out, L] [N, L, 1]
            out = torch.bmm(lstm_output.permute(0, 2, 1), attention).squeeze(2)#.unsqueeze(2)).squeeze(2)
            
            return out

    def forward(
            self,
            visual,
            text,
            user_id,
            user_des,
            category,
            label
    ):
        v_t = self.fusion([visual, text, category, user_id, user_des])
        # output:   N * L * H_out
        output_h, hn = self.lstm(v_t)
        output_h = self.att(output_h, hn, self.att_type)
        output = self.output_layer(output_h)
        # no att
        #output = self.output_layer(torch.mean(output_h, dim=1).squeeze(1))
        label_one = label[:, -1, :].squeeze(1)
        loss = self.loss_function(output, label_one)
        return loss, output

    def test(
            self,
            visual,
            text,
            user_id,
            user_des,
            category
    ):
        v_t = self.fusion([visual, text, category, user_id, user_des])
        output_h, hn = self.lstm(v_t)
        output_h = self.att(output_h, hn, self.att_type)
        output = self.output_layer(output_h)
        # no att
        #output = self.output_layer(torch.mean(output_h, dim=1).squeeze(1))
        return output


class lstm(nn.Module):
    def __init__(self,
                 input_size=256,
                 hidden_size=128
                 ):
        super().__init__()
        # D = 1, num_layers = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.rnn = LayerNormLSTM(input_size, hidden_size)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x:        N * L * H_in
        # output:   N * L * H_out
        # hn, cn:   1 * N * H_out
        h0 = torch.randn(1, x.shape[0], self.hidden_size).to(DEVICE)
        c0 = torch.randn(1, x.shape[0], self.hidden_size).to(DEVICE)
        #seq_len = [16] * x.shape[0]
        #output, (hn, cn) = self.rnn(x.permute(1, 0, 2), (h0, c0), seq_len)
        #output = output.permute(1, 0, 2)
        output, (hn, cn) = self.rnn(x, (h0, c0))
        return output, hn


class grn(nn.Module):
    def __init__(self, in_size, num, output_dim):
        super().__init__()
        self.linear_a = nn.Linear(in_size, output_dim)
        self.linear = nn.Linear(in_size, output_dim)
        self.linear_3 = nn.Linear(in_size, output_dim)
        self.linear_4 = nn.Linear(in_size, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.fusion = concat(in_size, 5, in_size)
    def forward(self, mods):
        #print(v1.shape, l1.shape)
        a = self.fusion(mods)
        n_2 = self.elu(self.linear_a(a))
        f = self.linear(n_2)
        z = self.sigmoid(self.linear_3(f)) * self.linear_4(f)
        final = a + z
        return final


class concat(nn.Module):
    def __init__(self, in_size, num, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_size * num, in_size * int(num // 2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_size * int(num // 2), output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )


    def forward(self, mods_list):
        mods = torch.cat(mods_list, dim=-1)
        z = self.mlp(mods)
        return z


class addition(nn.Module):
    def __init__(self, in_size, output_dim):
        super().__init__()
        self.linear = nn.Linear(5, 1)
    def forward(self, mods):
        y_1 = self.linear(torch.stack(mods, dim=-1))
        return y_1
