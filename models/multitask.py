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
        self.d_l = 512
        self.alpha = 1
        self.att_lstm = att_lstm(self.d_l)
        self.proj_v = nn.Sequential(
            #nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=3, padding=1, stride=1, bias=False),
            #nn.BatchNorm1d(self.d_l),
            nn.Linear(self.d_l, self.d_l // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_l // 4, self.d_l),
            nn.ReLU(inplace=True)
        )
        self.proj_l = nn.Sequential(
            #nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=3, padding=1, stride=1, bias=False),
            #nn.BatchNorm1d(self.d_l),
            nn.Linear(self.d_l, self.d_l // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_l // 4, self.d_l),
            nn.ReLU(inplace=True)
        )
        self.v_1conv = nn.Sequential(
            nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=1, stride=1, bias=False),
            #nn.BatchNorm1d(self.d_l),
            nn.ReLU(inplace=True)
        )
        self.l_1conv = nn.Sequential(
            nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=1, stride=1, bias=False),
            #nn.BatchNorm1d(self.d_l),
            nn.ReLU(inplace=True)
        )
        self.proj_u = nn.Sequential(
            nn.Linear(USER_DIM, self.d_l // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )
        self.user_emb = nn.Sequential(
            nn.Embedding(USER_EMB, self.d_l // 2),
            nn.Dropout(0.25)
        )
        self.cat_emb = nn.Sequential(
            nn.Embedding(CAT_EMB, self.d_l),
            #nn.Dropout(0.25)
        )
        self.subcat_emb = nn.Sequential(
            nn.Embedding(SUBCAT_EMB, self.d_l),
            #nn.Dropout(0.25)
        )
        self.concept_emb = nn.Sequential(
            nn.Embedding(CONCEPT_EMB, self.d_l),
            nn.Dropout(0.25)
        )
        self.fusion_category_1 = fusion_category(self.d_l, self.d_l)
        self.fusion_category_2 = fusion_category(self.d_l * 2, self.d_l)
        self.fusion_category = nn.Sequential(
            nn.Linear(self.d_l * 3, self.d_l),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.predict_cat = predict_cat(self.d_l, CAT_EMB, SUBCAT_EMB, CONCEPT_EMB)
        # 坑爹，加了初始化反而效果不好
        #self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

    def forward(
            self,
            visual,
            text,
            category,
            user,
            label
    ):
        # N * L * DIM -> N * L * H
        # visual_emb = self.proj_v(visual)
        # text_emb = self.proj_l(text)
        #visual_res = self.v_1conv(visual.transpose(1, 2)).transpose(1, 2)
        #text_res = self.l_1conv(text.transpose(1, 2)).transpose(1, 2)
        # visual_emb = (1 - self.alpha) * visual_emb + self.alpha * visual
        # text_emb = (1 - self.alpha) * text_emb + self.alpha * text

        user_id = self.user_emb(user.squeeze(2)[:, :, 0].long())
        user_des = self.proj_u(user.squeeze(2)[:, :, 1: USER_DIM + 1])
        user = torch.cat([user_id, user_des], dim=-1)
        cat_emb, loss_cat = self.predict_cat(visual, text, category)
        cat_emb = self.fusion_category(cat_emb)
        # cat = self.cat_emb(category[:, :, 0])
        # subcat = self.subcat_emb(category[:, :, 1])
        # concept = self.concept_emb(category[:, :, 2])
        # category_emb = self.fusion_category(self.fusion_category_2(self.fusion_category_1(cat, subcat), concept))
        #category_emb = self.fusion_category(torch.cat([cat, subcat, concept], dim=-1))
        loss, output = self.att_lstm(visual, text, user, cat_emb, label)

        return loss + 0.01 * loss_cat, output

    def test(
            self,
            visual,
            text,
            category,
            user
    ):
        # visual_emb = self.proj_v(visual)
        # text_emb = self.proj_l(text)
        #visual_res = self.v_1conv(visual.transpose(1, 2)).transpose(1, 2)
        #text_res = self.l_1conv(text.transpose(1, 2)).transpose(1, 2)
        # visual_emb = (1 - self.alpha) * visual_emb + self.alpha * visual
        # text_emb = (1 - self.alpha) * text_emb + self.alpha * text

        user_id = self.user_emb(user.squeeze(2)[:, :, 0].long())
        user_des = self.proj_u(user.squeeze(2)[:, :, 1: USER_DIM + 1])
        user = torch.cat([user_id, user_des], dim=-1)
        cat_emb = self.predict_cat.test(visual, text)
        cat_emb = self.fusion_category(cat_emb)
        # cat = self.cat_emb(category[:, :, 0])
        # subcat = self.subcat_emb(category[:, :, 1])
        # concept = self.concept_emb(category[:, :, 2])
        # category_emb = self.fusion_category(self.fusion_category_2(self.fusion_category_1(cat, subcat), concept))
        #category_emb = self.fusion_category(torch.cat([cat, subcat, concept], dim=-1))
        output = self.att_lstm.test(visual, text, user, cat_emb)

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


class predict_cat(nn.Module):
    def __init__(self, dim, cat_num, subcat_num, concept_num):
        super().__init__()
        self.d_l = dim * 2
        self.cat_num = cat_num
        self.subcat_num = subcat_num
        self.concept_num = concept_num
        self.emb_cat = nn.Linear(self.d_l, self.d_l // 2)
        self.emb_subcat = nn.Linear(self.d_l, self.d_l // 2)
        self.emb_concept = nn.Linear(self.d_l, self.d_l // 2)
        self.fusion_1 = fusion_category(self.d_l // 2, self.d_l // 2)
        self.fusion_2 = fusion_category(self.d_l, self.d_l // 2)
        self.output_cat = nn.Sequential(
            nn.Linear(self.d_l // 2, self.cat_num),
            nn.ReLU()
        )
        self.output_subcat = nn.Sequential(
            nn.Linear(self.d_l, subcat_num),
            nn.ReLU()
        )
        self.output_concept = nn.Sequential(
            nn.Linear(self.d_l // 2 * 3, concept_num),
            nn.ReLU()
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(
            self,
            visual,
            text,
            category
    ):
        #emb = self.emb_layer(torch.cat([visual, text], dim=-1))
        emb = torch.cat([visual, text], dim=-1)
        cat_emb = self.emb_cat(emb)
        subcat_emb = self.emb_subcat(emb)
        concept_emb = self.emb_concept(emb)
        subcat_fusion = self.fusion_1(cat_emb, subcat_emb)
        concept_fusion = self.fusion_2(subcat_fusion, concept_emb)
        cat_out = self.output_cat(cat_emb)
        subcat_out = self.output_subcat(subcat_fusion)
        concept_out = self.output_concept(concept_fusion)
        loss = self.loss(cat_out.transpose(1, 2), category[:, :, 0]) + \
               self.loss(subcat_out.transpose(1, 2), category[:, :, 1]) + \
               self.loss(concept_out.transpose(1, 2), category[:, :, 2])
        #loss = self.loss(concept_out.transpose(1, 2), category[:, :, 2])
        return concept_fusion, loss

    def test(
            self,
            visual,
            text
    ):
        emb = torch.cat([visual, text], dim=-1)
        cat_emb = self.emb_cat(emb)
        subcat_emb = self.emb_subcat(emb)
        concept_emb = self.emb_concept(emb)
        subcat_fusion = self.fusion_1(cat_emb, subcat_emb)
        concept_fusion = self.fusion_2(subcat_fusion, concept_emb)
        return concept_fusion


class att_lstm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d_l = dim
        self.hidden = dim // 2
        self.dim_k = dim // 2
        self.fusion = concat(self.d_l, 4, self.d_l)
        # self.fusion = grn(self.d_l, 5, self.d_l)
        #self.fusion = addition(self.d_l, self.d_l)
        self.lstm = lstm(self.d_l, self.hidden)
        self.output_layer = nn.Linear(self.dim_k, 1)
        # ib
        # self.fc_mu = nn.Linear(self.d_l * 5, self.d_l)
        # self.fc_std = nn.Linear(self.d_l * 5, self.d_l)
        # self.fc_mu_v = nn.Linear(self.d_l, self.d_l)
        # self.fc_std_v = nn.Linear(self.d_l, self.d_l)
        # self.fc_mu_l = nn.Linear(self.d_l, self.d_l)
        # self.fc_std_l = nn.Linear(self.d_l, self.d_l)
        # self.decoder_v = nn.Linear(self.d_l, 1)
        # self.decoder_l = nn.Linear(self.d_l, 1)
        self.num_heads = 4
        self.w_q = nn.Linear(self.hidden, self.dim_k)
        self.w_k = nn.Linear(self.hidden, self.dim_k)
        self.w_v = nn.ModuleList([nn.Linear(self.hidden, self.dim_k) for _ in range(self.num_heads)])
        self.w_out = nn.Linear(self.dim_k * self.num_heads, self.dim_k)
        self.w_k_Mul = nn.Linear(self.dim_k // self.num_heads, 1)
        self.att_type = "dot_satt"
        #self.layernorm = nn.LayerNorm(128)

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
            h_t = h_t.permute(1, 2, 0)
            # print(lstm_output.shape, h_t.shape)
            # lstm_output [N, L, H_out], h_t [N, H_out, 1]
            # att [N, L]
            att = torch.bmm(lstm_output, h_t)#.squeeze(2)
            # # Generate masks:判断是否是0向量，只针对0向量进行mask
            # padding_num = -2 ** 32 + 1
            # masks = torch.sign(torch.sum(torch.abs(tool), dim=-1).squeeze(-1))  # (N, L, H) -> (N, L)
            # # Apply masks to inputs
            # paddings = torch.ones_like(masks) * padding_num
            # mask_att = torch.where(torch.eq(masks, 0), paddings, att)  # (N, L)
            attention = F.softmax(att, 1)
            # bmm: [N, H_out, L] [N, L, 1]
            out = torch.bmm(lstm_output.permute(0, 2, 1), attention).squeeze(2)#.unsqueeze(2)).squeeze(2)
            
            return out
        '''
        add satt
        '''
        # batch_size, Doc_size, dim = lstm_output.shape
        # x = []
        # for i in range(h_t.size(0)):
        #     x.append(h_t[i, :, :])
        # hidden = torch.cat(x, dim=-1).unsqueeze(dim=-1)
        # ones = torch.ones(batch_size, 1, Doc_size).to(DEVICE)
        # hidden = torch.bmm(hidden, ones).transpose(1, 2)
        #
        # # 对lstm_out和hidden进行concat
        # h_i = torch.cat((lstm_output, hidden), dim=-1)
        #
        # dk = self.dim_k // self.num_heads  # dim_k of each head
        # # 分头，即，将h_i和权值矩阵w_q相乘的结果按列均分为n份，纬度变化如下：
        # # [batch_size, Doc_size, num_directions*hidden_dim*(1+n_layer)] -> [batch_size, Doc_size, dim_k]
        # # ->[batch_size, Doc_size, n, dk] -> [batch_size, n, Doc_size, dk]
        # query = self.w_q(h_i).reshape(batch_size, Doc_size, self.num_heads, dk).transpose(1, 2)
        # query = torch.tanh(query)  # query: [batch_size, n, Doc_size, dk]
        #
        # # 各头分别乘以不同的key，纬度变化如下：
        # # [batch_size, n, Doc_size, dk] * [batch_size, n, dk, 1]
        # # -> [batch_size, n, Doc_size, 1] -> [batch_size, n, Doc_size]
        # weights = self.w_k_Mul(query).transpose(0, 1) / math.sqrt(dk)  # weights: [n, batch_size, Doc_size, 1]
        # value = [wv(lstm_output).transpose(1, 2) for wv in self.w_v]  # value: n* [batch_size, dim_v, Doc_size]
        # soft_weights = F.softmax(weights, 2)
        # # value:[batch_size, dim, Doc_size]
        # out = [torch.matmul(v, w).squeeze() for v, w in zip(value, soft_weights)]
        #
        # # out[i]:[batch_size, dim, Doc_size] × [batch_size, Doc_size, 1] -> [batch_size, dim]
        # # out: [batch_size, dim] * n
        # out = torch.cat(out, dim=-1)
        # out: [batch_size, dim * n]
        # print(out.size())
        # out = self.w_out(out)  # 做一次线性变换，进一步提取特征


    # ib func start
    # def loss_normal_kl(self, mu, std, beta):
    #     kl = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2 * std.log() - 1)
    #     return beta * kl
    #
    # def encode(self, x):
    #     return self.fc_mu(x), F.softplus(self.fc_std(x) - 5, beta=1)
    #
    # def encode_v(self, x):
    #     return self.fc_mu_v(x), F.softplus(self.fc_std_v(x) - 5, beta=1)
    #
    # def encode_l(self, x):
    #     return self.fc_mu_l(x), F.softplus(self.fc_std_l(x) - 5, beta=1)
    #
    # def reparameterise(self, mu, std):
    #     # get epsilon from standard normal
    #     eps = torch.randn_like(std)
    #     return mu + std * eps
    # ib func end

    def forward(
            self,
            visual,
            text,
            user,
            cat,
            label
    ):
        # user = torch.cat([user_id, user_des], dim=-1)
        # mu_v, std_v = self.encode_v(visual)
        # visual = self.reparameterise(mu_v, std_v)
        # f_v = self.decoder_v(visual)
        # loss_v = self.loss_normal_kl(mu_v, std_v, 0.01) + self.loss_function(f_v, label)
        # mu_l, std_l = self.encode_l(text)
        # text = self.reparameterise(mu_l, std_l)
        # f_l = self.decoder_l(text)
        # loss_l = self.loss_normal_kl(mu_l, std_l, 0.01) + self.loss_function(f_l, label)

        v_t = self.fusion(torch.cat([visual, text, user, cat], dim=-1))
        # output:   N * L * H_out
        output_h, hn = self.lstm(v_t)
        #output_h = self.layernorm(output_h)
        output_h = self.att(output_h, hn, self.att_type)
        # ib start
        # mu, std = self.encode(output_h)
        # output_h = self.reparameterise(mu, std)
        # loss_ib = self.loss_normal_kl(mu, std, 0.01)
        # ib end
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
            user,
            cat
    ):
        # user = self.fusion_user(torch.cat([user_id, user_des], dim=-1))
        # user = torch.cat([user_id, user_des], dim=-1)
        # mu_v, std_v = self.encode_v(visual)
        # visual = self.reparameterise(mu_v, std_v)
        # mu_l, std_l = self.encode_l(text)
        # text = self.reparameterise(mu_l, std_l)

        v_t = self.fusion(torch.cat([visual, text, user, cat], dim=-1))
        output_h, hn = self.lstm(v_t)
        #output_h = self.layernorm(output_h)
        output_h = self.att(output_h, hn, self.att_type)
        # ib start
        # mu, std = self.encode(output_h)
        # output_h = self.reparameterise(mu, std)
        # ib end
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
        self.rnn = LayerNormLSTM(input_size, hidden_size)
        #self.norm = nn.LayerNorm(input_size)

    def forward(self, x):
        # x:        N * L * H_in
        # output:   N * L * H_out
        # hn, cn:   1 * N * H_out
        h0 = torch.randn(1, x.shape[0], self.hidden_size).to(DEVICE)
        c0 = torch.randn(1, x.shape[0], self.hidden_size).to(DEVICE)
        seq_len = [16] * x.shape[0]
        # x = self.norm(x)
        output, (hn, cn) = self.rnn(x.permute(1, 0, 2), (h0, c0), seq_len)
        return output.permute(1, 0, 2), hn


class grn(nn.Module):
    def __init__(self, in_size, num, output_dim):
        super().__init__()
        self.linear_a = nn.Linear(in_size, output_dim)
        # self.linear_c = nn.Sequential(nn.Linear(in_size, output_dim),
        #                               )
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
    def __init__(self, in_size, num, out_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_size * num, out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )


    def forward(self, mods):
        #print(v1.shape, l1.shape)
        z = self.mlp(mods)

        # mods: [N, L, H * 5] -> z: [N, L, H]


        # stack = torch.stack(mods, dim=2)
        # # att: [N, L, 5, H] * [N, L, H, 1]
        # att = torch.matmul(stack, z.unsqueeze(3))
        # att_weights = F.softmax(att, dim=2)
        # # F [N, L, H, 5] * [N, L, 5, 1]
        # f = torch.matmul(stack.transpose(2, 3), att_weights).squeeze(3)
        return z


class addition(nn.Module):
    def __init__(self, in_size, output_dim):
        super().__init__()
        self.linear = nn.Linear(in_size, output_dim)
    def forward(self, mods):
        y_1 = F.relu(self.linear(torch.sum(torch.stack(mods, dim=-1), dim=-1)))
        return y_1
