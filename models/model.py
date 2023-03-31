import logging
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F

from global_configs import TEXT_DIM, VISUAL_DIM, USER_DIM, USER_EMB, CAT_EMB, SUBCAT_EMB, CONCEPT_EMB, DEVICE


class MODEL(torch.nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.d_l = 256
        self.alpha = 0.6
        self.att_lstm = att_lstm(self.d_l)
        self.proj_v = nn.Sequential(
            nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.proj_l = nn.Sequential(
            nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.v_1conv = nn.Sequential(
            nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.l_1conv = nn.Sequential(
            nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.proj_u = nn.Sequential(
            nn.Linear(USER_DIM, self.d_l),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )
        self.user_emb = nn.Sequential(
            nn.Embedding(USER_EMB, self.d_l),
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
        # self.fusion_category = nn.Sequential(
        #     nn.Linear(self.d_l * 3, self.d_l),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )
        # self.predict_cat = predict_cat(self.d_l, CAT_EMB, SUBCAT_EMB, CONCEPT_EMB)
        self.fusion_category_1 = fusion_category(self.d_l, self.d_l)
        self.fusion_category_2 = fusion_category(self.d_l * 2, self.d_l)
        self.fusion_out = nn.Sequential(
            nn.Linear(self.d_l * 3, self.d_l),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        # 坑爹，加了初始化反而效果不好
        # self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
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
        visual_emb = self.proj_v(visual.transpose(1, 2)).transpose(1, 2)
        text_emb = self.proj_l(text.transpose(1, 2)).transpose(1, 2)
        visual_res = self.v_1conv(visual.transpose(1, 2)).transpose(1, 2)
        text_res = self.l_1conv(text.transpose(1, 2)).transpose(1, 2)
        visual_emb = (1 - self.alpha) * visual_emb + self.alpha * visual_res
        text_emb = (1 - self.alpha) * text_emb + self.alpha * text_res
        #emb, loss_cat = self.predict_cat(visual_emb, text_emb, category)
        user_id = self.user_emb(user.squeeze(2)[:, :, 0].long())
        user_des = self.proj_u(user.squeeze(2)[:, :, 1: USER_DIM + 1])
        cat = self.cat_emb(category[:, :, 0])
        subcat = self.subcat_emb(category[:, :, 1])
        concept = self.concept_emb(category[:, :, 2])
        #category_emb = self.fusion_category(torch.cat([cat, subcat, concept], dim=-1))
        category_emb = self.fusion_out(self.fusion_category_2(self.fusion_category_1(cat, subcat), concept))
        loss, output = self.att_lstm(visual_emb, text_emb, user_id, user_des, category_emb, label, visual)
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
        #emb = self.predict_cat.test(visual_emb, text_emb)
        user_id = self.user_emb(user.squeeze(2)[:, :, 0].long())
        user_des = self.proj_u(user.squeeze(2)[:, :, 1: USER_DIM + 1])
        cat = self.cat_emb(category[:, :, 0])
        subcat = self.subcat_emb(category[:, :, 1])
        concept = self.concept_emb(category[:, :, 2])
        #category_emb = self.fusion_category(torch.cat([cat, subcat, concept], dim=-1))
        category_emb = self.fusion_out(self.fusion_category_2(self.fusion_category_1(cat, subcat), concept))
        output = self.att_lstm.test(visual_emb, text_emb, user_id, user_des, category_emb, visual)

        return output


class fusion_category(nn.Module):
    def __init__(self, last_dim, this_dim):
        super(fusion_category, self).__init__()
        self.last_dim = last_dim
        self.this_dim = this_dim
        self.linear_last = nn.Linear(self.last_dim, self.last_dim)
        self.linear_this = nn.Linear(self.this_dim, self.this_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_last, f_this):
        f_this_trans = self.linear_this(f_this)
        f_last_trans = self.linear_last(f_last)
        if f_this_trans.shape[2] < f_last_trans.shape[2]:
            f_this_expand = torch.cat([f_this_trans, f_this_trans], dim=-1)
        else:
            f_this_expand = f_this_trans
        f_this_gate = self.sigmoid(f_last_trans + f_this_expand)
        f_last_gate = torch.mul(f_last, f_this_gate)
        f_this_fusion = torch.cat([f_this, f_last_gate], dim=-1)
        return f_this_fusion


class predict_cat(nn.Module):
    def __init__(self, dim, cat_num, subcat_num, concept_num):
        super().__init__()
        self.d_l = dim
        self.cat_num = cat_num
        self.subcat_num = subcat_num
        self.concept_num = concept_num
        self.emb_layer = nn.Sequential(
            nn.Linear(self.d_l * 2, self.d_l * 2),
            nn.ReLU(),
        )
        self.output_cat = nn.Sequential(
            nn.Linear(self.d_l * 2, self.d_l),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.d_l, cat_num),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.output_subcat = nn.Sequential(
            nn.Linear(self.d_l * 2, self.d_l),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.d_l, subcat_num),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.output_concept = nn.Sequential(
            nn.Linear(self.d_l * 2, self.d_l * 4),
            nn.ReLU(),
            nn.Linear(self.d_l * 4, concept_num),
            nn.ReLU(),
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
        cat = self.output_cat(emb)
        subcat = self.output_subcat(emb)
        concept = self.output_concept(emb)
        #loss = self.loss(cat.transpose(1, 2), category[:, :, 0]) + self.loss(subcat.transpose(1, 2), category[:, :, 1]) + self.loss(concept.transpose(1, 2), category[:, :, 2])
        loss = self.loss(concept.transpose(1, 2), category[:, :, 2])
        return emb, loss

    def test(
            self,
            visual,
            text
    ):
        emb = torch.cat([visual, text], dim=-1)
        return emb


class att_lstm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d_l = dim
        self.fusion = concat(self.d_l, 5, self.d_l)
        # self.fusion = grn(self.d_l, 5, self.d_l)
        #self.fusion = addition(self.d_l, self.d_l)
        self.lstm = lstm(self.d_l, 128)
        self.output_layer = nn.Linear(128, 1)
        # self.decode_v = nn.Linear(self.d_l, 1)
        # self.decode_l = nn.Linear(self.d_l, 1)
        # self.decode_uid = nn.Linear(self.d_l, 1)
        # self.decode_u = nn.Linear(self.d_l, 1)
        # self.decode_c = nn.Linear(self.d_l, 1)
        # ib
        # self.fc_mu = nn.Linear(self.d_l * 5, self.d_l)
        # self.fc_std = nn.Linear(self.d_l * 5, self.d_l)
        # self.fc_mu_v = nn.Linear(self.d_l, self.d_l)
        # self.fc_std_v = nn.Linear(self.d_l, self.d_l)
        # self.fc_mu_l = nn.Linear(self.d_l, self.d_l)
        # self.fc_std_l = nn.Linear(self.d_l, self.d_l)
        # self.decoder_v = nn.Linear(self.d_l, 1)
        # self.decoder_l = nn.Linear(self.d_l, 1)

    def loss_function(self, y_pred, y):
        loss_fct = MSELoss()
        MSE = loss_fct(y_pred.view(-1, ), y.view(-1, ))
        return MSE

    def att(self, lstm_output, h_t, tool):
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


        # lstm_output [N, L, H_out], h_t [1, N, H_out]
        # h_t [1, N, H_out] -> [N, H_out, 1]
        
        h_t = h_t.permute(1, 2, 0)
        # print(lstm_output.shape, h_t.shape)
        # lstm_output [N, L, H_out], h_t [N, H_out, 1]
        # att [N, L]
        att_weights = torch.bmm(lstm_output, h_t).squeeze(2)
        # Generate masks:判断是否是0向量，只针对0向量进行mask
        padding_num = -2 ** 32 + 1
        masks = torch.sign(torch.sum(torch.abs(tool), dim=-1).squeeze(-1)) # (N, L, H) -> (N, L)
        # Apply masks to inputs
        paddings = torch.ones_like(masks) * padding_num
        mask_att = torch.where(torch.eq(masks, 0), paddings, att_weights)  # (N, L)
        attention = F.softmax(mask_att, 1)
        # bmm: [N, H_out, L] [N, L, 1]
        attn_out = torch.bmm(lstm_output.permute(0, 2, 1), attention.unsqueeze(2))
        return attn_out.squeeze(2)

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
            user_id,
            user_des,
            category,
            label,
            tool
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
        label_one = label[:, -1, :]
        # result_v = self.decode_v(visual[:, -1, :].squeeze(1))
        # result_l = self.decode_l(text[:, -1, :].squeeze(1))
        # result_uid = self.decode_uid(user_id[:, -1, :].squeeze(1))
        # result_u = self.decode_u(user_des[:, -1, :].squeeze(1))
        # result_c = self.decode_c(category[:, -1, :].squeeze(1))
        #print(self.loss_function(result_v, label_one), self.loss_function(result_l, label_one), self.loss_function(result_uid, label_one), self.loss_function(result_u, label_one), self.loss_function(result_c, label_one))
        # loss_v = (self.loss_function(text.contiguous(), visual.contiguous()) + self.loss_function(text.contiguous(), user_id.contiguous()) +
        #           self.loss_function(text.contiguous(), user_des.contiguous()) + self.loss_function(text.contiguous(), category.contiguous())) / 4

        v_t = self.fusion(torch.cat([visual, text, user_id, user_des, category], dim=-1))
        # output:   N * L * H_out
        output_h, hn = self.lstm(v_t)
        output_h = self.att(output_h, hn, tool)
        # ib start
        # mu, std = self.encode(output_h)
        # output_h = self.reparameterise(mu, std)
        # loss_ib = self.loss_normal_kl(mu, std, 0.01)
        # ib end
        output = self.output_layer(output_h)
        # no att
        #output = self.output_layer(torch.mean(output_h, dim=1).squeeze(1))

        loss = self.loss_function(output, label_one)
        return loss, output

    def test(
            self,
            visual,
            text,
            user_id,
            user_des,
            category,
            tool
    ):
        # user = self.fusion_user(torch.cat([user_id, user_des], dim=-1))
        # user = torch.cat([user_id, user_des], dim=-1)
        # mu_v, std_v = self.encode_v(visual)
        # visual = self.reparameterise(mu_v, std_v)
        # mu_l, std_l = self.encode_l(text)
        # text = self.reparameterise(mu_l, std_l)

        v_t = self.fusion(torch.cat([visual, text, user_id, user_des, category], dim=-1))
        output_h, hn = self.lstm(v_t)
        output_h = self.att(output_h, hn, tool)
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
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        #self.norm = nn.LayerNorm(input_size)

    def forward(self, x):
        # x:        N * L * H_in
        # output:   N * L * H_out
        # hn, cn:   1 * N * H_out
        h0 = torch.randn(1, x.shape[0], self.hidden_size).to(DEVICE)
        c0 = torch.randn(1, x.shape[0], self.hidden_size).to(DEVICE)
        # x = self.norm(x)
        output, (hn, cn) = self.rnn(x, (h0, c0))
        return output, hn


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
    def __init__(self, in_size, num, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_size * num, in_size * int(num / 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_size * int(num / 2), output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
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
