import logging
import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from global_configs import TEXT_DIM, VISUAL_DIM, USER_DIM, USER_EMB, CAT_EMB, SUBCAT_EMB, CONCEPT_EMB, DEVICE, SEQ_LEN

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """

    def __init__(self, d_model, dropout=0.25, max_len=SEQ_LEN+1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class gate(nn.Module):
    def __init__(self, dim):
        super(gate, self).__init__()
        self.fc_1 = nn.Linear(dim, dim)
        self.fc_2 = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)
        #self.init_weights()

    def init_weights(self):
            for n, p in self.named_parameters():
                if 'bias' not in n:
                    torch.nn.init.xavier_uniform_(p)
                elif 'bias' in n:
                    torch.nn.init.zeros_(p)

    def forward(self, x):
        x = self.dropout(x)
        sig = self.sigmoid(self.fc_1(x))
        x = self.fc_2(x)
        # if len(x.shape) == 3:
        #     output = self.norm((x + gate_output).transpose(1, 2)).transpose(1, 2)
        # else:
        #output = self.norm(x + gate_output)
        return torch.mul(sig, x)


class TimeDistributed(nn.Module):
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class grn(nn.Module):
    def __init__(self, in_size, hidden_size, output_dim):
        super(grn, self).__init__()
        self.input_size = in_size
        self.hidden_size = hidden_size
        self.output_size = output_dim
        self.fc_1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, self.output_size)
        self.elu = nn.ELU()
        self.gate = gate(self.output_size)
        #self.norm =  TimeDistributed(nn.BatchNorm1d(self.output_size))
        self.norm = nn.LayerNorm(self.output_size)
        # self.dropout = nn.Dropout(0.25)
        if self.input_size != self.output_size:
            self.skip_layer = nn.Linear(self.input_size, self.output_size)

        #self.init_weights()
            
    def init_weights(self):
        for name, p in self.named_parameters():
            if 'fc_1' in name and 'bias' not in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif 'fc_2' in name and 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        # print(v1.shape, l1.shape)
        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x
        x = self.fc_1(x)
        x = self.elu(x)
        #x = F.relu(x)
        x = self.fc_2(x)
        x = self.gate(x)
        x = x + residual
        x = self.norm(x)
        return x


class concat(nn.Module):
    def __init__(self, in_size, num, output_dim):
        super(concat, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_size * num, output_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            #nn.Linear(in_size * int(num // 2), output_dim),
            #nn.ReLU(),
            #nn.Dropout(0.25)
        )

    def forward(self, mods_list):
        mods = torch.cat(mods_list, dim=-1)
        z = self.mlp(mods)
        return z


class fusion_category(nn.Module):
    def __init__(self, last_dim, this_dim):
        super(fusion_category, self).__init__()
        self.last_dim = last_dim
        self.this_dim = this_dim
        self.linear_last = nn.Linear(self.last_dim, self.last_dim)
        self.linear_this = nn.Linear(self.this_dim, self.this_dim)
        # self.linear_last_2 = nn.Linear(self.this_dim, self.this_dim)
        # self.out = nn.Linear(self.this_dim * 2, self.this_dim, bias=False)
        # self.linear_last_o = nn.Linear(self.last_dim, self.last_dim)
        # self.linear_this_o = nn.Linear(self.this_dim, self.this_dim)

    def forward(self, f_last, f_this):
        f_this_trans = self.linear_this(f_this)
        f_last_trans = self.linear_last(f_last)
        if f_this_trans.shape[2] < f_last_trans.shape[2]:
            f_this_expand = torch.cat([f_this_trans, f_this_trans], dim=-1)
        else:
            f_this_expand = f_this_trans
        f_this_gate = torch.sigmoid(f_last_trans + f_this_expand)
        f_last_gate = torch.mul(f_last, f_this_gate)
        # f_out_gate = torch.sigmoid(self.linear_last_o(f_last) + self.linear_this_o(f_this))
        # f_this_fusion = torch.mul(f_out_gate, torch.tanh(f_this + f_last_gate))
        f_this_fusion = torch.cat([f_this, f_last_gate], dim=-1)
        # f_this_fusion = f_this + f_last_gate
        # f_this_fusion = torch.tanh(f_this + f_last_gate)
        # f_last_trans = self.linear_last(f_last)
        # gate = torch.sigmoid(self.f_this.unsqueeze(0)) + self.linear_last_2(f_last.unsqueeze(0)))
        # f_last_gate = f_last_trans * gate
        # f_this_fusion = self.out(torch.cat([f_last_gate, f_this], dim=-1))
        return f_this_fusion


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, output_size):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_size =input_size
        self.output_size = output_size
        self.num_inputs = num_inputs
        self.flattened_grn = grn(self.num_inputs*self.input_size, self.hidden_size, self.num_inputs)

        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(grn(self.input_size, self.hidden_size, self.output_size))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding):
        sparse_weights = self.flattened_grn(embedding)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        var_outputs = []
        for i in range(self.num_inputs):
            ##select slice of embedding belonging to a single input
            var_outputs.append(self.single_variable_grns[i](embedding[:, :, (i*self.input_size) : (i+1)*self.input_size]))
        
        var_outputs = torch.stack(var_outputs, axis=-1)
        outputs = var_outputs*sparse_weights
        
        outputs = outputs.sum(axis=-1)

        return outputs

class multihead_attn(nn.Module):
    def __init__(self, input_size, hidden_size, nheads) -> None:
        super(multihead_attn, self).__init__()
        self.nheads = nheads
        self.d_out = input_size
        self.dim_k = hidden_size
        self.attn = nn.MultiheadAttention(self.d_out, self.nheads, batch_first=True, dropout=0.25)
        self.w_q = nn.Linear(self.d_out, self.dim_k)
        self.w_k = nn.Linear(self.d_out, self.dim_k)
        self.w_v = nn.Linear(self.d_out, self.dim_k)

    def forward(self, x, mask):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        mask = mask.tile((x.shape[0] * self.nheads, 1, 1)).to(DEVICE)
        attn_x, attn_w = self.attn(q, k, v, attn_mask=mask)
        return attn_x

class att_lstm(nn.Module):
    def __init__(self, dim):
        super(att_lstm, self).__init__()
        self.d_l = dim
        self.d_out = dim
        self.dim_k = dim
        self.attn_heads = 2
        self.fusion = concat(self.d_l, 6, self.d_l)
        self.lstm = lstm(self.d_l, self.d_out)
        self.post_lstm_grn = grn(self.d_out, self.d_out, self.d_out)
        self.post_lstm_gate = gate(self.d_out)
        self.post_lstm_norm = nn.LayerNorm(self.d_out)#TimeDistributed(nn.BatchNorm1d(self.d_out))
        
        #self.pre_lstm_grn = grn(self.d_out * 5, self.d_out, self.d_out)
        #self.pre_fc_grn = grn(self.d_out * 3, self.d_out * 3, self.d_out * 3)
        
        # self.post_attn_gate = gate(self.d_out)
        # self.post_attn_norm = nn.LayerNorm(self.d_out)
        # self.post_attn_grn = grn(self.d_out, self.d_out, self.d_out)
        #self.pre_fc_gate = gate(self.d_out)
        #self.pre_fc_norm = nn.LayerNorm(self.d_out)
        # self.fc_output = nn.Sequential(
        #     nn.Dropout(0.25),
        #     nn.Linear(self.d_out * 7, self.d_out * 3),
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(self.d_out * 3, 1)
        # ) 
        self.multihead_attn = multihead_attn(self.d_out, self.d_out, self.attn_heads)
        #self.fc_output = nn.Linear(self.d_out * 2, 1)
        self.fc_output = nn.Sequential(
            nn.Linear(self.d_out, self.d_out),
            nn.Linear(self.d_out, 1)
        )
        self.attn_mask = self.generate_square_subsequent_mask(SEQ_LEN)
        #self.fc_norm = nn.LayerNorm(self.d_out * 3)
        #self.fc_downscale = nn.Linear(self.d_l, self.d_out)
        #self.multihead_attn_t = multihead_attn(self.d_out, self.d_out, 4)
        # self.postion = PositionalEncoding(self.d_l)
        #self.vsn = VariableSelectionNetwork(self.d_out, 5, self.d_out, self.d_out)

    def generate_square_subsequent_mask(self, sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask  


    def att(self, lstm_output, h_t):
        h_t = h_t.permute(1, 2, 0)
        att = torch.bmm(lstm_output, h_t)
        attention = F.softmax(att, 1)
        out = torch.bmm(lstm_output.permute(0, 2, 1), attention).squeeze(2)
        return out

    def loss_function(self, y_pred, y):
        #loss_fct = MSELoss()
        loss_fct = MSELoss()
        MSE = loss_fct(y_pred.view(-1, ), y.view(-1, ))
        return MSE

    def forward(
            self,
            visual,
            text,
            user_id,
            user_des,
            category,
            label
    ):
        lstm_input = self.fusion([visual, text, category, user_des])
        lstm_output, hn = self.lstm(lstm_input)
        #lstm_output = lstm_input
        lstm_output = self.post_lstm_gate(lstm_output)
        attn_input = self.post_lstm_norm(lstm_input + lstm_output)
        attn_input = self.post_lstm_grn(lstm_output)
        #attn_input = lstm_output
        attn_output = self.multihead_attn(attn_input, self.attn_mask)[:, -1, :]
        #visual = self.att(visual, visual[:, [-1], :].transpose(0, 1))
        #text = self.att(text, text[:, [-1], :].transpose(0, 1))
        # category = self.att(category, category[:, [-1], :].transpose(0, 1))
        # visual = self.multihead_attn_v(self.postion(visual.transpose(0, 1)).transpose(0, 1))[:, -1, :]
        # text = self .multihead_attn_t(self.postion(text.transpose(0, 1)).transpose(0, 1))[:, -1, :]
        # category = self.multihead_attn(category)[:, [-1], :]
        # lstm_output, hn = self.lstm(lstm_input)
        # lstm_output = self.post_lstm_gate(lstm_output)
        # lstm_output = self.post_lstm_norm(lstm_input+lstm_output)#self.fc_downscale(lstm_input) + lstm_output)
        #attn_input = self.post_lstm_grn(lstm_input)

        # attn_input = lstm_input
        # attn_output = self.att(attn_input, attn_input[:, [-1], :].transpose(0, 1))
        
        # attn_output = self.post_attn_gate(attn_output)
        # attn_output = self.post_attn_norm(attn_input[:, -1, :] + attn_output)
        #attn_output = self.post_attn_grn(attn_output)
        #fc_input = self.pre_fc_gate(attn_output)
        #fc_input = self.pre_fc_norm(fc_input + lstm_output[:, -1, :])
        #attn_output = attn_input[:,-1,:]
        # fc_input = torch.cat([attn_output, user_id[:, -1, :], user_des[:, -1, :]], dim=-1)
        fc_input = attn_output
        #fc_input = self.fc_norm(fc_input)
        #fc_input = self.pre_fc_grn(fc_input)
        output = self.fc_output(fc_input)

        # no att
        # output = self.output_layer(torch.mean(output_h, dim=1).squeeze(1))
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
        lstm_input = self.fusion([visual, text, category, user_des])
        lstm_output, hn = self.lstm(lstm_input) 
        #lstm_output = lstm_input
        lstm_output = self.post_lstm_gate(lstm_output)
        attn_input = self.post_lstm_norm(lstm_input + lstm_output)
        attn_input = self.post_lstm_grn(lstm_output)
        #attn_input = lstm_output
        attn_output = self.multihead_attn(attn_input, self.attn_mask)[:, -1, :]
        # visual = self.att(visual, visual[:, [-1], :].transpose(0, 1))
        # text = self.att(text, text[:, [-1], :].transpose(0, 1))
        # category = self.att(category, category[:, [-1], :].transpose(0, 1))
        # visual = self.multihead_attn_v(self.postion(visual.transpose(0, 1)).transpose(0, 1))[:, -1, :]
        # text = self.multihead_attn_t(self.postion(text.transpose(0, 1)).transpose(0, 1))[:, -1, :]
        # lstm_output, hn = self.lstm(lstm_input)
        # lstm_output = self.post_lstm_gate(lstm_output)
        # lstm_output = self.post_lstm_norm(lstm_input+lstm_output)#self.fc_downscale(lstm_input) + lstm_output)
        #attn_input = self.post_lstm_grn(lstm_input)

        # attn_input = lstm_input
        # attn_output = self.att(attn_input, attn_input[:, [-1], :].transpose(0, 1))

        # mu, std = self.encode(attn_output)
        # attn_output = self.reparameterise(mu, std)

        #attn_output = self.post_attn_gate(attn_output)
        #attn_output = self.post_attn_norm(attn_input[:, -1, :] + attn_output)
        #attn_output = self.post_attn_grn(attn_output)
        #fc_input = self.pre_fc_gate(attn_output)
        #fc_input = self.pre_fc_norm(fc_input + lstm_output[:, -1, :])
        #attn_output = attn_input[:,-1,:]
        #fc_input = torch.cat([attn_output, user_id[:, -1, :], user_des[:, -1, :]], dim=-1)
        fc_input = fc_input = attn_output
        #fc_input = self.fc_norm(fc_input)
        #fc_input = self.pre_fc_grn(fc_input)
        output = self.fc_output(fc_input)

        # no att
        # output = self.output_layer(torch.mean(output_h, dim=1).squeeze(1))
        return output


class lstm(nn.Module):
    def __init__(self,
                 input_size=256,
                 hidden_size=128
                 ):
        super(lstm, self).__init__()
        # D = 1, num_layers = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.rnn = LayerNormLSTM(input_size, hidden_size)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x:        N * L * H_in
        # output:   N * L * H_out
        # hn, cn:   1 * N * H_out
        h0 = torch.zeros(1, x.shape[0], self.hidden_size).to(DEVICE)
        c0 = torch.zeros(1, x.shape[0], self.hidden_size).to(DEVICE)
        # seq_len = [16] * x.shape[0]
        # output, (hn, cn) = self.rnn(x.permute(1, 0, 2), (h0, c0), seq_len)
        # output = output.permute(1, 0, 2)
        output, (hn, cn) = self.rnn(x, (h0, c0))
        return output, hn



class MODEL(torch.nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.d_l = 256
        self.alpha = 0.2
        self.beta = 0.6
        self.att_lstm = att_lstm(self.d_l)
        self.proj_v = nn.Sequential(
            nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.d_l),
            nn.ReLU()
        )
        self.proj_l = nn.Sequential(
            nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.d_l),
            nn.ReLU()
        )
        self.v_1conv = nn.Sequential(
            nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(self.d_l),
        )
        self.l_1conv = nn.Sequential(
            nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(self.d_l),
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
        #self.fusion_category_2 = fusion_category(self.d_l, self.d_l)
        # self.fusion_category = nn.Sequential(
        #     nn.Linear(self.d_l * 3, self.d_l),
        #     #nn.ReLU(),
        #     nn.Dropout(0.25)
        # )


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
        text_emb = (1 - self.beta) * text_emb + self.beta * text_res
        user_id = self.user_emb(user.squeeze(2)[:, :, 0].long())
        user_des = self.proj_u(user.squeeze(2)[:, :, 1: USER_DIM + 1])
        cat = self.cat_emb(category[:, :, 0])
        subcat = self.subcat_emb(category[:, :, 1])
        concept = self.concept_emb(category[:, :, 2])
        category_emb = self.fusion_category_2(self.fusion_category_1(cat, subcat), concept)
        #category_emb = self.fusion_category(torch.cat([cat, subcat, concept], dim=-1))
        #category_emb = concept
        #category_emb = cat + subcat + concept
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
        text_emb = (1 - self.beta) * text_emb + self.beta * text_res
        user_id = self.user_emb(user.squeeze(2)[:, :, 0].long())
        user_des = self.proj_u(user.squeeze(2)[:, :, 1: USER_DIM + 1])
        cat = self.cat_emb(category[:, :, 0])
        subcat = self.subcat_emb(category[:, :, 1])
        concept = self.concept_emb(category[:, :, 2])
        category_emb = self.fusion_category_2(self.fusion_category_1(cat, subcat), concept)
        #category_emb = self.fusion_category(torch.cat([cat, subcat, concept], dim=-1))
        #category_emb = concept
        #category_emb = cat + subcat + concept
        output = self.att_lstm.test(visual_emb, text_emb, user_id, user_des, category_emb)

        return output
