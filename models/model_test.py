import logging
import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss, L1Loss, SmoothL1Loss
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
        self.norm = nn.LayerNorm(self.output_size)
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
        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x
        x = self.fc_1(x)
        x = self.elu(x)
        x = self.fc_2(x)
        x = self.gate(x)
        x = x + residual
        x = self.norm(x)
        return x


class concat(nn.Module):
    def __init__(self, in_size, num, output_dim):
        super(concat, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_size * num, in_size * (num // 2)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_size * (num // 2), output_dim)
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
        # self.out = nn.Sequential(
        #     nn.Linear(self.this_dim * 2, self.this_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.this_dim, self.this_dim)
        # )

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
        # f_this_fusion = self.out(f_this_fusion)
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
        self.attn = nn.MultiheadAttention(self.d_out, self.nheads, batch_first=True)
        self.norm = nn.LayerNorm(self.d_out)

    def forward(self, x, mask):
        residual = x
        q = x[:,[-1],:]
        k = x[:,:-1,:]
        v = x[:,:-1,:]
        #print(F.softmax(torch.dot(q, k.permute(0, 2, 1)), dim=-1))
        mask = mask.tile((x.shape[0] * self.nheads, 1, 1)).to(DEVICE)
        attn_x, attn_w = self.attn(q, k, v)
        attn_x = self.norm(attn_x + residual)
        return attn_x, attn_w

class MapBasedMultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.act = nn.ReLU()
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()

        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)

        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)

        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2)  # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3])  # [(n*b), lq, lk, dk]

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1)  # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3])  # [(n*b), lq, lk, dk]

        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

    
        # Map based Attention
        # output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3)  # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3)  # [(n*b), lq, lk]

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x lq x lk
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_q, l_k]

        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn
    
class att_lstm(nn.Module):
    def __init__(self, dim):
        super(att_lstm, self).__init__()
        self.d_l = dim
        self.d_out = dim
        self.dim_k = dim
        self.attn_heads = 4
        self.fusion = concat(self.d_l, 5, self.d_l)
        self.lstm = lstm(self.d_l, self.d_out)
        self.post_lstm_grn = grn(self.d_out, self.d_out, self.d_out)
        self.post_lstm_gate = gate(self.d_out)
        self.post_lstm_norm = nn.LayerNorm(self.d_out)

        self.multihead_attn = MapBasedMultiHeadAttention(n_head=self.attn_heads, 
                                                         d_model=self.d_out, 
                                                         d_k=self.d_out // self.attn_heads, 
                                                         d_v=self.d_out // self.attn_heads, 
                                                         dropout=0.25)
        self.fc_output = nn.Sequential(
            nn.Linear(self.d_out * 3, self.d_out),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.d_out, 1)
        )
        # self.attn_mask = self.generate_square_subsequent_mask(SEQ_LEN)


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
        #loss_fct = L1Loss()
        #loss_fct = SmoothL1Loss()
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
        lstm_input = self.fusion([visual, text, category])
        lstm_output, hn = self.lstm(lstm_input)
        #lstm_output = lstm_input
        lstm_output = self.post_lstm_gate(lstm_output)
        attn_input = self.post_lstm_norm(lstm_input + lstm_output)
        attn_input = self.post_lstm_grn(lstm_output)
        # attn_output = attn_input
        q, k, v = attn_input[:, [-1], :], attn_input[:, :-1, :], attn_input[:, :-1, :]
        attn_output, attn_w = self.multihead_attn(q, k, v)

        fc_input = torch.cat([attn_output[:, -1, :], user_id[:, -1, :], user_des[:, -1, :]], dim=-1)
        #fc_input = self.fc_norm(fc_input)
        #fc_input = self.pre_fc_grn(fc_input)
        output = self.fc_output(fc_input)

        # no att
        # output = self.output_layer(torch.mean(output_h, dim=1).squeeze(1))
        loss = self.loss_function(output, label)
        return loss, output

    def test(
            self,
            visual,
            text,
            user_id,
            user_des,
            category
    ):
        lstm_input = self.fusion([visual, text, category])
        lstm_output, hn = self.lstm(lstm_input) 
        lstm_output = self.post_lstm_gate(lstm_output)
        attn_input = self.post_lstm_norm(lstm_input + lstm_output)
        attn_input = self.post_lstm_grn(lstm_output)
        # attn_output = attn_input
        # attn_w = torch.zeros(1, 1).to(DEVICE)
        q, k, v = attn_input[:, [-1], :], attn_input[:, :-1, :], attn_input[:, :-1, :]
        attn_output, attn_w = self.multihead_attn(q, k, v)
        fc_input = torch.cat([attn_output[:, -1, :], user_id[:, -1, :], user_des[:, -1, :]], dim=-1)
        #fc_input = self.fc_norm(fc_input)
        #fc_input = self.pre_fc_grn(fc_input)
        output = self.fc_output(fc_input)
        # no att
        # output = self.output_layer(torch.mean(output_h, dim=1).squeeze(1))
        return output, attn_w


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
            nn.ReLU()
        )
        self.proj_l = nn.Sequential(
            nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )
        self.v_1conv = nn.Sequential(
            nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=1, stride=1, bias=False),
        )
        self.l_1conv = nn.Sequential(
            nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=1, stride=1, bias=False),
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
        self.fusion_category = nn.Sequential(
            nn.Linear(self.d_l * 3, self.d_l),
            #nn.ReLU(),
            nn.Dropout(0.25)
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
