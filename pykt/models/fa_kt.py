import torch
import torch.nn as nn
import torch.nn.functional as F
import math, ast
import numpy as np
from torch.nn.init import xavier_uniform_, constant_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CnnExpert(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv = CausalConv1d(d_model, d_model, kernel_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_conv = x.permute(0, 2, 1)
        x_conv = self.activation(self.conv(x_conv))
        return x_conv.permute(0, 2, 1)

class LstmExpert(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, 1, batch_first=True, dropout=0)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

class AttentionExpert(nn.Module):
    
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_model, d_feature, n_heads, dropout, kq_same)
        
    def forward(self, query, key, values, mask=None):
        
        batch_size, seq_len, _ = query.size()
        if mask is None:
           
            nopeek_mask = np.triu(np.ones((1, 1, seq_len, seq_len)), k=0).astype('uint8')
            src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)
        else:
            src_mask = mask
            
        
        attn_output = self.multihead_attn(query, key, values, mask=src_mask, zero_pad=True)
        return attn_output

class MambaExpert(nn.Module): 
    def __init__(self, d_model, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        from mamba_ssm import Mamba
        self.mamba = Mamba(d_model=d_model, d_state=mamba_d_state,d_conv=mamba_d_conv, expand=mamba_expand)
            
    def forward(self, x):
        return self.mamba(x)

class MoETransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same,
                 use_moe=True, num_experts=4, confidence_thresholds=None, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 min_experts=1, max_experts=None, single_expert_type=None):
        super().__init__()
        self.use_moe = use_moe
        self.d_model = d_model
        self.num_experts = num_experts
        self.min_experts = min_experts
        self.max_experts = max_experts or num_experts
        self.single_expert_type = single_expert_type  
        
        
        if confidence_thresholds is None:
            self.confidence_thresholds = [0.8, 0.6, 0.4] 
        else:
            self.confidence_thresholds = ast.literal_eval(confidence_thresholds) if isinstance(confidence_thresholds, str) else confidence_thresholds

        
        if single_expert_type:
            self.use_moe = False 
            if single_expert_type == 'attention':
                self.single_expert = AttentionExpert(d_model, d_feature, n_heads, dropout, kq_same)
            elif single_expert_type == 'lstm':
                self.single_expert = LstmExpert(d_model, dropout)
            elif single_expert_type == 'mamba':
                self.single_expert = MambaExpert(d_model, mamba_d_state, mamba_d_conv, mamba_expand)
            elif single_expert_type == 'cnn':
                self.single_expert = CnnExpert(d_model)
            else:
                raise ValueError(f"Unknown single expert type: {single_expert_type}")
            
            
            if single_expert_type != 'attention':
                self.concat_proj = nn.Linear(2*d_model, d_model, bias=False)
                
        elif use_moe:
            
            self.expert_dict = nn.ModuleDict({
                'cnn': CnnExpert(d_model),
                'lstm': LstmExpert(d_model, dropout),
                'attention': AttentionExpert(d_model, d_feature, n_heads, dropout, kq_same),
                'mamba': MambaExpert(d_model, mamba_d_state, mamba_d_conv, mamba_expand)
            })
            
            
            self.gate = nn.Linear(d_model, len(self.expert_dict))
            
           
            self.concat_proj = nn.Linear(2*d_model, d_model, bias=False)
        else:
           
            self.default_expert = AttentionExpert(d_model, d_feature, n_heads, dropout, kq_same)

        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def compute_confidence_metrics(self, routing_probs):
        
        B, L, E = routing_probs.shape
        
        entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-8), dim=-1)
        normalized_entropy = entropy / math.log(E)
        
        confidence_scores = 1 - normalized_entropy
        
        selected_k_values = self.determine_expert_count(confidence_scores)
        
        return confidence_scores, selected_k_values
    
    def determine_expert_count(self, confidence_scores):
       
        B, L = confidence_scores.shape
        k_values = torch.ones(B, L, dtype=torch.long, device=confidence_scores.device)
        
        
        high_conf_mask = confidence_scores >= self.confidence_thresholds[0]
        k_values[high_conf_mask] = self.min_experts
        
        
        medium_conf_mask = (confidence_scores >= self.confidence_thresholds[1]) & \
                           (confidence_scores < self.confidence_thresholds[0])
        k_values[medium_conf_mask] = min(2, self.max_experts)
        
        
        low_conf_mask = (confidence_scores >= self.confidence_thresholds[2]) & \
                        (confidence_scores < self.confidence_thresholds[1])
        k_values[low_conf_mask] = min(3, self.max_experts)
        
        
        very_low_conf_mask = confidence_scores < self.confidence_thresholds[2]
        k_values[very_low_conf_mask] = self.max_experts
        
        return k_values

    def adaptive_expert_selection(self, query, key, values, routing_probs):
        
        B, L, d_model = query.shape
        E = len(self.expert_dict)
        
       
        confidence_scores, k_values = self.compute_confidence_metrics(routing_probs)
        
        
        current_query = query  
        historical_values = F.pad(values[:, :-1, :], (0, 0, 1, 0), "constant", 0)
        non_attention_input = self.concat_proj(
            torch.cat([current_query, historical_values], dim=-1)
        )

        
        expert_outputs = torch.zeros(B, L, E, d_model, device=query.device)
        expert_names = list(self.expert_dict.keys())
        
        for i, expert_name in enumerate(expert_names):
            if expert_name == 'attention':
                expert_outputs[:, :, i, :] = self.expert_dict[expert_name](query, key, values)
            else:
                expert_outputs[:, :, i, :] = self.expert_dict[expert_name](non_attention_input)
        
       
        max_k = self.max_experts
        top_weights, top_indices = torch.topk(routing_probs, max_k, dim=-1)  # [B, L, max_k]
        
        position_mask = torch.arange(max_k, device=query.device).unsqueeze(0).unsqueeze(0) < k_values.unsqueeze(-1)  # [B, L, max_k]
        
        masked_weights = top_weights * position_mask.float()
        
        weight_sums = masked_weights.sum(dim=-1, keepdim=True)  # [B, L, 1]
        normalized_weights = masked_weights / (weight_sums + 1e-8)
        
        batch_indices = torch.arange(B, device=query.device).view(B, 1, 1).expand(B, L, max_k)
        seq_indices = torch.arange(L, device=query.device).view(1, L, 1).expand(B, L, max_k)
        
        selected_expert_outputs = expert_outputs[batch_indices, seq_indices, top_indices]
        
        moe_output = (selected_expert_outputs * normalized_weights.unsqueeze(-1)).sum(dim=2)
        
        
        expert_usage_stats = torch.zeros(E, device=query.device)
        for i in range(E):
            expert_mask = (top_indices == i)  # [B, L, max_k]
            expert_weight = (normalized_weights * expert_mask.float()).sum()
            expert_usage_stats[i] = expert_weight
        
        total_weight = expert_usage_stats.sum()
        if total_weight > 0:
            expert_usage_stats = expert_usage_stats / total_weight
        
        return moe_output, confidence_scores, expert_usage_stats

    def forward(self, mask, query, key, values, apply_pos=True):
        batch_size, seqlen, d_model = query.size()
        device = query.device

        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        zero_pad = (mask == 0)

        residual = query
        
        if self.single_expert_type:
            if self.single_expert_type == 'attention':
                output = self.single_expert(query, key, values, mask=src_mask)
            else:
                current_query = query
                historical_values = F.pad(values[:, :-1, :], (0, 0, 1, 0), "constant", 0)
                non_attention_input = self.concat_proj(
                    torch.cat([current_query, historical_values], dim=-1)
                )
                output = self.single_expert(non_attention_input)
            
           
            stats = {
                'single_expert_type': self.single_expert_type,
                'confidence_scores': torch.ones(batch_size, seqlen, device=device),  
                'expert_usage_stats': torch.zeros(4, device=device) 
            }
           
            expert_map = {'attention': 0, 'lstm': 1, 'cnn': 2, 'mamba': 3}
            if self.single_expert_type in expert_map:
                stats['expert_usage_stats'][expert_map[self.single_expert_type]] = 1.0
                
        elif self.use_moe:
            router_logits = self.gate(query)
            routing_probs = F.softmax(router_logits, dim=-1)
            
            moe_output, confidence_scores, expert_stats = self.adaptive_expert_selection(
                query, key, values, routing_probs)
            
            output = moe_output
            stats = {
                'confidence_scores': confidence_scores,
                'expert_usage_stats': expert_stats,
                'routing_probs': routing_probs
            }
        else:
            if zero_pad:
                output = self.default_expert(query, key, values, mask=src_mask)
            else:
                output = self.default_expert(query, key, values, mask=src_mask)
            stats = None
        output = residual + self.dropout1(output)
        output = self.layer_norm1(output)

        if apply_pos:
            ffn_output = self.linear2(self.dropout(self.activation(self.linear1(output))))
            output = output + self.dropout2(ffn_output)
            output = self.layer_norm2(output)
        
        return output, stats


class FA_KT(nn.Module):

    def __init__(self, n_question, n_pid, num_rgap, num_sgap, num_pcount,
             d_model, n_blocks, dropout, d_ff=256,
             loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2,
             nheads=4, seq_len=200, kernel_size1=3, kernel_size2=3, freq=True,
             kq_same=1, final_fc_dim=512, final_fc_dim2=256,
             num_attn_heads=8, separate_qa=False, l1=0.1,
             emb_type="qid", emb_path="", pretrain_dim=768,
             use_moe=True, num_experts=4, 
             confidence_thresholds=[0.8, 0.6, 0.4],
             mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
             min_experts=1, max_experts=None):
        super().__init__()
        
        self.model_name = "fa_kt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l1 = l1
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        
        if isinstance(confidence_thresholds, str):
            self.confidence_thresholds = ast.literal_eval(confidence_thresholds)
        else:
            self.confidence_thresholds = confidence_thresholds
        self.min_experts = min_experts
        self.max_experts = max_experts or num_experts

        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, embed_l)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

       
        single_expert_type = None
        if emb_type.find("onlyattn") != -1:
            single_expert_type = 'attention'
        elif emb_type.find("onlylstm") != -1:
            single_expert_type = 'lstm'
        elif emb_type.find("onlymamba") != -1:
            single_expert_type = 'mamba'
        elif emb_type.find("onlycnn") != -1:
            single_expert_type = 'cnn'
        
        
        if single_expert_type:
            actual_use_moe = False
        else:
            disable_moe = emb_type.find("nomoe") != -1
            actual_use_moe = use_moe and not disable_moe

        self.model = Architecture(
            n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads,
            dropout=dropout, d_model=d_model, d_feature=d_model // num_attn_heads,
            d_ff=d_ff, kq_same=self.kq_same, model_type=self.model_type,
            seq_len=seq_len, emb_type=emb_type, kernel_size1=kernel_size1, kernel_size2=kernel_size2,
            freq=freq, use_moe=actual_use_moe, num_experts=num_experts,
            confidence_thresholds=confidence_thresholds,
            mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv, mamba_expand=mamba_expand,
            min_experts=min_experts, max_experts=max_experts,
            single_expert_type=single_expert_type 
        ).to(device)

       
        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim), 
            nn.ReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), 
            nn.ReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )

        
        self.time_emb = timeGap(num_rgap, num_sgap, num_pcount, d_model)
        self.model2 = Architecture(
            n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads,
            dropout=dropout, d_model=d_model, d_feature=d_model / num_attn_heads,
            d_ff=d_ff, kq_same=self.kq_same, model_type=self.model_type,
            seq_len=seq_len, emb_type=emb_type, kernel_size1=kernel_size1, kernel_size2=kernel_size2,
            freq=freq, use_moe=actual_use_moe, num_experts=num_experts,
            confidence_thresholds=confidence_thresholds,
            mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv, mamba_expand=mamba_expand,
            min_experts=min_experts, max_experts=max_experts,
            single_expert_type=single_expert_type 
        )
        
        self.c_weight = nn.Linear(d_model, d_model)
        self.t_weight = nn.Linear(d_model, d_model)
        
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, dcur, dgaps, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        
        pid_data = torch.cat((q[:, 0:1], qshft), dim=1)
        q_data = torch.cat((c[:, 0:1], cshft), dim=1)
        target = torch.cat((r[:, 0:1], rshft), dim=1)

        emb_type = self.emb_type
        q_data = q_data.to(device)
        target = target.to(device)

        
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data.to(device))
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

       
        model_stats = {}
        y2, y3 = 0, 0
        
        if emb_type.startswith("qid"):
           
            rg, sg, p = dgaps["rgaps"].long(), dgaps["sgaps"].long(), dgaps["pcounts"].long()
            rgshft, sgshft, pshft = dgaps["shft_rgaps"].long(), dgaps["shft_sgaps"].long(), dgaps["shft_pcounts"].long()
            r_gaps = torch.cat((rg[:, 0:1], rgshft), dim=1)
            s_gaps = torch.cat((sg[:, 0:1], sgshft), dim=1)
            pcounts = torch.cat((p[:, 0:1], pshft), dim=1)
            temb = self.time_emb(r_gaps, s_gaps, pcounts)
            
            
            d_output, d_stats = self.model(q_embed_data, qa_embed_data)
            t_output, t_stats = self.model2(temb, qa_embed_data)
            
            if d_stats:
                model_stats.update({f'd_{k}': v for k, v in d_stats.items()})
            if t_stats:
                model_stats.update({f't_{k}': v for k, v in t_stats.items()})
            
            
            w = torch.sigmoid(self.c_weight(d_output) + self.t_weight(t_output))
            final_output = w * d_output + (1 - w) * t_output
                
            q_embed_data = q_embed_data + temb
            
            concat_q = torch.cat([final_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            preds = torch.sigmoid(output)
        else:
            raise ValueError(f"Unsupported emb_type: {emb_type}")

        if train:
            return preds, y2, y3
        else:
            return preds, concat_q if qtest else preds


class Architecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
             d_ff, n_heads, dropout, kq_same, model_type, seq_len, emb_type, 
             kernel_size1, kernel_size2, freq, use_moe=False, num_experts=4,
             confidence_thresholds=[0.8, 0.6, 0.4],
             mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
             min_experts=1, max_experts=None, single_expert_type=None):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type
        self.emb_type = emb_type
        self.freq = freq
        self.single_expert_type = single_expert_type  
        
        # MoE Transformer 
        if model_type in {'fa_kt'}:
            self.blocks_2 = nn.ModuleList([
                MoETransformerLayer(
                    d_model=d_model, d_feature=d_model // n_heads,
                    d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same,
                    use_moe=use_moe, num_experts=num_experts,
                    confidence_thresholds=confidence_thresholds,
                    mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv, mamba_expand=mamba_expand,
                    min_experts=min_experts, max_experts=max_experts,
                    single_expert_type=single_expert_type 
                ) for _ in range(n_blocks)
            ])
            
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)
        
        if emb_type.find("true") != -1:
            pass
        elif emb_type.find("band") != -1:
            self.freq_enhancer = ThreeBandFrequencyLayer(dropout, d_model, [kernel_size1, kernel_size2])
        else:
            pass

    def forward(self, q_embed_data, qa_embed_data):
        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb
        
        y = qa_embed_data
        x = q_embed_data
        
        if self.emb_type.find("true") != -1:
            x = self.filter_layer(x)
            y = self.filter_layer(y)
        elif self.emb_type.find("band") != -1:
            x, x_freq_bands = self.freq_enhancer(x)
            y, y_freq_bands = self.freq_enhancer(y)
        else:
            x = self.filter_layer(x)
            y = self.filter_layer(y)
        
        
        layer_stats_list = []
        
        for i, block in enumerate(self.blocks_2):
            x, block_stats = block(mask=0, query=x, key=x, values=y, apply_pos=True)
            
            if block_stats:
                layer_stats_list.append({f'layer_{i}_{k}': v for k, v in block_stats.items()})
        
        
        combined_stats = {}
        for stats in layer_stats_list:
            combined_stats.update(stats)
        
        return x, combined_stats



class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(1), :]

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation,
                              groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding > 0 else x

class ThreeBandFrequencyLayer(nn.Module):
    def __init__(self, dropout, hidden_size, kernel_sizes=[5, 15], filter_type='hamming'):
        super(ThreeBandFrequencyLayer, self).__init__()
        self.hidden_size = hidden_size
        self.out_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        self.kernel_sizes = sorted(kernel_sizes)
        
        self.lpf_stage1 = CausalConv1d(
            hidden_size, hidden_size, self.kernel_sizes[0], 
            groups=hidden_size, bias=False
        )
        self.lpf_stage2 = CausalConv1d(
            hidden_size, hidden_size, self.kernel_sizes[1], 
            groups=hidden_size, bias=False
        )
        
        init_lowpass_filter(self.lpf_stage1.conv, filter_type, cutoff_ratio=0.6)
        init_lowpass_filter(self.lpf_stage2.conv, filter_type, cutoff_ratio=0.1)
        
        self.band_weights = nn.Parameter(torch.tensor([0.1, 0.3, 0.6]))
        self.gammas = nn.Parameter(torch.ones(3))
        
        self.enable_band_transform = False
        if self.enable_band_transform:
            self.hf_transform = nn.Linear(hidden_size, hidden_size, bias=False)
            self.mf_transform = nn.Linear(hidden_size, hidden_size, bias=False)
            self.lf_transform = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, input_tensor):
        x = input_tensor.permute(0, 2, 1)
        
        lpf1_out = self.lpf_stage1(x)
        lpf2_out = self.lpf_stage2(lpf1_out)
        
        lpf1_out = lpf1_out.permute(0, 2, 1)
        lpf2_out = lpf2_out.permute(0, 2, 1)
        
        high_freq = input_tensor - lpf1_out
        mid_freq = lpf1_out - lpf2_out
        low_freq = lpf2_out
        
        with torch.no_grad():
            hf_std = high_freq.std(dim=(0, 1), keepdim=True) + 1e-8
            mf_std = mid_freq.std(dim=(0, 1), keepdim=True) + 1e-8
            lf_std = low_freq.std(dim=(0, 1), keepdim=True) + 1e-8
            
        high_freq = high_freq / hf_std
        mid_freq = mid_freq / mf_std
        low_freq = low_freq / lf_std
        
        high_freq = self.gammas[0] * high_freq
        mid_freq = self.gammas[1] * mid_freq
        low_freq = self.gammas[2] * low_freq
        
        if self.enable_band_transform:
            high_freq = self.hf_transform(high_freq)
            mid_freq = self.mf_transform(mid_freq)
            low_freq = self.lf_transform(low_freq)
        
        weights = F.softmax(self.band_weights, dim=0)
        combined = weights[0] * high_freq + weights[1] * mid_freq + weights[2] * low_freq
        
        output = self.out_dropout(combined)
        output = self.layer_norm(output + input_tensor)
        
        decomposed_bands = (high_freq, mid_freq, low_freq)
        return output, decomposed_bands

def init_lowpass_filter(conv_layer, filter_type='hamming', cutoff_ratio=0.5):
    if not hasattr(conv_layer, 'weight'):
        return
        
    kernel_size = conv_layer.kernel_size[0]
    
    if filter_type == 'hamming':
        window = np.hamming(kernel_size)
    elif filter_type == 'blackman':
        window = np.blackman(kernel_size)
    else:
        window = np.ones(kernel_size)

    n = np.arange(kernel_size)
    sinc_filter = np.sinc(2 * cutoff_ratio * (n - (kernel_size - 1) / 2))
    
    lowpass_kernel = sinc_filter * window
    lowpass_kernel = lowpass_kernel / np.sum(lowpass_kernel)
    
    with torch.no_grad():
        kernel_tensor = torch.tensor(lowpass_kernel, dtype=torch.float32)
        conv_layer.weight.data = kernel_tensor.view(1, 1, kernel_size).repeat(conv_layer.out_channels, 1, 1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)
        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k, q, v = k.transpose(1, 2), q.transpose(1, 2), v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output

def attention(q, k, v, d_k, mask, dropout, zero_pad):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(scores.size(0), scores.size(1), 1, scores.size(3), device=scores.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class timeGap(nn.Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, emb_size):
        super().__init__()
        self.num_rgap, self.num_sgap, self.num_pcount = num_rgap, num_sgap, num_pcount
        
        if num_rgap != 0:
            self.register_buffer('rgap_eye', torch.eye(num_rgap))
        if num_sgap != 0:
            self.register_buffer('sgap_eye', torch.eye(num_sgap))
        if num_pcount != 0:
            self.register_buffer('pcount_eye', torch.eye(num_pcount))
            
        input_size = num_rgap + num_sgap + num_pcount
        print(f"self.num_rgap: {self.num_rgap}, self.num_sgap: {self.num_sgap}, "
              f"self.num_pcount: {self.num_pcount}, input_size: {input_size}")
        self.time_emb = nn.Linear(input_size, emb_size, bias=False)

    def forward(self, rgap, sgap, pcount):
        infs = []
        if self.num_rgap != 0:
            rgap = self.rgap_eye[rgap]
            infs.append(rgap)
        if self.num_sgap != 0:
            sgap = self.sgap_eye[sgap]
            infs.append(sgap)
        if self.num_pcount != 0:
            pcount = self.pcount_eye[pcount]
            infs.append(pcount)
        tg = torch.cat(infs, -1)
        tg_emb = self.time_emb(tg)
        return tg_emb