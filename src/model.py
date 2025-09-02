#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import math

from src.Roberta import MultiHeadAttention, InteractionAttention
from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
from itertools import accumulate
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.2, d_out=None):
        super(PositionwiseFeedForward, self).__init__()
        if d_out is None: d_out = d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))



class InteractLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2, config=None):
        super(InteractLayer, self).__init__()
        head_size = int(d_model / num_heads)
        self.config = config
        self.interactionAttention = InteractionAttention(num_heads, d_model, head_size, head_size, dropout,
                                                         config=config)

        self.layer_norm_pre = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.layer_norm_post = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, global_x, mask, sentence_length, ):
        x = self.layer_norm_pre(self.interactionAttention(x, global_x, mask, )[0] + x)
        x = self.layer_norm_post(self.ffn(x) + x)
        x = self.dropout(x)
        return x


class GCN(nn.Module):
    def __init__(self, config, layer_num, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.config = config
        self.layer_list = nn.ModuleList()
        for i in range(layer_num):
            if i == layer_num - 1:
                self.layer_list.append(nn.Linear(hidden_dim, output_dim))
            elif i == 0:
                self.layer_list.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layer_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.gnn_dropout = nn.Dropout(dropout)
        self.gnn_activation = F.gelu

    def forward(self, x, mask, adj):
        D_hat = torch.diag_embed(torch.pow(torch.sum(adj, dim=-1), -1))
        if torch.isinf(D_hat).any():
            D_hat[torch.isinf(D_hat)] = 0.0
        adj = torch.matmul(D_hat, adj)
        # adj = torch.matmul(adj, D_hat)

        x_mask = mask.unsqueeze(-1)  # .expand(-1, -1, x.size(-1))
        for i, layer in enumerate(self.layer_list):
            if i != 0:
                x = self.gnn_dropout(x)
            x = torch.matmul(x, layer.weight.T) + layer.bias
            x = torch.matmul(adj, x)
            x = x * x_mask
            x = self.gnn_activation(x)

        return x


class AttentionScoreLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.2):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size // 2)
        self.key = nn.Linear(hidden_size, hidden_size // 2)
        self.score = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size // 2)

    def forward(self, x):
        # x: [seq_len, hidden_size]
        q = self.query(x)  # [seq_len, hidden_size//2]
        k = self.key(x)  # [seq_len, hidden_size//2]

        # 计算每个token对整个句子的注意力
        global_k = torch.mean(k, dim=0, keepdim=True)  # [1, hidden_size//2]
        attention = q * global_k  # [seq_len, hidden_size//2]
        attention = self.norm(attention)
        attention = self.dropout(torch.tanh(attention))
        scores = self.score(attention)

        return scores


class EnhancedUttRepresentation(nn.Module):
    def __init__(self, hidden_size, dropout=0.2):
        super().__init__()
        self.input_dim = hidden_size * 5
        self.hidden_dim = hidden_size * 2
        self.output_dim = hidden_size

        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, token_repr, token_weights, speaker_repr):
        # token_repr: [k, hidden_size]
        # token_weights: [k]
        # speaker_repr: [hidden_size]

        weighted_avg = torch.sum(token_repr * token_weights.unsqueeze(-1), dim=0)
        simple_avg = torch.mean(token_repr, dim=0)
        max_pool = torch.max(token_repr, dim=0)[0]
        weighted_max_idx = torch.argmax(token_weights)
        weighted_max = token_repr[weighted_max_idx]

        combined = torch.cat([
            weighted_avg, simple_avg, max_pool,
            weighted_max, speaker_repr
        ], dim=-1)

        return self.projection(combined)


class BertWordPair(nn.Module):
    def __init__(self, cfg):
        super(BertWordPair, self).__init__()
        self.ipw_mask = None
        self.cfg = cfg
        self.bert = AutoModel.from_pretrained(cfg.bert_path, output_hidden_states=True)
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)
        self.bert_config = bert_config

        bh = bert_config.hidden_size
        nhead = bert_config.num_attention_heads
        att_head_size = int(bh / nhead)

        self.cfg.loss_weight = {'ent': int(self.cfg.loss_w[0]), 'rel': int(self.cfg.loss_w[1]),
                                'pol': int(self.cfg.loss_w[2])}

        self.inner_dim = 256
        self.ent_dim = self.inner_dim * 4 * 4
        self.rel_dim = self.inner_dim * 4 * 3
        self.pol_dim = self.inner_dim * 4 * 4

        self.dense_all = nn.ModuleDict({
            'ent': nn.Linear(bh, self.ent_dim),
            'rel': nn.Linear(bh, self.rel_dim),
            'pol': nn.Linear(bh, self.pol_dim),
        })

        self.dense_aspects = nn.Linear(bh, bh, bias=False)  # 为了模型自己放缩
        self.dense_pols = nn.Linear(bh, bh, bias=False)
        self.dense_rels = nn.Linear(bh, bh, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        self.TSAF = nn.ModuleDict({
            'ent': InteractLayer(
                bert_config.hidden_size,
                bert_config.num_attention_heads,
                bert_config.hidden_dropout_prob,
                cfg
            ),
            'rel': InteractLayer(
                bert_config.hidden_size,
                bert_config.num_attention_heads,
                bert_config.hidden_dropout_prob,
                cfg
            ),
            'pol': InteractLayer(
                bert_config.hidden_size,
                bert_config.num_attention_heads,
                bert_config.hidden_dropout_prob,
                cfg
            ),
        })


        self.layernorm = nn.LayerNorm(bh, eps=1e-12)
        self.syngcn = GCN(cfg, cfg.gnn_layer_num, bh, bh, bh, cfg.gnn_dropout)

        self.semgcn = GCN(cfg, cfg.gnn_layer_num, bh, bh, bh, cfg.gnn_dropout)
        self.semantic_attention = MultiHeadAttention(bert_config.num_attention_heads, bh, att_head_size, att_head_size,
                                                     bert_config.attention_probs_dropout_prob)

        # topk
        # self.topK_select_layer = nn.Linear(bh, 1)
        self.topK_select_layer = AttentionScoreLayer(bh, cfg.dropout)
        # self.utt_linear = nn.Linear(3 * bh, bh)
        self.utt_representation = EnhancedUttRepresentation(bh, cfg.dropout)

        self.dscgcn = GCN(cfg, cfg.dscgnn_layer_num, bh, bh, bh, cfg.gnn_dropout)
        self.global_layernorm = nn.LayerNorm(bh, eps=1e-12)

    def expand_and_flatten_tensor(self, tensor):
        """
        tensor -- (batch, len)
        """
        batch_size, original_len = tensor.size()
        expanded_tensors = []

        for i in range(batch_size):
            vec = tensor[i].cpu().numpy()
            k = 0
            for j in reversed(range(original_len)):
                if vec[j] == 0:
                    k += 1
                else:
                    break

            expanded_vec = []
            for idx in range(original_len):
                if idx >= original_len - k:
                    expanded_vec.extend([0] * original_len)
                else:
                    expanded_vec.extend([vec[idx]] * (original_len - k))
                    expanded_vec.extend([0] * k)

            expanded_tensors.append(expanded_vec)

        flattened_tensor = torch.tensor(expanded_tensors, dtype=tensor.dtype).view(-1)

        return flattened_tensor

    def custom_sinusoidal_position_embedding(self, token_index, pos_type):
        """
        See RoPE paper: https://arxiv.org/abs/2104.09864
        https://blog.csdn.net/weixin_43646592/article/details/130924280
        """
        output_dim = self.inner_dim
        position_ids = token_index.unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(self.cfg.device)  # 128
        if pos_type == 0:
            indices = torch.pow(10000, -2 * indices / output_dim)
        else:
            indices = torch.pow(15, -2 * indices / output_dim)
        embeddings = position_ids * indices  # [seq_len, 128]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)  # [seq_len, 128, 2]
        embeddings = embeddings.repeat((1, *([1] * len(embeddings.shape))))  # [1, seq_len, 128, 2]
        embeddings = torch.reshape(embeddings, (1, len(token_index), output_dim))  # [1, seq_len, 256]
        embeddings = embeddings.squeeze(0)
        return embeddings

    def get_instance_embedding(self, qw: torch.Tensor, kw: torch.Tensor, token_index, thread_length, pos_type):
        """_summary_
        Parameters
        ----------
        qw : torch.Tensor, (seq_len, class_nums, hidden_size)
        kw : torch.Tensor, (seq_len, class_nums, hidden_size)
        """

        seq_len, num_classes = qw.shape[:2]

        accu_index = [0] + list(accumulate(thread_length))

        logits = qw.new_zeros([seq_len, seq_len, num_classes])

        # Compute the ROPE matrix
        for i in range(len(thread_length)):
            for j in range(len(thread_length)):
                rstart, rend = accu_index[i], accu_index[i + 1]
                cstart, cend = accu_index[j], accu_index[j + 1]

                cur_qw, cur_kw = qw[rstart:rend], kw[cstart:cend]
                x, y = token_index[rstart:rend], token_index[cstart:cend]

                # This is used to compute relative distance, see the matrix in Fig.8 of our paper
                x = - x if i > 0 and i < j else x
                y = - y if j > 0 and i > j else y

                x_pos_emb = self.custom_sinusoidal_position_embedding(x, pos_type)  # 38，256
                y_pos_emb = self.custom_sinusoidal_position_embedding(y, pos_type)

                # Refer to https://kexue.fm/archives/8265
                x_cos_pos = x_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)  # 38， 1， 256
                x_sin_pos = x_pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                cur_qw2 = torch.stack([-cur_qw[..., 1::2], cur_qw[..., ::2]], -1)  # [38, 6, 128, 2]
                cur_qw2 = cur_qw2.reshape(cur_qw.shape)  # [38, 6, 256]
                cur_qw = cur_qw * x_cos_pos + cur_qw2 * x_sin_pos  # [38, 6, 256]

                y_cos_pos = y_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                y_sin_pos = y_pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                cur_kw2 = torch.stack([-cur_kw[..., 1::2], cur_kw[..., ::2]], -1)
                cur_kw2 = cur_kw2.reshape(cur_kw.shape)
                cur_kw = cur_kw * y_cos_pos + cur_kw2 * y_sin_pos

                pred_logits = torch.einsum('mhd,nhd->mnh', cur_qw, cur_kw).contiguous()  # 38 38 6， 38 34 6
                logits[rstart:rend,
                cstart:cend] = pred_logits  # [0:38, 0:38]=[38,38,6] [0:38, 108:142]=[38,34,6] [38:108, 108:142]=[70,34,6]

        return logits

    def get_ro_embedding(self, qw, kw, token_index, thread_lengths, pos_type):
        # qw_res = qw.new_zeros(*qw.shape)
        # kw_res = kw.new_zeros(*kw.shape)
        # qw,kw (batch_size, seq_len, 6, 256)
        logits = []
        batch_size = qw.shape[0]
        for i in range(batch_size):
            pred_logits = self.get_instance_embedding(qw[i], kw[i], token_index[i], thread_lengths[i],
                                                      pos_type)  # [seqlen, seqlen, classnums]
            logits.append(pred_logits)
        logits = torch.stack(logits)
        return logits

    def classify_matrix(self, kwargs, sequence_outputs, input_labels, masks, mat_name='ent', mode='train'):

        utterance_index, token_index, thread_lengths = [kwargs[w] for w in
                                                        ['utterance_index', 'token_index', 'thread_lengths']]

        fusion_knowledge = torch.split(sequence_outputs, self.cfg.output_dim * 4, dim=-1)  # 6*b*sen_length*1024
        fusion_knowledge = torch.stack(fusion_knowledge, dim=-2)  # b*sen_length*6*1024

        q_token, q_utterance, k_token, k_utterance = torch.split(fusion_knowledge, self.cfg.output_dim,
                                                                 dim=-1)  # b*sen_length*6*256

        logits = self.get_ro_embedding(q_token, k_token, token_index, thread_lengths,
                                       pos_type=0)  # pos_type=0 for token-level relative distance encoding
        if mat_name != 'ent':
            logits += self.get_ro_embedding(q_utterance, k_utterance, utterance_index, thread_lengths,
                                            pos_type=1)
        nums = logits.shape[-1]
        active_loss = masks.view(-1) == 1

        if mode == 'train':
            ipw_masks = self.ipw_mask[active_loss]
            criterion = nn.CrossEntropyLoss(sequence_outputs.new_tensor([1.0] +[self.cfg.loss_weight[mat_name]] * (nums - 1)), reduction='none')
        else:
            criterion = nn.CrossEntropyLoss(sequence_outputs.new_tensor([1.0] + [self.cfg.loss_weight[mat_name]] * (nums - 1)))
        active_logits_k = logits.view(-1, logits.shape[-1])[active_loss]
        active_labels = input_labels.view(-1)[active_loss]
        fuison_loss = criterion(active_logits_k, active_labels)
        if mode == 'train':
            fuison_loss = fuison_loss * ipw_masks
            fuison_loss = torch.mean(fuison_loss)
        return fuison_loss, logits



    def merge_sentence(self, sequence_outputs, input_masks, dialogue_length):
        res = []
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            stack = []
            for j in range(s, e):
                lens = input_masks[j].sum()
                stack.append(sequence_outputs[j, :lens])
            res.append(torch.cat(stack))
        new_res = sequence_outputs.new_zeros([len(res), max(map(len, res)), sequence_outputs.shape[-1]])
        for i, w in enumerate(res):
            new_res[i, :len(w)] = w
        return new_res  # batch_size, max_dialogue_length, hidden_size

    def root_merge_sentence(self, sequence_outputs, input_masks, dialogue_length, thread_lengths):
        if self.cfg.root_merge == 0:
            return self.merge_sentence(sequence_outputs, input_masks, dialogue_length)

        res = []
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            stack = []
            root_stack = []
            root_len = thread_lengths[i][0]
            for j in range(s, e):
                lens = input_masks[j].sum()
                root_stack.append(sequence_outputs[j, :root_len])
                stack.append(sequence_outputs[j, root_len:lens])

            root = torch.stack(root_stack).sum(0) / len(root_stack)

            stack = [root] + stack
            res.append(torch.cat(stack))
        new_res = sequence_outputs.new_zeros([len(res), max(map(len, res)), sequence_outputs.shape[-1]])
        for i, w in enumerate(res):
            new_res[i, :len(w)] = w
        return new_res  # batch_size, max_dialogue_length, hidden_size

    def stable_entropy(self, probabilities, eps=1e-12):
        probs = probabilities.clamp(min=eps)
        log_probs = torch.log(probs)
        entropy = -(probs * log_probs).sum()
        return entropy

    def dynamic_k_selection(self, scores, sentence_length):
        if sentence_length <= 0 or scores.numel() == 0:
            return 1

        base_k = max(int(self.cfg.topk * sentence_length), 1)

        normalized_scores = F.softmax(scores, dim=0)

        # Entropy Adjustment
        entropy = self.stable_entropy(normalized_scores)
        max_entropy = math.log(sentence_length) if sentence_length > 1 else 1.0

        entropy_ratio = (entropy / max_entropy).item() if max_entropy > 0 else 0.0
        entropy_ratio = 0.0 if math.isnan(entropy_ratio) else entropy_ratio

        k_adjustment = int(sentence_length * self.cfg.k_alpha * entropy_ratio)

        score_mean = scores.mean().item()
        score_max = scores.max().item()
        # Deviation Control
        if score_max > 2 * score_mean and base_k > 1:
            k_reduction = int(base_k * self.cfg.k_beta)
            base_k = max(base_k - k_reduction, 1)

        final_k = min(max(base_k + k_adjustment, 1), sentence_length)

        return final_k

    def dynamic_k_selector(self, sentence_sequence_outputs, global_masks):
        batch_size, max_dialogue_length, hidden_size = sentence_sequence_outputs.shape
        batch_size, max_sentence_num, max_dialogue_length, _ = global_masks.shape

        sentence_lengths = global_masks.sum(dim=2).squeeze(-1)  # [batch_size, max_sentence_num]
        split_sentences = []

        for i in range(batch_size):
            split_sentences.append([])
            for j in range(max_sentence_num):
                sentence_length = sentence_lengths[i, j]
                if sentence_length > 0:
                    start_index = (global_masks[i, j, :, :] == 1).nonzero()[0, 0].item()
                    end_index = int(start_index + sentence_length.item())
                    token_representation = sentence_sequence_outputs[i, start_index:end_index - 1, :]
                    speaker_representation = sentence_sequence_outputs[i, end_index - 1, :]

                    # Attention Score
                    score = self.topK_select_layer(token_representation).squeeze(-1)

                    k = self.dynamic_k_selection(score, token_representation.size(0))

                    k = min(k, token_representation.size(0))
                    topk_indices = torch.topk(score, k, dim=0, largest=True).indices

                    selected_token_representations = token_representation[topk_indices]
                    selected_scores = torch.softmax(score[topk_indices], dim=0)

                    utt_representation = self.utt_representation(selected_token_representations, selected_scores,
                                                                 speaker_representation)

                    split_sentences[i].append(utt_representation)
                else:
                    split_sentences[i].append(sentence_sequence_outputs.new_zeros([hidden_size]))

        split_sentences = torch.stack([torch.stack(bat) for bat in split_sentences], dim=0)

        return split_sentences

    def dynamic_selective_fusion(self, speaker_ids, sentence_sequence_outputs, global_masks, utterance_level_reply_adj,
                                 utterance_level_speaker_adj, utterance_level_mask):
        # sentence_sequence_outputs: batch_size, max_dialogue_length, hidden_size
        # global_masks: batch_size, max_sentence_num, max_dialogue_length, 1

        utterance_sequence = self.dynamic_k_selector(sentence_sequence_outputs, global_masks)
        global_outputs = self.dscgcn(utterance_sequence, utterance_level_mask, utterance_level_reply_adj)
        global_outputs = self.global_layernorm(utterance_sequence + global_outputs)

        return global_outputs

    def utterance2thread(self, sequence_outputs, thread_idxes, sentence_length, thread_lengths, merged_input_masks):
        # sequence_outputs: batch_size, max_sentence_length, hidden_size
        thread_num, max_thread_len = merged_input_masks.shape

        thread_sequence_output = sequence_outputs.new_zeros([thread_num, max_thread_len, sequence_outputs.shape[-1]])
        thread_idx = 0
        for bat_idx, bat in enumerate(thread_idxes):
            for t_idx, thread in enumerate(bat):
                thread_list = []
                for s_idx, sent_idx in enumerate(thread):
                    thread_list.append(sequence_outputs[bat_idx, :sentence_length[bat_idx][sent_idx], :])
                thread_list = torch.cat(thread_list, dim=0)
                thread_sequence_output[thread_idx, :thread_list.shape[0], :] = thread_list
                thread_idx += 1

        return thread_sequence_output




    def forward(self, mode='train', **kwargs):
        if self.cfg.merged_thread == 0:
            input_ids, input_masks, input_segments = [kwargs[w] for w in ['input_ids', 'input_masks', 'input_segments']]

        sentence_length, thread_idxes, merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length, merged_dialog_length, thread_lengths, adj_matrixes, merged_len \
            = [kwargs[w] for w in
               ['sentence_length', 'thread_idxes', 'merged_input_ids', 'merged_input_masks', 'merged_input_segments',
                'merged_sentence_length', 'merged_dialog_length', 'thread_lengths', 'adj_matrixes', 'merged_len']]

        ent_matrix, rel_matrix, pol_matrix = [kwargs[w] for w in ['ent_matrix', 'rel_matrix', 'pol_matrix']]
        reply_masks, speaker_masks, thread_masks = [kwargs[w] for w in ['reply_masks', 'speaker_masks', 'thread_masks']]
        sentence_masks, full_masks, dialogue_length = [kwargs[w] for w in
                                                       ['sentence_masks', 'full_masks', 'dialogue_length']]
        # backdoor adjustment
        if mode == 'train':
            self.ipw_mask = kwargs['ipw_mask']
            self.ipw_mask = self.expand_and_flatten_tensor(self.ipw_mask).to(self.cfg.device)


        # DO
        # Subtask Guide Encoder
        if self.cfg.merged_thread == 1:
            outputs = self.bert(merged_input_ids, token_type_ids=merged_input_segments,
                                attention_mask=merged_input_masks, output_hidden_states=True)
            sequence_outputs = outputs[0]
            outputs = outputs[2]
            sentence_sequence_outputs = self.root_merge_sentence(sequence_outputs, merged_input_masks,
                                                                 merged_dialog_length, thread_lengths)
        # guide
            pol_outputs = torch.mean(torch.stack(outputs[1:4]), dim=0)
            pol_sequence_outputs = self.root_merge_sentence(pol_outputs, merged_input_masks,
                                                                 merged_dialog_length, thread_lengths)

            rel_outputs = torch.mean(torch.stack(outputs[7:10]), dim=0)
            rel_sequence_outputs = self.root_merge_sentence(rel_outputs, merged_input_masks,
                                                            merged_dialog_length, thread_lengths)

            ent_outputs = torch.mean(torch.stack(outputs[4:7]), dim=0)
            ent_sequence_outputs = self.root_merge_sentence(ent_outputs, merged_input_masks,
                                                                 merged_dialog_length, thread_lengths)
        else:  # w/o thread
            sequence_outputs = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[
                0]  # utterance_num, seq_len, hidden_size
            sentence_sequence_outputs = self.merge_sentence(sequence_outputs, input_masks, dialogue_length)

        sentence_sequence_outputs = self.dropout(sentence_sequence_outputs)

        ent_sequence_outputs = self.dropout(self.dense_aspects(ent_sequence_outputs))
        pol_sequence_outputs = self.dropout(self.dense_pols(pol_sequence_outputs))
        rel_sequence_outputs = self.dropout(self.dense_rels(rel_sequence_outputs))

        if self.cfg.merged_thread == 1:
            syngcn_outputs = self.syngcn(sequence_outputs, merged_input_masks,
                                         adj_matrixes)
            syngcn_outputs = self.root_merge_sentence(syngcn_outputs, merged_input_masks, merged_dialog_length,
                                                      thread_lengths)
        else:  # w/o thread
            syngcn_outputs = self.syngcn(sequence_outputs, input_masks, adj_matrixes)
            syngcn_outputs = self.merge_sentence(syngcn_outputs, input_masks, dialogue_length)

        syngcn_outputs = self.dropout(syngcn_outputs)

        _, semantic_adj = self.semantic_attention(sequence_outputs, sequence_outputs, sequence_outputs)
        semantic_adj = semantic_adj.mean(dim=1)
        if self.cfg.merged_thread == 1:
            semgcn_output = self.semgcn(sequence_outputs, merged_input_masks, semantic_adj)
            semgcn_output = self.root_merge_sentence(semgcn_output, merged_input_masks, merged_dialog_length,
                                                     thread_lengths)
        else:  # w/o thread
            semgcn_output = self.semgcn(sequence_outputs, input_masks, semantic_adj)
            semgcn_output = self.merge_sentence(semgcn_output, input_masks, dialogue_length)
        semgcn_output = self.dropout(semgcn_output)

        sequence_outputs_ent = self.layernorm(
            sentence_sequence_outputs + syngcn_outputs + semgcn_output + ent_sequence_outputs)
        sequence_outputs_rel = self.layernorm(
            sentence_sequence_outputs + syngcn_outputs + semgcn_output + rel_sequence_outputs)
        sequence_outputs_pol = self.layernorm(
            sentence_sequence_outputs + syngcn_outputs + semgcn_output + pol_sequence_outputs)

        # Dynamic Selective Fusion
        global_masks, utterance_level_reply_adj, utterance_level_speaker_adj, utterance_level_mask, speaker_ids = [
            kwargs[w] for w in
            ['global_masks', 'utterance_level_reply_adj', 'utterance_level_speaker_adj', 'utterance_level_mask',
             'speaker_ids']]
        global_outputs = self.dynamic_selective_fusion(speaker_ids, sentence_sequence_outputs, global_masks,
                                                       utterance_level_reply_adj, utterance_level_speaker_adj,
                                                       utterance_level_mask)

        # Task-Specific Attention Fusion
        thread_masks = thread_masks.bool().unsqueeze(1)

        fusion_knowledge_ent = self.TSAF['ent'](sequence_outputs_ent, global_outputs, thread_masks,
                                                sentence_length=sentence_length, )
        fusion_knowledge_rel = self.TSAF['rel'](sequence_outputs_rel, global_outputs, thread_masks,
                                                sentence_length=sentence_length, )
        fusion_knowledge_pol = self.TSAF['pol'](sequence_outputs_pol, global_outputs, thread_masks,
                                                sentence_length=sentence_length, )

        fusion_knowledge_ent = self.dense_all['ent'](fusion_knowledge_ent)
        fusion_knowledge_rel = self.dense_all['rel'](fusion_knowledge_rel)
        fusion_knowledge_pol = self.dense_all['pol'](fusion_knowledge_pol)




        ent_loss, ent_logit = self.classify_matrix(kwargs, fusion_knowledge_ent, ent_matrix, sentence_masks,
                                                       'ent', mode)
        rel_loss, rel_logit = self.classify_matrix(kwargs, fusion_knowledge_rel, rel_matrix, full_masks,
                                                       'rel', mode)
        pol_loss, pol_logit = self.classify_matrix(kwargs, fusion_knowledge_pol, pol_matrix, full_masks,
                                                       'pol', mode)

        total_loss = ent_loss + rel_loss + pol_loss

        return total_loss, [ent_loss, rel_loss, pol_loss], (ent_logit, rel_logit, pol_logit)