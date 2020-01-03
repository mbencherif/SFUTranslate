import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext import data
from configuration import cfg, device


class BeamSearchNode:
    def __init__(self, _id, decoder_lstm_context, next_token, c_t, eos_predicted, coverage_vector, result,
                 max_attention_indices, cumulative_loss, loss_size, tokens, lm_score):
        self.decoder_lstm_context = decoder_lstm_context
        self.next_token = next_token
        self.c_t = c_t
        self.eos_predicted = eos_predicted
        self.coverage_vector = coverage_vector
        self.result = result
        self.max_attention_indices = max_attention_indices
        self.cumulative_loss = cumulative_loss
        self.loss_size = loss_size
        self.tokens = tokens
        self.lm_score = lm_score
        self.result_output = None
        self.result_query = None
        self.result_context = None
        self._id = _id

    @property
    def id(self):
        return self._id

    def set_result(self, output, query, decoder_lstm_context):
        self.result_output = output
        self.result_query = query
        self.result_context = decoder_lstm_context


class STS(nn.Module):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        super(STS, self).__init__()
        self.SRC = SRC
        self.TGT = TGT
        self.criterion = nn.CrossEntropyLoss(ignore_index=TGT.vocab.stoi[cfg.pad_token], reduction='sum')
        self.bahdanau_attention = bool(cfg.bahdanau_attention)
        print("Creating the Seq2Seq Model with {} attention".format("Bahdanau" if self.bahdanau_attention else "Loung"))
        self.coverage = bool(cfg.coverage_required)
        print("Coverage model (See et al. definition [P17-1099]) is {}".format(
            "also considered" if self.coverage else "not considered"))
        self.encoder_emb = nn.Embedding(len(SRC.vocab), int(cfg.encoder_emb_size),
                                        padding_idx=SRC.vocab.stoi[cfg.pad_token])
        self.decoder_emb = nn.Embedding(len(TGT.vocab), int(cfg.decoder_emb_size),
                                        padding_idx=TGT.vocab.stoi[cfg.pad_token])
        self.encoder_layers = int(cfg.encoder_layers)
        self.encoder_bidirectional = True
        self.encoder_hidden = int(cfg.encoder_hidden_size)
        self.encoder = nn.LSTM(int(cfg.encoder_emb_size), self.encoder_hidden, self.encoder_layers,
                               bidirectional=self.encoder_bidirectional,
                               dropout=float(cfg.encoder_dropout_rate) if self.encoder_layers > 1 else 0.0)
        self.decoder_hidden = int(cfg.decoder_hidden_size)
        self.attention_W = nn.Linear(self.encoder_hidden * (2 if self.encoder_bidirectional else 1),
                                     self.decoder_hidden, bias=False)
        if self.bahdanau_attention:
            self.attention_U = nn.Linear(self.decoder_hidden, self.decoder_hidden, bias=True)
            self.attention_V = nn.Linear(self.decoder_hidden, 1, bias=False)
        else:
            self.attention_U = None
            self.attention_V = None
        # self.attention_proj = nn.Linear(self.encoder_hidden * (2 if self.encoder_bidirectional else 1)
        #                                 + self.decoder_hidden, self.decoder_hidden, bias=False)
        if self.encoder_hidden * (2 if self.encoder_bidirectional else 1) != self.decoder_hidden:
            self.enc_dec_hidden_bridge = nn.Linear(self.encoder_hidden * (2 if self.encoder_bidirectional else 1),
                                                   self.decoder_hidden, bias=False)
        else:
            self.enc_dec_hidden_bridge = None
        self.decoder_input_size = int(cfg.decoder_emb_size) + self.encoder_hidden * (
            2 if self.encoder_bidirectional else 1)
        self.decoder_layers = int(cfg.decoder_layers)
        self.decoder = nn.LSTM(self.decoder_input_size, self.decoder_hidden, self.decoder_layers,
                               dropout=float(cfg.decoder_dropout_rate) if self.decoder_layers > 1 else 0.0)
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.emb_dropout = nn.Dropout(p=float(cfg.emb_dropout))
        self.out_dropout = nn.Dropout(p=float(cfg.out_dropout))
        self.out = nn.Linear(self.encoder_hidden * (2 if self.encoder_bidirectional else 1)
                             + self.decoder_hidden + int(cfg.decoder_emb_size), len(TGT.vocab))
        if self.coverage:
            if not self.bahdanau_attention:
                raise ValueError("Coverage model is just integrated with Bahdanau Attention")
            # self.u_phi_j = nn.Linear(self.encoder_hidden * (2 if self.encoder_bidirectional else 1), 1, bias=False)
            # self.phi_j_N = int(cfg.coverage_phi_n)
            self.coverage_lambda = float(cfg.coverage_lambda)
            self.attention_C = nn.Linear(1, self.decoder_hidden, bias=False)
            self.coverage_dropout = nn.Dropout(p=float(cfg.coverage_dropout))
        else:
            # self.u_phi_j = None
            # self.phi_j_N = 1
            self.coverage_lambda = 0.0
            self.attention_C = None
            self.coverage_dropout = None
        self.beam_search_decoding = False
        self.beam_size = int(cfg.beam_size)
        self.beam_search_length_norm_factor = float(cfg.beam_search_length_norm_factor)
        self.beam_search_coverage_penalty_factor = float(cfg.beam_search_coverage_penalty_factor)

    def forward(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        """
        if self.beam_search_decoding:
            return self.beam_search_decode(input_tensor_with_lengths, beam_size=self.beam_size)
        else:
            return self.greedy_decode(input_tensor_with_lengths, output_tensor_with_length, test_mode)

    def greedy_decode(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        """
        input_tensor, input_lengths = input_tensor_with_lengths
        if output_tensor_with_length is not None:
            output_tensor, outputs_lengths = output_tensor_with_length
            tokens_count = float(outputs_lengths.sum().item())
        else:
            output_tensor, outputs_lengths = None, None
            tokens_count = 0.0
        predicted_tokens_count = 0.0
        input_sequence_length, batch_size = input_tensor.size()
        embedded_input = self.emb_dropout(self.encoder_emb(input_tensor))  # seq_length * batch_size * emd_size
        packed_input = pack_padded_sequence(embedded_input, input_lengths, enforce_sorted=False)
        encoded_pack_output, encoder_lstm_context = self.encoder(packed_input)
        encoder_lstm_output, _ = pad_packed_sequence(encoded_pack_output)
        # encoder_lstm_output, encoder_lstm_context = self.encoder(
        # embedded_input, self.encoder_init(input_tensor.size(1)))
        decoder_lstm_context = self.reformat_encoder_hidden_states(encoder_lstm_context)
        target_length = min(int(cfg.maximum_decoding_length * 1.1), input_sequence_length * 2)
        next_token = output_tensor.select(0, 0) if output_tensor is not None and not test_mode else \
            torch.LongTensor().new_full((batch_size,), self.TGT.vocab.stoi[cfg.bos_token]).to(device)  # size = batch_size
        pad_token = torch.LongTensor().new_full((batch_size,), self.TGT.vocab.stoi[cfg.pad_token]).to(device)
        result = torch.zeros(target_length, batch_size, device=device)
        cumulative_loss = 0.0
        c_t = torch.zeros(batch_size, self.encoder_hidden * (2 if self.encoder_bidirectional else 1), device=device)
        eos_predicted = torch.zeros(batch_size, device=device).byte()
        if self.bahdanau_attention:
            preprocessed_attention_encoder_representations = self.attention_W(encoder_lstm_output).transpose(0, 1)
        else:
            preprocessed_attention_encoder_representations = self.attention_W(encoder_lstm_output).permute(1, 2, 0)
        attention_mask = input_tensor.transpose(0, 1).unsqueeze(1) != self.SRC.vocab.stoi[cfg.pad_token]
        loss_size = 0.0

        if self.coverage:
            coverage_vector = torch.zeros(batch_size, input_sequence_length, 1, device=device).float()
            # phi_j: batch_size * 1 * max_input_length
            # phi_j = self.phi_j_N * self.sigmoid(
            #    self.u_phi_j(self.coverage_dropout(encoder_lstm_output.transpose(0, 1)))).squeeze(-1).unsqueeze(1)
        max_attention_indices = torch.zeros(target_length, batch_size, device=device)
        for t in range(target_length):
            dec_emb = self.emb_dropout(self.decoder_emb(next_token))  # batch_size * decoder_emb_size
            decoder_input = torch.cat([dec_emb, c_t], dim=1).view(1, batch_size, self.decoder_input_size)
            _, decoder_lstm_context = self.decoder(decoder_input, decoder_lstm_context)
            query = decoder_lstm_context[0][-1].view(batch_size, self.decoder_hidden)
            semi_output = self.out_dropout(torch.cat([dec_emb, query, c_t], dim=1))
            o = self.out(self.tanh(semi_output)) \
                .view(batch_size, len(self.TGT.vocab))  # batch_size, target_vocab_size
            greedy_prediction = torch.argmax(o, dim=1).detach()
            eos_predicted = torch.max(eos_predicted, (greedy_prediction == self.TGT.vocab.stoi[cfg.eos_token]))
            if output_tensor is not None:
                # Input Feeding
                next_token = output_tensor.select(0, t + 1) if t < output_tensor.size(0) - 1 else pad_token
                cumulative_loss += self.criterion(o, next_token)
                loss_size += 1.0
            else:
                # greedy approach
                next_token = greedy_prediction
            predicted_tokens_count += batch_size - eos_predicted.sum().item()
            if sum(eos_predicted.int()) == batch_size:
                break
            # overwrite the Input Feeding criteria
            if test_mode:
                next_token = greedy_prediction
            result[t, :] = greedy_prediction
            # Calculate the new context vector
            # encoded representations size: innput_seq_length * batch_size * (2 * encoder_hidden representation size)
            # decoder representation size: 1 * batch_size * decoder representation size
            # inp_hidden = encoder_lstm_output.transpose(0, 1)
            # mapped_inputs = self.attention_W(inp_hidden)  # batch_size, input_len, dec hidden
            # alphas = torch.bmm(query.unsqueeze(1), mapped_inputs.transpose(1, 2))  # batch_size,1, input_len
            # alphas = self.softmax(alphas)
            # c_t = torch.bmm(alphas, inp_hidden).squeeze(1)
            if self.bahdanau_attention:
                attention_inputs = self.attention_U(query.unsqueeze(1).repeat(1, input_sequence_length, 1)) + \
                                   preprocessed_attention_encoder_representations
                if self.coverage:
                    attention_inputs = attention_inputs + self.attention_C(coverage_vector)
                alphas = self.attention_V(self.tanh(attention_inputs)).squeeze(2).unsqueeze(1) # b_size,1,input_len
            else:  # Loung general
                alphas = query.unsqueeze(1) @ preprocessed_attention_encoder_representations  # b_size,1,input_len
            alphas = torch.where(attention_mask, alphas, alphas.new_full([1], float('-inf')))
            alphas = self.softmax(alphas)  # batch_size * 1 * max_input_length
            if self.coverage:
                # coverage_vector = coverage_vector + ((1.0 / (phi_j + 1e-32)) * alphas).squeeze(1).unsqueeze(-1)
                cvg_formatted_alphas = alphas.squeeze(1)
                coverage_vector = coverage_vector + cvg_formatted_alphas.unsqueeze(-1)
            # input_tensor => max_seq_length * batch_size
            max_attention_indices[t, :] = alphas.max(dim=-1)[1].view(-1).detach()  # batch_size
            c_t = (alphas @ encoder_lstm_output.transpose(0, 1)).squeeze(1)
            if self.coverage and output_tensor is not None:
                masked_coverage = coverage_vector.squeeze(2) * attention_mask.float().squeeze(1)
                min_coverage_and_attn = torch.min(masked_coverage, cvg_formatted_alphas)
                cumulative_loss = cumulative_loss + self.coverage_lambda * min_coverage_and_attn.sum()
        return result, max_attention_indices, cumulative_loss,  loss_size, tokens_count

    def beam_search_decode(self, input_tensor_with_lengths, beam_size=1):
        input_tensor, input_lengths = input_tensor_with_lengths
        input_sequence_length, batch_size = input_tensor.size()
        embedded_input = self.emb_dropout(self.encoder_emb(input_tensor))  # seq_length * batch_size * emd_size
        packed_input = pack_padded_sequence(embedded_input, input_lengths, enforce_sorted=False)
        encoded_pack_output, encoder_lstm_context = self.encoder(packed_input)
        encoder_lstm_output, _ = pad_packed_sequence(encoded_pack_output)
        decoder_lstm_context = self.reformat_encoder_hidden_states(encoder_lstm_context)
        attention_mask = input_tensor.transpose(0, 1).unsqueeze(1) != self.SRC.vocab.stoi[cfg.pad_token]
        tokens_count = 0.0
        target_length = min(int(cfg.maximum_decoding_length * 1.1), input_sequence_length * 2)
        if self.bahdanau_attention:
            preprocessed_attention_encoder_representations = self.attention_W(encoder_lstm_output).transpose(0, 1)
        else:
            preprocessed_attention_encoder_representations = self.attention_W(encoder_lstm_output).permute(1, 2, 0)
        next_token = torch.LongTensor().new_full((batch_size,), self.TGT.vocab.stoi[cfg.bos_token]).to(device)
        c_t = torch.zeros(batch_size, self.encoder_hidden * (2 if self.encoder_bidirectional else 1), device=device)
        eos_predicted = torch.zeros(batch_size, device=device).byte()
        coverage_vector = torch.zeros(batch_size, input_sequence_length, 1, device=device).float() \
            if self.coverage else None

        result = torch.zeros(target_length, batch_size, device=device)
        max_attention_indices = torch.zeros(target_length, batch_size, device=device)
        cumulative_loss = 0.0
        loss_size = 0.0
        last_created_node_id = 0
        tokens = torch.zeros((batch_size, 1), device=device).long()
        lm_score = torch.zeros((batch_size,), device=device)
        nodes = [BeamSearchNode(last_created_node_id, decoder_lstm_context, next_token, c_t, eos_predicted,
                                coverage_vector, result, max_attention_indices, cumulative_loss, loss_size, tokens, lm_score)]
        final_results = []
        m_softmax = nn.Softmax(dim=-1)
        for step in range(target_length):
            k = beam_size - len(final_results)
            if k < 1:
                break
            all_predictions = torch.zeros(batch_size, len(nodes) * k, device=device).long()
            all_lm_scores = torch.zeros(batch_size, len(nodes) * k, device=device).float()

            for n_id, node in enumerate(nodes):
                dec_emb = self.emb_dropout(self.decoder_emb(node.next_token))  # batch_size * decoder_emb_size
                decoder_input = torch.cat([dec_emb, node.c_t], dim=1).view(1, batch_size, self.decoder_input_size)
                _, decoder_lstm_context = self.decoder(decoder_input, node.decoder_lstm_context)
                query = decoder_lstm_context[0][-1].view(batch_size, self.decoder_hidden)
                semi_output = self.out_dropout(torch.cat([dec_emb, query, node.c_t], dim=1))
                o = self.out(self.tanh(semi_output)).view(batch_size, len(self.TGT.vocab))  # batch_size, target_vocab_size
                node.set_result(m_softmax(o), query, decoder_lstm_context)
                k_values, k_indices = torch.topk(node.result_output, dim=1, k=k)
                for beam_index in range(k):
                    overall_index = n_id * k + beam_index
                    all_predictions[:, overall_index] = k_indices[:, beam_index]
                    all_lm_scores[:, overall_index] = node.lm_score + torch.log(k_values[:, beam_index])
            k_values, k_indices = torch.topk(all_lm_scores, dim=1, k=k)
            temp_next_nodes = []
            for beam_index in range(k):
                node_ids = k_indices[:, beam_index] / k
                node_ids = list(node_ids.cpu().numpy())  # list of size batch_size
                pred_ids = list(k_indices[:, beam_index].cpu().numpy())
                greedy_prediction = torch.zeros((batch_size,), device=device).long()
                for b in range(batch_size):
                    greedy_prediction[b] = all_predictions[b, pred_ids[b]]
                lm_score = k_values[:, beam_index]
                last_created_node_id += 1
                query = torch.cat([nodes[n_id].result_query[b_id].unsqueeze(0) for b_id, n_id in enumerate(node_ids)], dim=0)
                coverage_vector = torch.cat(
                    [nodes[n_id].coverage_vector[b_id].unsqueeze(0) for b_id, n_id in enumerate(node_ids)], dim=0)
                max_attention_indices = torch.cat([nodes[n_id].max_attention_indices[:, b_id].unsqueeze(1) for b_id, n_id in enumerate(node_ids)], dim=1)
                if self.bahdanau_attention:
                    attention_inputs = self.attention_U(query.unsqueeze(1).repeat(1, input_sequence_length, 1)) + \
                                       preprocessed_attention_encoder_representations
                    if self.coverage:
                        attention_inputs = attention_inputs + self.attention_C(coverage_vector)
                    alphas = self.attention_V(self.tanh(attention_inputs)).squeeze(2).unsqueeze(1) # b_size,1,input_len
                else:  # Loung general
                    alphas = query.unsqueeze(1) @ preprocessed_attention_encoder_representations  # b_size,1,input_len
                # BeamSearchNode.make_new_node(last_created_node_id, nodes, node_ids, lm_score,
                #                                        greedy_prediction, self.TGT.vocab.stoi[cfg.eos_token], step))
                alphas = torch.where(attention_mask, alphas, alphas.new_full([1], float('-inf')))
                alphas = self.softmax(alphas)  # batch_size * 1 * max_input_length
                if self.coverage:
                    # coverage_vector = coverage_vector + ((1.0 / (phi_j + 1e-32)) * alphas).squeeze(1).unsqueeze(-1)
                    cvg_formatted_alphas = alphas.squeeze(1)
                    coverage_vector = coverage_vector + cvg_formatted_alphas.unsqueeze(-1)
                # input_tensor => max_seq_length * batch_size
                max_attention_indices[step, :] = alphas.max(dim=-1)[1].view(-1).detach()  # batch_size
                c_t = (alphas @ encoder_lstm_output.transpose(0, 1)).squeeze(1)
                decoder_lstm_context = (torch.cat([nodes[n_id].result_context[0][:, b_id, :]
                                                   for b_id, n_id in enumerate(node_ids)], dim=0).unsqueeze(0),
                                        torch.cat([nodes[n_id].result_context[1][:, b_id, :]
                                                   for b_id, n_id in enumerate(node_ids)], dim=0).unsqueeze(0))
                eos_p = torch.cat([nodes[n_id].eos_predicted[b_id].unsqueeze(0) for b_id, n_id in enumerate(node_ids)], dim=0)
                eos_predicted = torch.max(eos_p, (greedy_prediction == self.TGT.vocab.stoi[cfg.eos_token]))
                prev_tokens = torch.cat([nodes[n_id].tokens[b_id].unsqueeze(0) for b_id, n_id in enumerate(node_ids)], dim=0)
                new_tokens = torch.cat((prev_tokens, greedy_prediction.unsqueeze(-1)), dim=1)
                result = torch.cat([nodes[n_id].result[:, b_id].unsqueeze(1) for b_id, n_id in enumerate(node_ids)], dim=1)
                result[step, :] = greedy_prediction
                c_beam = BeamSearchNode(last_created_node_id, decoder_lstm_context, greedy_prediction,
                                        c_t, eos_predicted, coverage_vector, result, max_attention_indices, 0.0, 1.0,
                                        new_tokens, lm_score)

                if sum(eos_predicted.int()) == batch_size:
                    final_results.append(c_beam)
                else:
                    temp_next_nodes.append(c_beam)
            del nodes[:]
            nodes = temp_next_nodes
        if not len(final_results):
            for node in nodes:
                final_results.append(node)
        result = torch.zeros(target_length, batch_size, device=device)
        max_attention_indices = torch.zeros(target_length, batch_size, device=device)
        lp = lambda l: ((5 + l) ** self.beam_search_length_norm_factor) / (5 + 1) ** self.beam_search_length_norm_factor
        for b_ind in range(batch_size):
            best_score = float('-inf')
            best_tokens = None
            best_att = None
            for node in final_results:
                tokens = node.tokens[b_ind]
                eos_ind = (tokens == self.TGT.vocab.stoi[cfg.eos_token]).nonzero().view(-1)
                if eos_ind.size(0):
                    tsize = eos_ind[0].item()
                else:
                    tsize = tokens.size(0)
                # based on Google's NMT system paper [https://arxiv.org/pdf/1609.08144.pdf]
                cp = sum([math.log(min(x, 1.0)) for x in
                          list(node.coverage_vector[b_ind].view(-1).cpu().numpy()) if x > 0.0])
                lms = node.lm_score[b_ind].item() / lp(tsize) + self.beam_search_coverage_penalty_factor * cp
                if lms > best_score:
                    best_score = lms
                    best_tokens = tokens
                    best_att = node.max_attention_indices[:, b_ind]
            result[:best_tokens[1:].size(0), b_ind] = best_tokens[1:]
            max_attention_indices[:, b_ind] = best_att
        return result, max_attention_indices, torch.zeros(1, device=device),  1.0, tokens_count

    def reformat_encoder_hidden_states(self, encoder_hidden_prams):
        """
        :param encoder_hidden_prams: Pair of size 2 of 3-D Tensors
        [num_encoder_directions*num_encoder_layers, batch_size, hidden_size]
        """
        hidden = encoder_hidden_prams[0]
        context = encoder_hidden_prams[1]
        if self.encoder_bidirectional:
            hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
            context = torch.cat([context[0:context.size(0):2], context[1:context.size(0):2]], 2)
        if self.decoder_layers < hidden.size(0):
            hidden = hidden[hidden.size(0)-self.decoder_layers:]
            context = context[context.size(0)-self.decoder_layers:]
        if self.enc_dec_hidden_bridge is not None:
            hidden = self.tanh(self.enc_dec_hidden_bridge(hidden))
            context = self.tanh(self.enc_dec_hidden_bridge(context))
        # context override for debugging the effect of ctx in decoder initialization
        # context = torch.zeros_like(context,  device=device, dtype=torch.float32)
        return hidden, context

    def encoder_init(self, batch_size):
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(self.encoder_layers * (2 if self.encoder_bidirectional else 1),
                           batch_size, self.encoder_hidden, device=device, dtype=torch.float32), \
               torch.zeros(self.encoder_layers * (2 if self.encoder_bidirectional else 1),
                           batch_size, self.encoder_hidden, device=device, dtype=torch.float32)

    def decoder_init(self, batch_size):
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(self.decoder_layers, batch_size, self.decoder_hidden, device=device, dtype=torch.float32), \
               torch.zeros(self.decoder_layers, batch_size, self.decoder_hidden, device=device, dtype=torch.float32)