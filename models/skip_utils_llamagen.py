import torch
import random

from torch.nn import functional as F
from .utils_llamagen import cfg_logit_process
from torch.nn.utils.rnn import pad_sequence

@torch.jit.script
def compute_similarity_pytorch_jit(
    sequences_tensors: list[torch.Tensor], 
    latents: torch.Tensor, 
    mode: str = "exp_decay", 
    decay: float = 0.9
):
    embed_sequences: list[torch.Tensor] = []
    for seq in sequences_tensors:
        if seq.shape[0] > 1:
            valid_tokens = seq[seq != -1]  # Ignore -1, -1 denotes no token
            embed_sequences.append(latents[valid_tokens[1:]])
    
    if len(embed_sequences) < 2:
        print("no valid sequences to compute similarity")
        return torch.tensor(0.0, device=latents.device, dtype=latents.dtype)

    N = len(embed_sequences)

    padded_embeds = pad_sequence(embed_sequences, batch_first=True, padding_value=0.0)
    normalized_embeds = F.normalize(padded_embeds, p=2.0, dim=2)

    sims_at_all_pos = torch.einsum('nld,mld->lmn', normalized_embeds, normalized_embeds)
    max_len = sims_at_all_pos.shape[0]

    mask = torch.zeros((N, max_len), dtype=torch.bool, device=latents.device)
    for i, seq in enumerate(embed_sequences):
        mask[i, :seq.shape[0]] = True
        
    mask_per_pos = mask.T # (max_len, N)
    valid_pos_mask = (mask_per_pos.unsqueeze(2) & mask_per_pos.unsqueeze(1)) # (max_len, N, N)
    masked_sims = sims_at_all_pos * valid_pos_mask.to(latents.dtype)

    if mode == "exp_decay":
        weights = torch.pow(decay, torch.arange(max_len, device=latents.device, dtype=latents.dtype))
        weights = weights / weights.sum()
    elif mode == "linear_decay":
        weights = torch.arange(max_len, 0, -1, device=latents.device, dtype=latents.dtype)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(max_len, device=latents.device, dtype=latents.dtype) / float(max_len)
        
    final_sim_matrix = torch.einsum('l,lnm->nm', weights, masked_sims)
    
    triu_indices = torch.triu_indices(N, N, offset=1, device=latents.device)
    if triu_indices.shape[1] == 0:
        return torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
        
    avg_sim = final_sim_matrix[triu_indices[0], triu_indices[1]].mean()
    
    return avg_sim

# @timer
def skip_verify(
    cur_step,
    steps_in_1st_row,
    skip_interval,
    new_token,
    draft_candidates,
    head_avg_sim,
    skipped_steps,
    last_step_skip: bool,
    policy="uniform",
    latents=None,
    sim_threshold=0.70,
):
    # To ensure we have enough cached features
    if new_token <= 5:
        return False
    if policy == "uniform":
        if skip_interval == 1024:  # 1024 means not skip
            return False
        cur_step -= steps_in_1st_row  # count step after first row
        return (cur_step % skip_interval == 1)
    elif policy == "dynamic":
        if last_step_skip:  
            return False
        down_sample_draft_candidates = draft_candidates[::2]
        sequences_list = torch.unbind(down_sample_draft_candidates, dim=0)
        avg_sim = compute_similarity_pytorch_jit(sequences_list, latents)
        if avg_sim < sim_threshold: 
            return False
    return True



def reuse_logits(cached_logits, logits_processor):
    gt_logits = cached_logits[0, -1][None]
    gt_logits = logits_processor(None, gt_logits)[0]
    sample_p = torch.softmax(gt_logits, dim=0)
    return sample_p


# ****************** Policy Function to Select Tokens for Skipping *********************

def calculate_row_similarity(row, compare_tokens, codebook):
    valid_tokens = row[row != -1]  # [valid_length]
    compare_length = len(valid_tokens)
    truncated_compare_tokens = compare_tokens[:compare_length]  # [compare_length]
    valid_tokens = valid_tokens.cpu()
    truncated_compare_tokens = truncated_compare_tokens.cpu()
    valid_embeddings = codebook[valid_tokens]  # [valid_length, embed_dim]
    compare_embeddings = codebook[
        truncated_compare_tokens
    ]  # [compare_length, embed_dim]

    similarity = F.cosine_similarity(
        valid_embeddings, compare_embeddings, dim=1
    )  # [valid_length]
    return similarity.mean().item()

def skip_evaluate(
    logits,
    candidates,
    logits_processor=None,
    policy="random",
):
    # may we should only remain the tokens in avg length
    count_valid_tokens = torch.sum(candidates != -1).item()
    avg_acp_len = count_valid_tokens // candidates.shape[0]

    if policy == "random":
        if logits_processor is not None:
            best_candidate = random.randint(0, candidates.shape[0] - 1)
        # for greedy decoding
        else:
            best_candidate = random.randint(0, candidates.shape[0] - 1)

    if (candidates[best_candidate] == -1).nonzero().numel() == 0:
        last_acp_token_index = (
            candidates.shape[1] - 1
        )  # no -1 token found, use the last token
    else:
        last_acp_token_index = (candidates[best_candidate] == -1).nonzero()[0] - 1
    # accepted tokens not include root token, which is sampled seperately and always accepted
    accept_length = min(last_acp_token_index, avg_acp_len - 1)  # Truncation

    if logits_processor is not None:
        sample_p = reuse_logits(logits, logits_processor)
    else:
        gt_logits = logits[best_candidate, accept_length - 1]
        sample_p = torch.softmax(gt_logits, dim=0)

    return best_candidate, accept_length, sample_p


# ****************** Function of Tree Decoding for Verification Skipping *********************

def tree_decoding_skip_llamagen(
    model,
    tree_candidates,
    past_key_values,
    current_length_data,
    tree_position_ids,
    last_skip_tokens,
    input_ids,
    retrieve_indices,
    cfg_scale,
    attention_mask=None,
):
    chain_candidates = last_skip_tokens.repeat(2, 1)
    full_candidates = torch.cat([chain_candidates, tree_candidates], dim=1)

    tree_position_ids = tree_position_ids + input_ids.shape[1]
    chain_position_ids = (
        torch.arange(0, last_skip_tokens.shape[1], device=input_ids.device)
        + input_ids.shape[1]
        - last_skip_tokens.shape[1]
    )
    full_position_ids = torch.cat([chain_position_ids, tree_position_ids], dim=0)

    assert full_candidates.shape[1] == full_position_ids.shape[0]

    # input_ids already contains the chain candidates ℹ️

    if attention_mask is not None:
        remaining_length = (
            input_ids.shape[1] + tree_candidates.shape[1] - attention_mask.shape[1]
        )
        one_padding = torch.ones(
            (attention_mask.shape[0], remaining_length),
            dtype=torch.long,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask, one_padding], dim=1)

    # Note: tree_logits is model.lm_head(outputs[0]), hidden_states is outputs[0]
    assert current_length_data[0] == input_ids.shape[1] - last_skip_tokens.shape[1]
    outputs, logits, hidden_state = model(
        input_ids=full_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=full_position_ids,
        attention_mask=attention_mask,
    )

    # extract tree_logits
    tree_logits = logits[:, chain_candidates.shape[1] :, :]  # shape: [2, 59, 16384]
    assert tree_logits.shape[1] == 59
    tree_logits = cfg_logit_process(tree_logits, cfg_scale)  # shape: [2, 59, 16384]
    tree_logits = tree_logits[0, retrieve_indices]  # shape like: ([43, 5, 16384])
    return tree_logits, hidden_state, input_ids


# ****************** Functions of Reusing Features for Next Round Drafting *********************


def retrieve_features(width, cached_features, num_skip_token, policy="reuse_prev"):
    """
    Returns:
    A tensor of shape [2, num_skip_token, 1280] that contains the reused features.
    """
    shape = cached_features.shape
    shape_list = list(shape)
    shape_list[1] = num_skip_token
    return_shape = torch.Size(shape_list)

    if policy == "reuse_prev":
        # If skip_token_num is greater than the cached feature length,
        # we can repeat the last feature to fill the required length
        if num_skip_token <= cached_features.shape[1]:
            return cached_features[:, cached_features.shape[1] - num_skip_token :]
        else:
            non_repeated_features = cached_features[:, :num_skip_token, :]
            last_feature = cached_features[:, -1, :].unsqueeze(1)
            repeated_features = last_feature.repeat(
                1, num_skip_token - cached_features.shape[1], 1
            )
            retrieved_features = torch.cat(
                (non_repeated_features, repeated_features), dim=1
            )
    assert retrieved_features.shape == return_shape, (
        f"Expected shape {return_shape}, but got {retrieved_features.shape}"
    )
    return retrieved_features


@torch.no_grad()
def update_inference_inputs_skip(
    model,
    idx,
    input_ids,
    candidates,
    cand_hiddens,
    best_candidate,
    accept_length,
    retrieve_indices,
    logits_processor,
    new_token,
    past_key_values_data_list,
    current_length_data,
    hidden_state,
    cached_features,
    sample_p,
    cfg_scale,
    cur_skip=False,
    feat_shift_lab=False,
    feat_shift_staleness=-1,
    feat_cons_lab=False,
    feat_cons_staleness=[-1, -1],
    feat_policy="reuse_prev",
    width=32,
):
    accept_hidden_state_new = None
    skip_tokens = None
    token = None
    # assert accept_length >= 1

    if cur_skip:
        # no sampling new token version
        new_tokens = candidates[best_candidate, : accept_length + 1][
            None
        ]  # including a fake newly sampled token
        skip_tokens = candidates[best_candidate, :accept_length][None]
        skip_tokens_num = skip_tokens.shape[1]
        accept_hidden_state_new = retrieve_features(
            width, cached_features, skip_tokens_num, feat_policy
        )
        ea_input_ids = torch.cat([input_ids, new_tokens], dim=1).repeat(2, 1)
        input_ids = torch.cat([input_ids, skip_tokens], dim=1)

    else:
        prev_input_len = input_ids.shape[1]

        # return the indices of accepted tokens in full sequence (120 + ... max 2048)
        select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
        )

        # retrieve_indices.shape like: [43, 5], candidates shape like [43, 5]
        # input_ids shape like [2, 120] and after concatenation, it will be [2, 120 + accept_length + 1]
        input_ids = torch.cat(
            [input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1
        )
        # Update the past key values based on the selected tokens
        # Source tensor that contains relevant past information based on the selected candidate
        for past_key_values_data in past_key_values_data_list:
            tgt = past_key_values_data[
                ..., select_indices.to(past_key_values_data.device), :
            ]
            # Destination tensor where the relevant past information will be stored
            dst = past_key_values_data[
                ..., prev_input_len : prev_input_len + tgt.shape[-2], :
            ]
            # Copy relevant past information from the source to the destination
            dst.copy_(tgt, non_blocking=True)

        # Update the current length tensor (currently only support batch size is 1)
        current_length_data.fill_(prev_input_len + tgt.shape[-2])

        retrieve_hidden_state_new = hidden_state[:, retrieve_indices]
        # shape like [2, 3, 1280], accept_length + 1 = 3
        accept_hidden_state_new = retrieve_hidden_state_new[
            :, best_candidate, : accept_length + 1
        ]

        if feat_shift_lab and new_token >= width:
            num_tokens = candidates[best_candidate, : accept_length + 1][None].shape[1]
            old_features = cached_features[:, :-num_tokens, :]
            if feat_shift_staleness == -1:
                # -1 means using fresh features
                retrieve_hidden_state_new = hidden_state[:, retrieve_indices]
                # shape like [2, 3, 1280], accept_length + 1 = 3
                accept_hidden_state_new = retrieve_hidden_state_new[
                    :, best_candidate, : accept_length + 1
                ]
            elif feat_shift_staleness == 0:
                # 0 means using most recent old features without additional shift
                num_tokens = candidates[best_candidate, : accept_length + 1][
                    None
                ].shape[1]
                accept_hidden_state_new = retrieve_features(
                    width, old_features, num_tokens, feat_policy
                )
            else:
                # using shifted features with additional token positions
                num_tokens = candidates[best_candidate, : accept_length + 1][
                    None
                ].shape[1]
                accept_hidden_state_new = retrieve_features(
                    width,
                    old_features[:, :-feat_shift_staleness, :],
                    num_tokens,
                    feat_policy,
                )

        if feat_cons_lab and new_token >= width:
            # we only do the interleaving experiment with two staleness
            assert len(feat_cons_staleness) == 2
            num_tokens = candidates[best_candidate, : accept_length + 1][None].shape[1]
            old_features = cached_features[:, :-num_tokens, :]
            if idx % 2 == 0:
                if feat_cons_staleness[0] == -1:
                    # fresh features
                    retrieve_hidden_state_new = hidden_state[:, retrieve_indices]
                    # shape like [2, 3, 1280], accept_length + 1 = 3
                    accept_hidden_state_new = retrieve_hidden_state_new[
                        :, best_candidate, : accept_length + 1
                    ]
                elif feat_cons_staleness[0] == 0:
                    # using most recent old features without additional shift
                    accept_hidden_state_new = retrieve_features(
                        width, old_features, num_tokens, feat_policy
                    )
                else:
                    # using shifted features with additional token positions
                    accept_hidden_state_new = retrieve_features(
                        width,
                        old_features[:, : -feat_cons_staleness[0], :],
                        num_tokens,
                        feat_policy,
                    )
            else:
                if feat_cons_staleness[1] == -1:
                    # fresh features
                    retrieve_hidden_state_new = hidden_state[:, retrieve_indices]
                    # shape like [2, 3, 1280], accept_length + 1 = 3
                    accept_hidden_state_new = retrieve_hidden_state_new[
                        :, best_candidate, : accept_length + 1
                    ]
                elif feat_cons_staleness[1] == 0:
                    # using most recent old features without additional shift
                    accept_hidden_state_new = retrieve_features(
                        width, old_features, num_tokens, feat_policy
                    )
                else:
                    # using shifted features with additional token positions
                    accept_hidden_state_new = retrieve_features(
                        width,
                        old_features[:, : -feat_cons_staleness[1], :],
                        num_tokens,
                        feat_policy,
                    )

        # sample a new token here
        prob = sample_p
        if logits_processor is not None:
            token = torch.multinomial(prob, 1)
            token = token[None]
        else:
            token = torch.argmax(prob)
            token = token[None, None]
        ea_input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1).repeat(
            2, 1
        )

    # The content sent to ea_layer like:
    # token 1  | token 2  | token 3
    # feature0 | feature1 | feature2

    # draft_tokens shape like: [1, 59], accept_hidden_state_new shape like: [2, 3, 1280]
    draft_tokens, draft_hiddens, retrieve_indices, tree_mask, tree_position_ids = (
        model.ea_layer.topK_genrate_v2(
            accept_hidden_state_new,
            input_ids=ea_input_ids,
            head=model.base_model.lm_head,
            logits_processor=logits_processor,
            cfg_scale=cfg_scale,
        )
    )
    if cur_skip:
        new_token += accept_length
    else:
        new_token += accept_length + 1

    return (
        input_ids,
        draft_tokens,
        draft_hiddens,
        skip_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        new_token,
        accept_hidden_state_new,
        token,
    )
