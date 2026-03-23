import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List


def pad_nested_list_left(nested_list):
    # Find the length of the longest sublist
    max_length = max(len(sublist) for sublist in nested_list)

    # Pad each sublist with 1s at the start (left padding only)
    padded_list = [
        [1] * (max_length - len(sublist)) + sublist for sublist in nested_list
    ]

    return padded_list, max_length

### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        # logits = torch.ones_like(logits).to(logits.device)
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits

def sample(
    logits,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    sample_logits=True,
):
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs

def cfg_logit_process(combined_logits, cfg_scale=4.0):
    # Classifier-Free Guidance
    cond_logits, uncond_logits = torch.split(
        combined_logits, len(combined_logits) // 2, dim=0
    )
    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    return logits

def calculate_tvd(tensor1, tensor2):
    # Total Variation Distance (TVD) between two tensors
    tvd = 0.5 * torch.abs(tensor1 - tensor2)
    return tvd

def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def evaluate_posterior_v2(
    logits, # shape like: [43, 5, 16384]
    candidates,
    idx,
    logits_processor=None,
    token_shift_lab=False,
    token_shift_interval=2,
    lantern=False,
    lantern_k=1000,
    lantern_delta=0.1,
    nearest_latents=None,
):
    if logits_processor is not None:  # for temperature not zero, which needs sampling
        accept_length = 1
        accept_cand = candidates[0][:1]  # candidates shape like: [43, 5], root, so tokens in the first column are the same
        best_candidate = 0
        
        """For the token tree path matrix variable 'candidates':
        - i means the column, corresponding to the token place, also the level of tree
        - j means the row, corresponding to the token path, each path lead to a leaf node"""

        # for-loop over each token place, but first token is generated by target model last round, so start from 1
        for i in range(1, candidates.shape[1]):  
            # candidates.shape like (43, 5), not constant because of dynamic tree
            if i != accept_length:  # this means in this token place, no token is accepted, then go to resampling
                break

            adjustflag = False
            # used to check wether all nodes in current token path (row) are equal to the accepted accept_cand
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            # get the row index of the first eq token in current token place
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]

            gt_logits = logits[fi, i - 1][None]  # [fi i-1] means [row, col] to get target model logits of specific token
            gt_logits = logits_processor(None, gt_logits)[0]
            # Q: why each gtp can be used alongside one whole col ?
            # A: this is in each token place, we actually only veirfy subtokens for one parent token
            # this align with the strategy: once the highest prob token is accepted in one token place
            # later tokens wont be verified again, so each token place only get at most one token accepeted, then verify next token place
            gtp = torch.softmax(gt_logits, dim=0)

            candidates_set = []  # store both the accepted tokens and the rejected tokens which need resampling
            
            # for-loop over candidates on each token place
            for j in range(candidates.shape[0]):
                # if this token is in a column that all previous tokens in path are equal to accepted candidates
                if is_eq[j]:  # greeedy verify, so only verify based on token accepted on last token place 
                    x = candidates[j, i]
                    xi = x.item()

                    if xi in candidates_set or xi == -1:  # if already exists token need resampling, skip this toke place
                        continue

                    candidates_set.append(xi)

                    r = random.random()

                    px = gtp[xi]
                    if lantern:
                        nearest_probs = gtp[
                            nearest_latents[xi, :lantern_k]
                        ].reshape(lantern_k, 1)
                        cumsum_nearest_probs = torch.cumsum(nearest_probs, dim=0)

                        if lantern_delta > 1.0:  # for lantern ++
                            indices = (
                                cumsum_nearest_probs <= (lantern_delta - 1) * px
                            ).nonzero(as_tuple=True)[0]
                        else:  # for lantern
                            indices = (
                                cumsum_nearest_probs <= lantern_delta
                            ).nonzero(as_tuple=True)[0]
                        if indices.numel() == 0:
                            indices = -1
                        else:
                            indices = indices[-1]
                        if indices == -1:
                            px = px
                        else:
                            px = px + cumsum_nearest_probs[indices]

                    qx = 1.0
                    acp = px / qx 

                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)  # add accpet token to accept path
                        accept_length += 1 # if accept this candidate token, then this token place does not need to be verified anymore
                        best_candidate = j
                        break # later tokens in this token place wont be verified again
                    else: 
                        # every time the newest token in this token place is rejected, we need to refresh the distribution of target model
                        # although only the last rejection would realy use this adjusted distribution
                        gtp[xi] = 0

                        if lantern:
                            if indices != -1:
                                gtp[nearest_latents[xi, : lantern_k + 1]] = 0

                        if gtp.sum() == 0:
                            gtp = torch.ones_like(gtp)

                        gtp /= gtp.sum()
                        adjustflag = True  # adjust distribution of target model


        if adjustflag and accept_length != candidates.shape[1]:  # if reject and resamling
            sample_p = gtp
        else: 
            gt_logits = logits[best_candidate, accept_length - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            sample_p = torch.softmax(gt_logits, dim=0)

        if token_shift_lab:
            old_accept_length = accept_length
            if token_shift_interval == 0:
                token_drift = False
            else:
                token_drift = (token_shift_interval == 1) or (idx % token_shift_interval == 1)
            if token_drift:
                best_candidate = 0
                if (candidates[best_candidate] == -1).nonzero().numel() == 0:
                    last_acp_token_index = candidates.shape[1] - 1 # no -1 found, use the last token
                else:
                    last_acp_token_index = (candidates[best_candidate] == -1).nonzero()[0] - 1
                accept_length = last_acp_token_index + 1
                accept_length = min(accept_length, old_accept_length)

        return (
            torch.tensor(best_candidate), # best candidate is actually best token path (row number)
            accept_length - 1, # minus to get the last one accepted token index
            sample_p,  # return the last target logits (for resampling or new logits)
        )

    else:  # for temperature zero, which needs greedy decoding, select the max ones
        device = logits.device
        batch_size, seq_len, vocab_size = logits.size()
        candidates_verify = candidates[:, 1:]  # Shape: (batch_size, seq_len)

        # Compute softmax probabilities over logits
        gtp = torch.softmax(
            logits, dim=-1
        )  # Shape: (batch_size, seq_len, vocab_size)

        # Get the token indices from candidates
        xi = candidates_verify  # Shape: (batch_size, seq_len)

        # Mask for positions where xi == -1
        valid_mask = (xi != -1).to(device)  # Shape: (batch_size, seq_len)

        # Adjust xi to have valid indices for indexing operations
        xi_valid = xi.clone()
        xi_valid[~valid_mask] = (
            0  # Replace invalid indices with 0 (or any valid index)
        )

        # Gather probabilities of xi
        px = gtp.gather(dim=-1, index=xi_valid.unsqueeze(-1)).squeeze(
            -1
        )  # Shape: (batch_size, seq_len)
        px = px * valid_mask
        if isinstance(nearest_latents, np.ndarray):
            nearest_latents = torch.from_numpy(nearest_latents).to(device)
        if not lantern:
            # Greedy decoding
            top_tokens = torch.argmax(
                logits[:, :-1], dim=-1
            )  # Shape: (batch_size, seq_len)
            posterior_mask = (xi == top_tokens).int() * valid_mask
            candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(
                dim=1
            )
            accept_length = candidates_accept_length.max()
        else:
            # Adaptive decoding with nearest latent tokens
            search_space = lantern_k
            nearest_indices = nearest_latents[
                xi_valid
            ]  # Shape: (batch_size, seq_len, k)
            nearest_indices = nearest_indices[
                :, :, :search_space
            ]  # Limit search space

            # For invalid positions, set nearest_indices to zero
            nearest_indices[
                ~valid_mask.unsqueeze(-1).expand_as(nearest_indices)
            ] = 0

            # Get probabilities of nearest latent tokens
            nearest_probs = gtp.gather(
                dim=-1, index=nearest_indices
            )  # Shape: (batch_size, seq_len, search_space)
            nearest_probs = nearest_probs * valid_mask.unsqueeze(
                -1
            )  # Zero out invalid positions

            # Compute cumulative sum of nearest probabilities
            cumsum_nearest_probs = torch.cumsum(
                nearest_probs, dim=-1
            )  # Shape: (batch_size, seq_len, search_space)

            # Prepare target and approximate distributions
            px_expanded = px.unsqueeze(-1).repeat(
                1, 1, search_space
            )  # Shape: (batch_size, seq_len, search_space)
            approx_p = (
                px_expanded + cumsum_nearest_probs
            )  # Shape: (batch_size, seq_len, search_space)
            approx_p = approx_p * valid_mask.unsqueeze(
                -1
            )  # Zero out invalid positions

            # Concatenate distributions for TVD
            target_p = torch.cat(
                [px_expanded, nearest_probs], dim=-1
            )  # Shape: (batch_size, seq_len, 2 * search_space)
            approx_p_full = torch.cat(
                [approx_p, torch.zeros_like(nearest_probs)], dim=-1
            )

            # Zero out invalid positions in target and approximate distributions
            target_p = target_p * valid_mask.unsqueeze(-1).to(torch.float32)
            approx_p_full = approx_p_full * valid_mask.unsqueeze(-1).to(
                torch.float32
            )

            # Compute TVD
            tvd = calculate_tvd(target_p, approx_p_full)

            tvd = torch.nan_to_num(tvd, nan=0.0)
            tvd_px = tvd[:, :, :search_space]
            tvd_cumsum = torch.cumsum(tvd[:, :, search_space:], dim=-1)
            tvd = tvd_px + tvd_cumsum
            # For invalid positions, set tvd to a high value to avoid selecting them
            tvd[~valid_mask] = float("inf")

            # Determine indices where TVD exceeds threshold
            # Create a boolean mask where tvd does not exceed coeff_a
            if lantern_delta > 1.0:
                tvd_not_exceeds = tvd <= (lantern_delta - 1) * px.unsqueeze(-1)
            else:
                tvd_not_exceeds = tvd <= lantern_delta

            # Get the size of the last dimension
            dim_size = tvd.shape[-1]

            # Create indices for the last dimension
            indices = (
                torch.arange(dim_size).unsqueeze(0).unsqueeze(0).to(tvd.device)
            )
            indices = indices.expand(tvd.shape[0], tvd.shape[1], dim_size)

            # Use the mask to select valid indices, set invalid positions to -1
            masked_indices = torch.where(
                tvd_not_exceeds, indices, torch.full_like(indices, -1)
            )

            # Find the maximum valid index for each (batch_size, seq_len)
            indices = masked_indices.max(dim=-1)[0]

            # Update probabilities based on indices
            idx_mask = indices >= 0
            idx_values = indices * idx_mask
            idx_values = idx_values.unsqueeze(-1)

            # Handle positions where idx_values == -1
            px_adjusted = torch.where(
                idx_mask, approx_p.gather(dim=-1, index=idx_values).squeeze(-1), px
            )
            px_adjusted = px_adjusted * valid_mask  # Zero out invalid positions

            # Update gtp with adjusted probabilities
            gtp.scatter_(
                dim=-1, index=xi_valid.unsqueeze(-1), src=px_adjusted.unsqueeze(-1)
            )

            # Compute posterior mask
            top_tokens = torch.argmax(gtp, dim=-1)[:, :-1]  # Adjusted to match xi
            posterior_mask = (xi == top_tokens).int() * valid_mask
            candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(
                dim=1
            )
            accept_length = candidates_accept_length.max()

        # Choose the best candidate
        if accept_length == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

        return best_candidate, accept_length, logits[best_candidate, accept_length]


@torch.no_grad()
def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    tree_position_ids,
    input_ids,
    retrieve_indices,
    cfg_scale,
    attention_mask=None,
):
    position_ids = tree_position_ids + input_ids.shape[1]
    if attention_mask is not None:
        remaining_length = (
            input_ids.shape[1] + tree_candidates.shape[1] - attention_mask.shape[1]
        )
        one_padding = torch.ones(
            (attention_mask.shape[0], remaining_length),
            dtype=torch.long,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask, one_padding], dim=1) # 

    outputs, tree_logits, hidden_state = model(
        input_ids=tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    tree_logits = cfg_logit_process(tree_logits, cfg_scale) 
    logits = tree_logits[0, retrieve_indices] 
    return logits, hidden_state, outputs


@torch.no_grad()
def update_inference_inputs(
    model,
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    logits_processor,
    new_token,
    past_key_values_data_list,
    current_length_data,
    hidden_state_new,
    sample_p,
    cfg_scale,
    static_tree=False,
    cur_skip=False,
    last_skip=False,
):
    prev_input_len = input_ids.shape[1]

    # return the indices of accepted tokens in full sequence (120 + ... max 2048)
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )

    # retrieve_indices.shape like: [40, 5]
    # candidates shape like [43, 5]
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

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[
        :, best_candidate, : accept_length + 1
    ]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    ea_input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1).repeat(
        2, 1
    )
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = (
        model.ea_layer.topK_genrate_v2(
            accept_hidden_state_new,
            input_ids=ea_input_ids,
            head=model.base_model.lm_head,
            logits_processor=logits_processor,
            cfg_scale=cfg_scale,
        )
    )
    new_token += accept_length + 1
    return (
        input_ids,
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        new_token,
        None,
        token,
    )


def reset_tree_mode(model):
    model.base_model.model.tree_mode = True
    model.base_model.model.tree_mask = None


def generate_candidates(
    tree_logits,
    tree_indices,
    retrieve_indices,
    sample_token,
    logits_processor,
):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits[0]

    candidates = torch.cat(
        [candidates_logit, candidates_tree_logits.view(-1)], dim=-1
    )

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [
            tree_candidates,
            torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1,
        ],
        dim=0,
    )

    cart_candidates = tree_candidates_ext[retrieve_indices]

    if logits_processor is not None:
        candidates_tree_prob = tree_logits[1]
        candidates_prob = torch.cat(
            [
                torch.ones(
                    1, device=candidates_tree_prob.device, dtype=torch.float32
                ),
                candidates_tree_prob.view(-1),
            ],
            dim=-1,
        )

        tree_candidates_prob = candidates_prob[tree_indices]
        tree_candidates_prob_ext = torch.cat(
            [
                tree_candidates_prob,
                torch.ones(
                    (1), dtype=torch.float32, device=tree_candidates_prob.device
                ),
            ],
            dim=0,
        )
        cart_candidates_prob = tree_candidates_prob_ext[retrieve_indices]
    else:
        cart_candidates_prob = None
    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, cart_candidates_prob, tree_candidates
