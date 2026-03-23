import json
import time
import torch
import os
import torch.nn as nn
import numpy as np

from typing import List, Optional
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

from .drafters.kv_cache import initialize_past_key_values
from .drafters.cnets_llamagen import Model
from .configs.configs import EConfig
from .drafters.utils import prepare_logits_processor
from .kv_variants.modeling_llamagen_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .utils_llamagen import (
    cfg_logit_process,
    tree_decoding,
    evaluate_posterior_v2,
    reset_tree_mode,
)
from .skip_utils_llamagen import (
    skip_verify,
    skip_evaluate,
    tree_decoding_skip_llamagen,
    update_inference_inputs_skip,
)

TOPK = 10  # topk for sparse tree

class EaModel(nn.Module):
    def __init__(
        self,
        base_model,
        base_model_name_or_path,
        ea_model_path,
        total_token,  # for tree search
        depth,
        top_k,
        threshold,
        ea_layer_state_dict,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        # self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path,use_fast=False)
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except Exception as e:
            bias = True
            print(f"Exception {e}, setting bias as: {bias}")

        # Load eagle layer based on llama structure
        self.ea_layer = Model(
            config,
            bias=bias,
            total_tokens=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
        )

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

        # for lantern
        self.nearest_latents = np.load("ckpts/llamagen/top_16383_indices.npy")

        # for vvs
        self.code_emb = np.load("ckpts/llamagen/code_emb.npy")

        from models.base_models.llamagen.vq_model import VQ_16

        checkpoint_path = "ckpts/llamagen/vq_ds16_t2i.pt"
        vq_model = VQ_16(codebook_size=16384, codebook_embed_dim=8)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        vq_model.load_state_dict(checkpoint["model"])
        self.latents = vq_model.quantize.embedding.weight  # (16384, 8)

    @classmethod
    def from_pretrained(
        cls,
        Type="LLaMA",
        base_model_path=None,
        ea_model_path=None,
        total_token=59,
        depth=4,
        top_k=10,
        threshold=1.0,
        **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type == "LlamaForCausalLM":
            base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {Type}")

        # load config
        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            print(
                f"Config file not found at {configpath}, downloading from Hugging Face Hub..."
            )
            configpath = hf_hub_download(ea_model_path, "config.json")

        # load drafter model
        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(
                load_model_path, map_location=base_model.device
            )
        except Exception as e:
            from safetensors.torch import load_file

            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)

            print(
                f"Because exception: {e}, Loading state dict from {load_model_path} with safetensors"
            )

        model = cls(
            base_model,
            base_model_path,
            configpath,
            total_token,  # total token is the number of candidates for each token place
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
        )

        # token -1 means automatically selecting total tokens for tree search based on speed
        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(
                    0, model.config.vocab_size - 200, (1, length)
                ).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        _ = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])  # compute per token time
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
        self,
        cond_idx=None,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
    ):
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                cond_idx=cond_idx,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            # original logits without softmax
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def initialize_tree_v2( 
        self,
        cond_combined,
        past_key_values,
        logits_processor,
        cfg_scale,
        attention_mask=None,
    ):

        outputs, orig, hidden_states = self(
            cond_idx=cond_combined,
            past_key_values=past_key_values,
            output_orig=True,
            attention_mask=attention_mask,
        )
        logits = cfg_logit_process(orig[:, -1], cfg_scale)

        if logits_processor is not None:
            logits = logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            token = torch.multinomial(probabilities, 1)
        else:
            token = torch.argmax(logits)
            token = token[None, None]
        token = torch.cat([token, token], dim=0)
        zero_padding = torch.zeros(
            (token.shape[0], 120), dtype=torch.long, device=token.device
        )
        input_ids = torch.cat((zero_padding, token.to(cond_combined.device)), dim=1)
        (
            draft_tokens,
            draft_hiddens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
        ) = self.ea_layer.topK_genrate_v2(
            hidden_states,
            input_ids,
            self.base_model.lm_head,
            logits_processor,
            cfg_scale,
        )

        return (
            draft_tokens,
            draft_hiddens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            orig,
            hidden_states,
            token,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cfg: Optional[float] = None,
        lantern: Optional[bool] = None,
        lantern_k: Optional[int] = None,
        lantern_delta: Optional[float] = None,
        **model_kwargs,
    ):

        target_forward_num = 0
        if model_kwargs["skip"]:
            skip_interval = model_kwargs.get("skip_interval")
            skip_policy = model_kwargs.get("skip_policy")
            token_policy = model_kwargs.get("select_token")
            feature_policy = model_kwargs.get("reuse_feature")
            feat_shift_lab = model_kwargs.get("feat_shift_lab")
            feat_shift_staleness = model_kwargs.get("feat_shift_staleness")
            token_shift_lab = model_kwargs.get("token_shift_lab", False)
            token_shift_interval = model_kwargs.get("token_shift_interval", 2)
            feat_cons_lab = model_kwargs.get("feat_cons_lab")
            feat_cons_staleness = model_kwargs.get("feat_cons_staleness")
            sim_threshold = model_kwargs.get("sim_threshold", 0.70)

        prompt_idx = model_kwargs.get("idx")
        if prompt_idx == 0 and model_kwargs["skip"]:
            print(
                f"Skip policy: {skip_policy}, interval: {skip_interval}, "
                f"token policy: {token_policy}, feature policy: {feature_policy}"
            )
            if feat_shift_lab:
                if feat_shift_staleness == -1:
                    print("Staleness is -1, no shift feature lab is triggered...")
                else:
                    print("feat_shift_lab is triggered...")
                    print(f"Staleness: {feat_shift_staleness}")

            if token_shift_lab:
                skip_interval = 1024
                print(
                    f"token_shift_lab is triggered with interval {token_shift_interval}..."
                )

            if feat_cons_lab:
                print("feat_cons_lab is triggered...")
                print(f"Staleness: {feat_cons_staleness}")

        profile_data_dict = {}
        profile_data_dict["tree_token_lst"] = []
        profile_data_dict["acp_len_lst"] = []

        accept_length_list = []
        caption_embs, emb_masks = self.base_model.t5_model.get_text_embeddings(prompt)
        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []

        for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            new_caption_emb = torch.cat(
                [caption_emb[valid_num:], caption_emb[:valid_num]]
            )
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)

        c_indices = new_caption_embs * new_emb_masks[:, :, None]
        c_emb_masks = new_emb_masks
        st = time.time()
        if hasattr(self.base_model, "past_key_values"):
            past_key_values = self.base_model.past_key_values
            past_key_values_data = self.base_model.past_key_values_data
            current_length_data = self.base_model.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, 2)
            self.base_model.past_key_values = past_key_values
            self.base_model.past_key_values_data = past_key_values_data
            self.base_model.current_length_data = current_length_data
        if cfg is not None:
            cond_null = torch.zeros_like(
                c_indices, device=c_indices.device
            ) + self.base_model.model.cls_embedding.uncond_embedding.to(
                c_indices.device
            )
            cond_combined = torch.cat([c_indices, cond_null])
        else:
            cond_combined = c_indices
        T = cond_combined.shape[1]
        max_batch_size = c_indices.shape[0]
        if c_emb_masks is not None:
            assert c_emb_masks.shape[0] == max_batch_size
            assert c_emb_masks.shape[1] == T
            if cfg is not None:
                attention_mask = torch.cat([c_emb_masks, c_emb_masks])
            else:
                attention_mask = c_emb_masks
        cond_combined = cond_combined.to(self.base_model.dtype)
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(cond_combined.device)
        self.ea_layer.reset_kv()

        # only when temperature 0, using logits processor
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_k=top_k, top_p=top_p
            )
        else:
            logits_processor = None

        st = time.time()

        if hasattr(self.base_model, "past_key_values"):
            past_key_values = self.base_model.past_key_values
            past_key_values_data = self.base_model.past_key_values_data
            current_length_data = self.base_model.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, 2)
            self.base_model.past_key_values = past_key_values
            self.base_model.past_key_values_data = past_key_values_data
            self.base_model.current_length_data = current_length_data

        reset_tree_mode(self)

        (
            draft_tokens,
            draft_hiddens,
            retrieve_indices,   
            tree_mask,
            tree_position_ids,
            logits,
            hidden_state,
            sample_token,
        ) = self.initialize_tree_v2(
            cond_combined, past_key_values, logits_processor, cfg, attention_mask
        )

        max_steps = max_length
        input_ids = torch.zeros((max_batch_size, 120), dtype=torch.long).to(
            cond_combined.device
        )
        new_token = 0

        cached_features = None
        last_verify_logits = None
        last_hidden_state_new = None
        last_step_skip = False
        last_skip_tokens = None
        steps_in_first_row = 0
        skipped_steps = 0
        head_avg_sim = 1
        skipped_acp_len_lst = []
        verified_acp_len_lst = []

        for idx in range(max_steps):
            self.base_model.model.tree_mask = tree_mask
            tree_draft_tokens = torch.cat([draft_tokens, draft_tokens]).to(
                self.base_model.device
            )
            draft_tokens = torch.cat(
                (draft_tokens, padding), dim=1
            )
            candidates = draft_tokens[0, retrieve_indices]
            cand_hiddens = draft_hiddens[
                0, :, retrieve_indices, :
            ]

            cur_step_skip = skip_verify(
                prompt_idx,
                idx,
                steps_in_first_row,
                skip_interval,
                new_token,
                candidates,
                head_avg_sim,
                skipped_steps,
                last_step_skip=last_step_skip,
                policy=skip_policy,
                latents=self.latents,
                sim_threshold=sim_threshold,
            )

            # ************ Verify Process ***********

            if not cur_step_skip:
                if not last_step_skip:
                    logits, hidden_state_new, outputs = tree_decoding(
                        self,
                        tree_draft_tokens,
                        past_key_values,
                        tree_position_ids,
                        input_ids,
                        retrieve_indices,
                        cfg,
                        attention_mask,
                    )
                    last_hidden_state_new = hidden_state_new
                else:
                    logits, hidden_state_new, input_ids = tree_decoding_skip_llamagen(
                        self,
                        tree_draft_tokens,
                        past_key_values,
                        current_length_data,
                        tree_position_ids,
                        last_skip_tokens,
                        input_ids,
                        retrieve_indices,
                        cfg,
                        attention_mask,
                    )
                    tree_features = hidden_state_new[:, last_skip_tokens.shape[1] :, :]
                    last_hidden_state_new = tree_features
                last_verify_logits = logits  # shape like: [43, 5, 16384]
                target_forward_num += 1  # target model forward once

                # Verified by target model
                (
                    best_candidate,
                    accept_length,
                    sample_p,
                ) = evaluate_posterior_v2(
                    logits,
                    candidates,
                    idx,
                    logits_processor,
                    token_shift_lab=token_shift_lab,
                    token_shift_interval=token_shift_interval,
                    lantern=lantern,
                    lantern_k=lantern_k,
                    lantern_delta=lantern_delta,
                    nearest_latents=self.nearest_latents,
                )  # Verify function

                # ********** cached target model features ***********
                if not last_step_skip:
                    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
                    accept_hidden_state_new = retrieve_hidden_state_new[
                        :, best_candidate, : accept_length + 1
                    ]
                    if cached_features is None:
                        cached_features = accept_hidden_state_new
                    else:
                        cached_features = torch.cat(
                            (cached_features, accept_hidden_state_new), dim=1
                        )
                else:
                    chain_features = hidden_state_new[:, : last_skip_tokens.shape[1], :]
                    tree_features = hidden_state_new[:, last_skip_tokens.shape[1] :, :]
                    retrieve_hidden_state_new = tree_features[:, retrieve_indices]
                    accept_hidden_state_new = retrieve_hidden_state_new[
                        :, best_candidate, : accept_length + 1
                    ]
                    if cached_features is None:
                        cached_features = accept_hidden_state_new
                    else:
                        cached_features = torch.cat(
                            (cached_features, chain_features), dim=1
                        )
                        cached_features = torch.cat(
                            (cached_features, accept_hidden_state_new), dim=1
                        )

                last_step_skip = False
                if torch.is_tensor(accept_length):
                    verified_acp_len_lst.append(accept_length.item() + 1)
                    accept_length_list.append(accept_length.item() + 1)
                else:
                    verified_acp_len_lst.append(accept_length + 1)
                    accept_length_list.append(accept_length + 1)
            else:
                best_candidate, accept_length, sample_p = skip_evaluate(
                    last_verify_logits,
                    candidates,
                    logits_processor,
                    policy=token_policy,
                )
                last_step_skip = True
                skipped_steps += 1

                if torch.is_tensor(accept_length):
                    skipped_acp_len_lst.append(accept_length.item())
                    accept_length_list.append(accept_length.item())
                else:
                    skipped_acp_len_lst.append(accept_length)
                    accept_length_list.append(accept_length)

            # ************ Update and Draft for Next Step ***********
            (
                input_ids,
                draft_tokens,
                draft_hiddens,
                skip_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                new_token,
                hidden_state,
                sample_token,
            ) = update_inference_inputs_skip(
                self,
                idx,
                input_ids,
                candidates,
                cand_hiddens,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                last_hidden_state_new,
                cached_features,
                sample_p,
                cfg,
                cur_skip=cur_step_skip,
                feat_shift_lab=feat_shift_lab,
                feat_shift_staleness=feat_shift_staleness,
                feat_cons_lab=feat_cons_lab,
                feat_cons_staleness=feat_cons_staleness,
                feat_policy=feature_policy,
            )
            if skip_tokens is not None:
                last_skip_tokens = skip_tokens
            if new_token > max_length:
                break

        return (
            input_ids[:, 120 : 120 + max_length],
            sum(accept_length_list) / target_forward_num,
            time.time() - st,
            profile_data_dict,
            accept_length_list,
            skipped_acp_len_lst,
            verified_acp_len_lst,
            skipped_steps,
        )

    @torch.no_grad()
    def decode_ids(self, ids):
        return self.base_model.decode_ids(ids)

