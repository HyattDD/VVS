import argparse
import torch
import numpy as np
import random
import os
import json
import csv
from tqdm import tqdm
import re

import models.drafters.choices as choices
from torchvision.utils import save_image


USE_EXPERIMENTAL_FEATURES = os.getenv("USE_EXPERIMENTAL_FEATURES", "0") == "1"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(args):
    if args.model == "lumina_mgpt":
        if args.model_type == "base":
            from models.base_models.lumina_mgpt.inference_solver import FlexARInferenceSolver
            model = FlexARInferenceSolver(
                model_path=args.model_path,
                precision=args.precision,
                target_size=args.target_size,
                cfg_mode=args.cfg_mode,
            )

        elif args.model_type == "eagle":
            from models.base_models.lumina_mgpt.eagle_inference_solver import FlexARInferenceSolver
            model = FlexARInferenceSolver(
                model_path=args.model_path,
                drafter_path=args.drafter_path,
                precision=args.precision,
                target_size=args.target_size,
                cfg_mode=args.cfg_mode,
                eagle_version=args.eagle_version,
            )
        else:
            raise ValueError(f"Model type {args.model_type} is not supported for model {args.model}")
    elif args.model == "anole":
        
        if args.model_type == 'base':
            from models.kv_variants.modeling_anole_kv import ChameleonForConditionalGeneration
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            model = ChameleonForConditionalGeneration.from_pretrained(args.model_path).to(dtype=dtype, device='cuda')
            model.eval()
        
        elif args.model_type == 'eagle':
            from models.ea_anole import EaModel
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            model = EaModel.from_pretrained(base_model_path=args.model_path, ea_model_path=args.drafter_path).to(dtype=dtype, device='cuda')
            model.eval()
        else:
            raise ValueError(f"Model type {args.model_type} is not supported for model {args.model}")
    elif "llamagen" in args.model:
        if args.model_type == "base":
            # Load the base LLAMAGen model
            from models.kv_variants.modeling_llamagen_kv import LlamaForCausalLM

            dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }[args.precision]
            model = LlamaForCausalLM.from_pretrained(args.model_path).to(dtype=dtype, device="cuda")
            model.eval()
        elif args.model_type == "eagle":
            # Load the LLAMAGen model with EAGLE drafter
            from models.ea_llamagen import EaModel

            dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }[args.precision]
            model = EaModel.from_pretrained(
                base_model_path=args.model_path, 
                ea_model_path=args.drafter_path
            ).to(dtype=dtype, device="cuda")
            model.eval()
        else:
            raise ValueError(
                f"Model type {args.model_type} is not supported for model {args.model}"
            )
    else:
        raise NotImplementedError(f"Model {args.model} is not supported")

    return model

def load_prompts(args):
    prompts = []
    if args.prompt == "MSCOCO2017Val":
        with open('data/prompts/captions_val2017_longest.json', 'r') as f:
            captions = json.load(f)
            for caption in captions:
                prompts.append(caption)
    elif args.prompt == "VVSPrompts":
        with open('data/prompts/VVS_prompts.json', 'r') as f:
            captions = json.load(f)
            for caption in captions:
                prompts.append(caption)
    else:
        print("Prompt file not recognized. Please provide a valid prompt file.")
        raise ValueError(f"Prompt file {args.prompt} not recognized. Please provide a valid prompt file.")
        # Single prompt input
        prompts = [args.prompt] * args.num_images

    if args.slice is not None:
        assert re.match(r'^\d+-\d+$', args.slice), f"Invalid format: '{args.slice}'. Expected format is 'start-end'."

        start, end = map(int, args.slice.split('-'))
        assert start < end, f"Invalid range: '{args.slice}'. Start value must be less than end value."
        assert start >= 0 and end >= 0, "Slice values must be non-negative."

        prompts = prompts[start:end]
    
    if args.num_images < len(prompts):
        print(f"Number of images to generate is less than the number of prompts. Sampling {args.num_images} prompts.")
        prompts = random.sample(prompts, args.num_images)
    else:
        print(f"Number of images to generate is greater than the number of prompts. Generating only {len(prompts)} images and no sampling.")
        pass
    
    return prompts

#======================= Image Generation Function =========================

def generate_and_save_image(model, model_name, prompt, img_save_path, **kwargs):
    # print(f"Generating image for prompt: {prompt}")
    if model_name in ["llamagen", "llamagen2"]:
        if model_name == "llamagen":
            max_gen_len = 256
        else:
            max_gen_len = 1024
        generate_params = {
            "model": model_name,
            "prompt": [prompt],
            "max_length": max_gen_len,
            "temperature": kwargs["temperature"],
            "top_k": kwargs["top_k"],
            "top_p": kwargs["top_p"],
            "cfg": kwargs["cfg"],
            "static_tree": kwargs["static_tree"],
        }
    elif model_name == "lumina_mgpt":
        generate_params = {
            "images": [],
            "qas": [[prompt, None]],
            "max_gen_len": 2354,
            "temperature": kwargs["temperature"],
            "top_k": kwargs["top_k"],
            "cfg_scale": kwargs["cfg"],
        }
    else:
        raise NotImplementedError(f"Model {model_name} is not supported for image generation.")
    
    if "lantern" in kwargs:
        generate_params["lantern"] = kwargs.get("lantern", False)
        generate_params["lantern_k"] = kwargs.get("lantern_k", 1000)
        generate_params["lantern_delta"] = kwargs.get("lantern_delta", 0.1)

    if USE_EXPERIMENTAL_FEATURES:
        generate_params["tree_choices"] = kwargs["tree_choices"]
        generate_params["drafter_top_k"] = kwargs["drafter_top_k"]
        
    # for skip experiment
    if "skip" in kwargs:
        generate_params["idx"] = kwargs.get("idx")
        generate_params["prompts"] = kwargs.get("prompts")
        generate_params["skip"] = kwargs.get("skip", False)
        generate_params["skip_policy"] = kwargs.get("skip_policy", "uniform")
        generate_params["skip_interval"] = kwargs.get("skip_interval", 1024)
        generate_params["select_token"] = kwargs.get("select_token", "highest_cs")
        generate_params["reuse_feature"] = kwargs.get("reuse_feature", "reuse_prev")
        generate_params["feat_shift_lab"] = kwargs.get("feat_shift_lab")
        generate_params["feat_shift_staleness"] = kwargs.get("feat_shift_staleness")
        generate_params["token_shift_lab"] = kwargs.get("token_shift_lab")
        generate_params["token_shift_interval"] = kwargs.get("token_shift_interval", 2)
        generate_params["feat_cons_lab"] = kwargs.get("feat_cons_lab")
        generate_params["feat_cons_staleness"] = kwargs.get("feat_cons_staleness")
        generate_params["AR_lab"] = kwargs.get("AR_lab")
        generate_params["sim_threshold"] = kwargs.get("sim_threshold", 0.70)
        
    if generate_params["AR_lab"]:
        result = model.generate(**generate_params)
        if len(result) == 4:
            generated_tokens, step_compression, latency, profiling_summary = result
        else:
            generated_tokens, step_compression, latency = result
            profiling_summary = None
    else:
        (
            generated_tokens, 
            step_compression, 
            latency, 
            profile_data_dict,
            accpet_length_list,
            skipped_acp_len_lst,
            verified_acp_len_lst,
            skipped_steps
        )= model.generate(**generate_params)
    _, generated_image = model.decode_ids(generated_tokens)

    if "llamagen" in model_name:
        save_image(generated_image, img_save_path, normalize=True, value_range=(-1, 1))
    elif model_name in ["lumina_mgpt"]:
        generated_image[0].save(img_save_path, "png")

    if generate_params["AR_lab"]:
        if profiling_summary is not None:
            return (
                step_compression,
                latency,
                profiling_summary,
            )
        else:
            return (
                step_compression,
                latency,
            )
    else:
        return (
            step_compression,
            latency,
            profile_data_dict,
            accpet_length_list,
            skipped_acp_len_lst,
            verified_acp_len_lst,
            skipped_steps,
        )


#======================= Main Generation Loop =========================

def run_generate_image(args):
    assert args.model_type != "vllm", "VLLM model is not supported for single image generation"

    if args.set_seed:
        set_seed(args.random_seed)
        
    if args.debug:
        import debugpy

        try:
            debugpy.listen(("localhost", 7999))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
        except Exception as e:
            print(f"Debugpy failed to start: {e}")
            pass
    
    model = load_model(args)
    prompts = load_prompts(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # store profile data of all prompts, each prompt with one dict in this list
    profile_data_dict_all = {}
    all_profiling_data = {}
    
    if args.skip and args.skip_interval:
        print(f"Skipping verification is activated and interval is set to {args.skip_interval}.")

    global_statistics = {}  
    for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        if idx < args.start_idx or idx >= args.end_idx:
            continue
        if args.model == "lumina_mgpt":
            q1 = f"Generate an image of 768x768 according to the following prompt:\n{prompt}"
        else:
            q1 = prompt

        #************** Passing new args to ea_llamagen genreate func here *****************
            
        generate_image_kwargs = {
            "model" : model,
            "model_name" : args.model,
            "prompt" : q1,
            "temperature" : args.temperature,
            "top_k" : args.top_k,
            "top_p" : args.top_p,
            "cfg" : args.cfg,
            "lantern": args.lantern,
            "lantern_k": args.lantern_k,
            "lantern_delta": args.lantern_delta,
            "img_save_path": f"{args.output_dir}/prompt_{idx}.png",
            "static_tree": args.static_tree,
            # for VVS experiment
            "idx": idx,
            "prompts": args.prompt,
            "skip": args.skip,
            "skip_interval": args.skip_interval,
            "skip_policy": args.skip_policy,
            "select_token": args.select_token,
            "reuse_feature": args.reuse_feature,
            "feat_shift_lab": args.feat_shift_lab,
            "feat_shift_staleness": args.feat_shift_staleness,
            "feat_cons_lab": args.feat_cons_lab,
            "feat_cons_staleness": args.feat_cons_staleness,
            "token_shift_lab": args.token_shift_lab,
            "token_shift_interval": args.token_shift_interval,
            "post_verify": args.post_verify,
            "AR_lab": args.AR_lab,
            "sim_threshold": args.sim_threshold,
        }

        if USE_EXPERIMENTAL_FEATURES:
            try:
                tree_choices = getattr(choices, args.tree_choices)
            except AttributeError:
                print(f"Tree choices {args.tree_choices} is not a valid choice")
                return
            
            generate_image_kwargs["tree_choices"] = tree_choices
            generate_image_kwargs["drafter_top_k"] = args.drafter_top_k
        
        if generate_image_kwargs["AR_lab"]:
            if idx == 0:
                print("AR_lab is triggered, running vanilla AR generation...")
            result = generate_and_save_image(**generate_image_kwargs)
            step_compression, latency = result
            statistics = {
                "prompt": prompt,
                "step_compression": step_compression,
                "latency": latency,
            }
        else:
            (
                step_compression, 
                latency, 
                profile_data_dict,
                accpet_length_list,
                skipped_acp_len_lst,
                verified_acp_len_lst,
                skipped_steps
            ) = generate_and_save_image(**generate_image_kwargs)
            profile_data_dict_all[f"prompt_{idx}"] = profile_data_dict
            statistics = {
                "prompt": prompt,
                "step_compression": step_compression,
                "latency": latency,
                "acp_len_list": str(accpet_length_list),
                "skipped_acp_len_list": str(skipped_acp_len_lst),
                "verified_acp_len_lst": str(verified_acp_len_lst),
                "num_skipped_steps": str(skipped_steps)
            }

        global_statistics[f"prompt_{idx}"] = statistics

    with open(
        f"{args.output_dir}/global_statistics_{args.start_idx}_{args.end_idx}.json", "w"
    ) as f:
        json.dump(global_statistics, f, indent=4)
    if generate_image_kwargs["AR_lab"]:
        with open(
            f"{args.output_dir}/profiling_data_{args.start_idx}_{args.end_idx}.json",
            "w",
        ) as f:
            json.dump(all_profiling_data, f, indent=4)
        print(
            f"All profiling data is saved to: {args.output_dir}/profiling_data_{args.start_idx}_{args.end_idx}.json"
        )

    with open(f"{args.output_dir}/generation_configs.json", "w") as f:
        json.dump(vars(args), f, indent=4, separators=(",", ": "))
    if args.model == "lumina_mgpt":
        torch.save(
            profile_data_dict_all,
            f"./profile_data_lumina/{args.prompt}_{args.start_idx}_{len(prompts) - 1}_lst.pt",
        )
        with open(
            f"./profile_data_lumina/generation_configs_{args.prompt}_{args.start_idx}_{len(prompts) - 1}.json",
            "w",
        ) as f:
            json.dump(vars(args), f, indent=4, separators=(",", ": "))
    elif args.skip and args.skip_interval == 1024 and args.skip_policy == "uniform":
        torch.save(
            profile_data_dict_all,
            f"./profile_data/{args.prompt}_{args.start_idx}_{len(prompts) - 1}_lst.pt",
        )
        with open(
            f"./profile_data/generation_configs_{args.prompt}_{args.start_idx}_{len(prompts) - 1}.json",
            "w",
        ) as f:
            json.dump(vars(args), f, indent=4, separators=(",", ": "))
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for image generation",
        default="llamagen",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Model type; choices: ['eagle', 'base']",
        default="eagle",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model",
        default="Alpha-VLLM/Lumina-mGPT-7B-768",
    )
    parser.add_argument(
        "--drafter_path",
        type=str,
        help="Path to the drafter model",
        default="ckpts/lumina_mgpt/trained_drafters/llamagen2_drafter",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="Precision for model; choices: ['fp16', 'bf16', 'fp32']",
        default="bf16",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        help="Target size for generated images; must be a multiple of 64",
        default=768,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for image generation",
        default="data/prompts/captions_val2017_longest.json",
    )
    parser.add_argument(
        "--num_images", type=int, help="Number of images to generate", default=10
    )
    parser.add_argument(
        "--slice",
        type=str,
        help="Slice of prompts to use; format: 'start-end'",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for generated images",
        default="output_images",
    )
    parser.add_argument(
        "--set_seed", action="store_true", help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Set debug mode"
    )
    parser.add_argument(
        "--random_seed", type=int, help="Random seed for reproducibility", default=42
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature for generation", default=1.0
    )
    parser.add_argument("--top_k", type=int, help="Top-k for generation", default=2000)
    parser.add_argument("--top_p", type=float, help="Top-p for generation", default=1.0)
    parser.add_argument("--cfg", type=float, help="CFG for generation", default=3.0)
    
    
    # LANTERN-specific arguments
    parser.add_argument("--lantern", action="store_true", help="Use LANTERN for image generation")
    parser.add_argument("--lantern_k", type=int, default="1000", help="Value of k for LANTERN")
    parser.add_argument("--lantern_delta", type=float, default="0.1", help="Value of delta for LANTERN")
    parser.add_argument("--grid_search", action="store_true", help="Run grid search for LANTERN hyperparameters")

    parser.add_argument("--static_tree", action="store_true", help="Use static tree based drafting")
    parser.add_argument("--eagle_version", type=int, default=2, help="EAGLE version")
    parser.add_argument("--cfg_mode", type=str, default="parallel", help="CFG mode") # for lumina

    # Experimental arguments
    parser.add_argument("--tree_choices", type=str, help="Tree choice for LANTERN",
                        default="mc_sim_7b_63")
    parser.add_argument("--drafter_top_k", type=int, default=None, help="Top-k for drafter")
    
    
    # legacy arguments
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for image generation")
    parser.add_argument("--end_idx", type=int, default=10000, help="End index for image generation")
    
    # VVS experiment arguments
    parser.add_argument("--skip", action="store_true", help="Enable VVS experiment")
    parser.add_argument("--skip_interval", type=int, default=1024, help="Interval for skipping verification")
    parser.add_argument("--skip_policy",
        type=str,
        help="Skip policy; choices: ['uniform', 'dynamic', 'center']",
        default="uniform")
    parser.add_argument(
        "--select_token", 
        type=str,
        help="Token selection policy; choices: ['highest_cs', 'upper_simi', 'random']",
        default="highest_cs")
    parser.add_argument(
        "--reuse_feature", 
        type=str,
        help="Future reusing policy; choices: ['reuse_prev', 'fuse_prev', 'reuse_upper']",
        default="reuse_prev")
    parser.add_argument(
        "--feat_shift_lab",
        action="store_true",
        help="To explore the effect on ACR of feature shifting",
    )
    parser.add_argument(
        "--feat_shift_staleness",
        type=int,
        default=-1,
        help="Staleness for feature shifting",
    )
    parser.add_argument(
        "--token_shift_lab",
        action="store_true",
        help="To explore the effect on ACR of token selection",
    )
    parser.add_argument(
        "--token_shift_interval",
        type=int,
        default=2,
        help="Interval for token shifting",
    )
    parser.add_argument(
        "--feat_cons_lab",
        action="store_true",
        help="To explore the effect on ACR of feature consensus",
    )
    parser.add_argument(
        "--feat_cons_staleness",
        nargs="+",
        type=int,
        help="Staleness for feature consensus",
    )
    parser.add_argument(
        "--post_verify",
        action="store_true",
        help="To explore the effect on ACR of post verification",
    )
    parser.add_argument(
        "--AR_lab",
        action="store_true",
        help="To explore the effect on ACR of AR baselines",
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.70,
        help="Threshold for similarity",
    )
    
    return parser
