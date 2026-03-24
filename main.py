import argparse
import entrypoints.gen_images as gen_images
import entrypoints.eval_fid_clip as eval_fid_clip
import entrypoints.eval_hpsv2 as eval_hpsv2
import entrypoints.eval_prec_recall as eval_prec_recall
import entrypoints.extract_code as extract_code
import entrypoints.gen_train_data as gen_train_data

def get_task_parser(task_name):
    """Returns the argument parser for the specified task."""
    if task_name == "gen_images":
        return gen_images.parse_args()
    elif task_name == "eval_fid_clip":
        return eval_fid_clip.parse_args()
    elif task_name == "eval_hpsv2":
        return eval_hpsv2.parse_args()
    elif task_name == "eval_prec_recall":
        return eval_prec_recall.parse_args()
    elif task_name == "extract_code":
        return extract_code.parse_args()
    elif task_name == "gen_train_data":
        return gen_train_data.parse_args()
    else:
        raise ValueError(f"Invalid task name: {task_name}")
    
def get_task_runner(task_name):
    """Returns the function to run for the specified task."""
    if task_name == "gen_images":
        return gen_images.run_generate_image
    elif task_name == "eval_fid_clip":
        return eval_fid_clip.run_eval_fid_clip
    elif task_name == "eval_hpsv2":
        return eval_hpsv2.run_eval_hpsv2
    elif task_name == "eval_prec_recall":
        return eval_prec_recall.run_eval_prec_recall
    elif task_name == "extract_code":
        return extract_code.run_extract_code
    elif task_name == "gen_train_data":
        return gen_train_data.run_generate_data
    else:
        raise ValueError(f"Invalid task name: {task_name}")
    

def main():
    parser = argparse.ArgumentParser(description='var_spec')
    # subparsers is an action that allows us to define sub-commands
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands for different tasks")

    # Add subparsers for each task
    subparsers.add_parser("gen_images", help="Generate images")
    subparsers.add_parser("eval_fid_clip", help="Evaluate FID and CLIP")
    subparsers.add_parser("eval_hpsv2", help="Evaluate HPSv2")
    subparsers.add_parser("eval_prec_recall", help="Evaluate Precision and Recall")
    subparsers.add_parser("extract_code", help="Extract code from images")
    subparsers.add_parser("gen_train_data", help="Generate training data")
    
    args, remaining_args = parser.parse_known_args()

    # get specific task parser and its arguments
    task_parser = get_task_parser(args.command)
    task_args = task_parser.parse_args(remaining_args)

    # run the specific task with its arguments
    task_runner = get_task_runner(args.command)
    task_runner(task_args)
    
if __name__ == "__main__":
    main()