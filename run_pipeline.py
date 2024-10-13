"""
Main script to run entire pipeline (from fine-tuning, compression to evaluation)
"""
import argparse
import logging
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Utilities from the SliceGPT package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "compression/pruning/TransformerCompression/src")))
from slicegpt.hf_utils import load_sliced_model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        required=True,
        help="Model to load e.g. meta-llama/Llama-2-7b-hf or haoranxu/ALMA-7B",)
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",)
    parser.add_argument(
        '--sliced-model-path',
        help="Directory containing the sliced model (if applicable).")
    parser.add_argument(
        '--sparsity',
        type=float,
        help="Sparsity level of the sliced model (if applicable).")
    parser.add_argument(
        '--round-interval',
        type=int,
        default=8,  # Best for A100 GPUs according to Snellius docs
        help="Interval for rounding the model weights.")
    parser.add_argument(
        '--hf-token',
        type=str,
        default=os.getenv('HF_TOKEN', None))
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help="Batch size for evaluating with lm eval harness.")
    parser.add_argument(
        '--wandb-project',
        type=str,
        default="slicegpt-lm-eval",
        help="wandb project name.")
    parser.add_argument(
        '--no-wandb',
        action="store_true",
        help="Disable wandb.")
    parser.add_argument(
        '--save-dir',
        type=str,
        default=".",
        help="Path to save the lm eval results")
    parser.add_argument(
        '--data_dir',
        required=False,
        help="Directory containing JSON test data files.")
    parser.add_argument(
        '--beam',
        type=int,
        default=5,
        help="Beam size for generation.")
    parser.add_argument(
        '--gen_max_tokens',
        type=int,
        default=256,
        help="Max number of tokens to generate.")
    parser.add_argument(
        '--dtype',
        default="float16",
        help="Data type: float16, bfloat16, etc.")
    parser.add_argument(
        '--fine-tune',
        action="store_true",
        help="Whether to fine-tune the model")
    parser.add_argument(
        '--compress',
        action="store_true",
        help="Compress the model")
    parser.add_argument(
        '--evaluate',
        action="store_true",
        help="Evaluate the model")
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help="Prompt for the model to translate. Please follow this format: 'Translate this from [language1] to [language2]:\[language1]: [sentence]\n[language2]:'")
    parser.add_argument(
        '--debug',
        action="store_true",
        help="Run in debug mode. Adds verbosity")
    return parser


def load_alma_model(args):
    logging.info(f"Loading sliced {args.model} model from {args.sliced_model_path} with sparsity {args.sparsity}")
    model_adapter, tokenizer = load_sliced_model(
        args.model,
        args.sliced_model_path,
        token=args.hf_token,
        sparsity=args.sparsity,
        round_interval=args.round_interval
    )
    model_adapter.model.to('cuda')
    return model_adapter, tokenizer


def main(args):
    if args.sliced_model_path:
        model, tokenizer = load_alma_model(args)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left', device='cuda')

    input_ids = tokenizer(args.prompt, return_tensors="pt", padding=True, max_length=40, truncation=True).input_ids.cuda()

    # Translation
    with torch.no_grad():
        if args.sliced_model_path:
            generated_ids = model.model.generate(
                input_ids=input_ids,
                num_beams=args.beam,
                max_new_tokens=args.gen_max_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=4.0,  # THESE THREE HERE
                no_repeat_ngram_size=3,  # MIGHT HELP
                eos_token_id=tokenizer.eos_token_id)  # TO AVOID REPEATING
        else:
            generated_ids = model.generate(
                input_ids=input_ids,
                num_beams=args.beam,
                max_length=args.gen_max_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9)
                # repetition_penalty=1.2,
                # no_repeat_ngram_size=3,
                # eos_token_id=tokenizer.eos_token_id)
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("\n++++++++++++++")
    print(outputs)
    print("\n++++++++++++++")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    main(args)