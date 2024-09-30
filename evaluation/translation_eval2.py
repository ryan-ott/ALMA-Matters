import argparse
import os
import sys
import torch
import json
import logging
from tqdm import tqdm

# Utilities from the SliceGPT package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../compression/pruning/TransformerCompression/src")))
from slicegpt.hf_utils import load_sliced_model, get_model_and_tokenizer

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
        required=True,
        help="Directory containing JSON test data files.")
    parser.add_argument(
        '--beam',
        type=int,
        required=True,
        help="Beam size for generation.")
    parser.add_argument(
        '--gen_max_tokens',
        type=int,
        default=256,
        help="Max number of tokens to generate.")
    parser.add_argument(
        '--dtype',
        required=True,
        help="Data type: float16, bfloat16, etc.")
    return parser


def dynamic_batching(tokenizer, texts, batch_size, max_length):
    batch = []
    batch_length = 0

    for text in texts:
        input_length = len(tokenizer.encode(text, truncation=True, max_length=max_length))
        if len(batch) > 0 and (batch_length + input_length > max_length or len(batch) == batch_size):
            yield batch
            batch = []
            batch_length = 0
        
        batch.append(text)
        batch_length = max(batch_length, input_length)

    if len(batch) > 0:
        yield batch


def load_model(args):
    if args.sliced_model_path:
        logging.info(f"Loading sliced {args.model} model from {args.sliced_model_path} with sparsity {args.sparsity}")
        model_adapter, tokenizer = load_sliced_model(
            args.model,
            args.sliced_model_path,
            token=args.hf_token,
            sparsity=args.sparsity,
            round_interval=args.round_interval
        )
    else:
        logging.info(f"Loading original {args.model} model")
        model_adapter, tokenizer = get_model_and_tokenizer(args.model)

    model_adapter.model.to('cuda')

    return model_adapter, tokenizer


def main(args):
    model_adapter, tokenizer = load_model(args)
    
    for filename in tqdm(os.listdir(args.data_dir), desc="Evaluating tranlations", unit="language pair"):
        if filename.endswith(".json"):
            file_path = os.path.join(args.data_dir, filename)
            src, tgt = filename.replace("ALMA_test_", "").replace(".json", "").split('-')
            result_filename = f"result_{src}-{tgt}.txt"
            result_path = os.path.join(args.save_dir, result_filename)

            with open(result_path, "w") as file_out:
                with open(file_path, 'r') as f:
                    lines = json.load(f)

                total_batches = (len(lines) + args.batch_size - 1) // args.batch_size
                for batch in tqdm(dynamic_batching(tokenizer,
                                                   [line['translation'][src] for line in lines],
                                                   args.batch_size,
                                                   args.gen_max_tokens), total=total_batches):
                    prompts = [f"Translate this from {src} to {tgt}:\n{src}: {line}\n{tgt}:" for line in batch]

                    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')
                    with torch.no_grad():
                        generated_ids = model_adapter.model.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            num_beams=args.beam,
                            max_new_tokens=args.gen_max_tokens
                        )

                    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    for prompt, output in zip(prompts, outputs):
                        translation = output[len(prompt):].strip()
                        file_out.write(translation.replace("\n", " ") + "\n")

            logging.info(f"Completed evaluation for {src}-{tgt}. Results saved to {result_path}")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
