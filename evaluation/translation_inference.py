import argparse
import gc
import ijson
import json
import logging
import os
import sys
import torch
from tqdm import tqdm

# Utilities from the SliceGPT package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../compression/pruning/TransformerCompression/src")))
from slicegpt.hf_utils import load_sliced_model, get_model_and_tokenizer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        required=True,
        help="Model to load e.g. meta-llama/Llama-2-7b-hf or haoranxu/ALMA-7B")
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (for local models)")
    parser.add_argument(
        '--sliced-model-path',
        type=str,
        help="Directory containing the sliced model (if applicable).")
    parser.add_argument(
        '--sparsity',
        type=float,
        help="Sparsity level of the sliced model (if applicable).")
    parser.add_argument(
        '--round-interval',
        type=int,
        default=8, help="Interval for rounding the model weights.")
    parser.add_argument(
        '--hf-token',
        type=str,
        default=os.getenv('HF_TOKEN', None))
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8)
    parser.add_argument(
        '--save-dir',
        type=str,
        default=".", help="Path to save the lm eval results")
    parser.add_argument(
        '--json-file',
        required=True,
        help="Path to the JSON file containing test data")
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
    return parser


def json_generator(json_file_path):
    """
    Generator that yields one item at a time from a JSON array.
    Assumes the JSON file contains a list of objects.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        parser = ijson.items(f, 'item')
        for item in parser:
            yield item


def dynamic_batching(tokenizer, data_generator, batch_size, max_length):
    """
    Generator that yields batches of texts.
    """
    batch = []
    batch_length = 0
    for text in data_generator:
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
        model_adapter, tokenizer = get_model_and_tokenizer(args.model, args.model_path)
    model_adapter.model.to(args.device, dtype=getattr(torch, args.dtype))
    logging.info(f"Model dtype: {next(model_adapter.model.parameters()).dtype}")
    
    return model_adapter, tokenizer


def main(args):
    model_adapter, tokenizer = load_model(args)
    # Extract source and target language from the file name
    src, tgt = os.path.basename(args.json_file).replace("ALMA_test_", "").replace(".json", "").split('-')
    os.makedirs(args.save_dir, exist_ok=True)
    result_path = os.path.join(args.save_dir, f"result_{src}-{tgt}.txt")

    # Initialize the JSON generator
    data_gen = json_generator(args.json_file)
    text_gen = (item['translation'][src] for item in data_gen)

    with open(result_path, "w", encoding='utf-8') as file_out:
        for batch in tqdm(dynamic_batching(tokenizer, text_gen, args.batch_size, args.gen_max_tokens)):
            prompts = [f"Translate this from {src} to {tgt}:\n{src}: {line}\n{tgt}:" for line in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.gen_max_tokens).to(args.device)
            
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
            
            # Clear memory after each batch
            del inputs, generated_ids, outputs
            torch.cuda.empty_cache()
            gc.collect()
    logging.info(f"Completed evaluation for {src}-{tgt}. Results saved to {result_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
