import argparse
import os
import sys
import torch
import json
import logging
import time  # Added import for timing
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

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
        '--lora-model-path',
        help="Directory containing the lora adaptor (if applicable).")
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
        help="Batch size for evaluating translations.")
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
        help="Path to save the results and metrics.")
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
    parser.add_argument(
        '--max_sequence_length',
        type=int,
        default=512,
        help="Maximum sequence length for tokenizer encoding.")
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help="Device to run the model on.")
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
    elif args.lora_model_path:
        logging.info(f"Loading lora adaptors for {args.model} model from {args.sliced_model_path}")
        # Set up 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4' 
        )
        # Load the model with quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map='auto',
        )
        adapter_dir = os.path.join(args.lora_model_path, 'adapter_model')
        model_adapter = PeftModel.from_pretrained(base_model, adapter_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.lora_model_path)
    else:
        logging.info(f"Loading original {args.model} model")
        model_adapter, tokenizer = get_model_and_tokenizer(args.model)

    model_adapter.model.to(args.device)
    model_adapter.model.eval()

    return model_adapter, tokenizer

def main(args):
    model_adapter, tokenizer = load_model(args)

    # Initialize metrics
    torch.cuda.reset_peak_memory_stats()
    total_tokens_processed = 0
    total_time_taken = 0.0  # In seconds
    total_compute_time = 0.0  # In seconds

    batch_times = []
    memory_usages = []

    num_gpus = torch.cuda.device_count()
    logging.info(f"Number of GPUs available: {num_gpus}")

    # Loop over the translation files
    for filename in tqdm(os.listdir(args.data_dir), desc="Evaluating translations", unit="language pair"):
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
                                                   args.max_sequence_length), total=total_batches):
                    prompts = [f"Translate this from {src} to {tgt}:\n{src}: {line}\n{tgt}:" for line in batch]

                    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_sequence_length).to(args.device)

                    # Record start time
                    batch_start_time = time.perf_counter()

                    with torch.no_grad():
                        generated_ids = model_adapter.model.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            num_beams=args.beam,
                            max_new_tokens=args.gen_max_tokens
                        )

                    # Record end time
                    batch_end_time = time.perf_counter()
                    batch_time = batch_end_time - batch_start_time
                    batch_times.append(batch_time)
                    total_time_taken += batch_time

                    # Calculate tokens generated
                    tokens_generated = generated_ids.numel() - inputs.input_ids.numel()
                    total_tokens_processed += tokens_generated

                    # Record peak memory usage for this batch
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
                    memory_usages.append(peak_memory)

                    # Reset memory stats for next batch
                    torch.cuda.reset_peak_memory_stats()

                    # Calculate time per token
                    time_per_token = batch_time / tokens_generated if tokens_generated > 0 else 0
                    logging.info(f"Batch Size: {len(batch)}, Batch Time: {batch_time:.4f}s, Tokens Generated: {tokens_generated}, Time per Token: {time_per_token*1000:.2f} ms, Peak Memory: {peak_memory:.2f} GB")

                    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    for prompt, output in zip(prompts, outputs):
                        translation = output[len(prompt):].strip()
                        file_out.write(translation.replace("\n", " ") + "\n")

            logging.info(f"Completed evaluation for {src}-{tgt}. Results saved to {result_path}")

    # Calculate overall metrics
    total_compute_time = total_time_taken * num_gpus  # Total compute time in seconds
    throughput = total_tokens_processed / total_time_taken if total_time_taken > 0 else 0  # tokens per second
    average_time_per_token = (total_time_taken / total_tokens_processed) * 1000 if total_tokens_processed > 0 else 0  # in ms
    average_peak_memory = max(memory_usages) if memory_usages else 0  # Max peak memory across batches

    logging.info(f"Total Tokens Processed: {total_tokens_processed}")
    logging.info(f"Total Time Taken: {total_time_taken:.4f} seconds")
    logging.info(f"Throughput: {throughput:.2f} tokens/sec")
    logging.info(f"Average Time per Token: {average_time_per_token:.2f} ms")
    logging.info(f"Total Compute Time: {total_compute_time:.4f} seconds")
    logging.info(f"Average Peak Memory Usage: {average_peak_memory:.2f} GB")

    # Save metrics to a JSON file
    metrics = {
        "total_tokens_processed": total_tokens_processed,
        "total_time_taken_sec": total_time_taken,
        "throughput_tokens_per_sec": throughput,
        "average_time_per_token_ms": average_time_per_token,
        "total_compute_time_sec": total_compute_time,
        "num_gpus": num_gpus,
        "average_peak_memory_GB": average_peak_memory,
        "batch_size": args.batch_size,
        "beam_size": args.beam,
        "model": args.model,
        "dtype": args.dtype
    }

    metrics_filename = os.path.join(args.save_dir, "metrics.json")
    with open(metrics_filename, 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
    logging.info(f"Metrics saved to {metrics_filename}")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
