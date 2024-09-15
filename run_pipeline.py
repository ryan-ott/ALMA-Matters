"""
Main script to run entire pipeline (from fine-tuning, compression to evaluation)
"""

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer


def main():
    parser = argparse.ArgumentParser(description="Run the entire pipeline")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune the model")
    parser.add_argument("--compress", action="store_true", help="Compress the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    args = parser.parse_args()
    
    if args.fine_tune:
        # fine_tune_model()
        pass
    if args.compress:
        # compress_model()
        pass
    if args.evaluate:
        # evaluate_model()
        pass
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained("haoranxu/ALMA-7B", torch_dtype=torch.float16, device_map="auto")
    tokenizer = LlamaTokenizer.from_pretrained("haoranxu/ALMA-7B", padding_side='left')

    # Add the source setence into the prompt template
    prompt="Translate this from German to English:\German: Ich esse gerne Fischbrötchen wenn das Wetter schön ist.\nEnglish:"
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=40, truncation=True).input_ids.cuda()

    # Translation
    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=20, do_sample=True, temperature=0.6, top_p=0.9)
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(outputs)


if __name__ == "__main__":
    main()