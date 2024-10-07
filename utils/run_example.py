import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# Load base model and LoRA weights
model = AutoModelForCausalLM.from_pretrained("haoranxu/ALMA-7B", torch_dtype=torch.float16).to('cuda')
tokenizer = AutoTokenizer.from_pretrained("haoranxu/ALMA-7B", padding_side='left', device='cuda')

# Add the source sentence into the prompt template
prompt="Translate this from English to German:\nEnglish: This is a test sentence that should be translated.\German:"
input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=40, truncation=True).input_ids.cuda()

# Translation
with torch.no_grad():
    generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=20, do_sample=True, temperature=0.6, top_p=0.9)
outputs = tokenizer.batch_decode(generated_ids)
print(outputs)