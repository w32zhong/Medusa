import torch
import time
import argparse
from medusa.model.medusa_model import MedusaModel
from fastchat.conversation import get_conv_template

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name or path.")
parser.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit quantization")
parser.add_argument("--temperature", type=float, default=0.)
parser.add_argument("--max-steps", type=int, default=512)
args = parser.parse_args()

model = MedusaModel.from_pretrained(
    args.model,
    None,
    medusa_num_heads = 4,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    load_in_8bit=args.load_in_8bit,
)
tokenizer = model.get_tokenizer()
device = model.base_model.device

conv = get_conv_template("vicuna_v1.1")
conv.append_message(conv.roles[0], "tell me a few interesting facts about the sun and the moon.")
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
print(prompt)

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

t0 = time.time()
last_len = 0
for stream in model.medusa_generate(input_ids, temperature=args.temperature, max_steps=args.max_steps):
    tokens = stream['out_tokens']
    print(tokenizer.decode(tokens[last_len:]), end=' ', flush=True)
    last_len = len(tokens)
t1 = time.time()
print('')
time_delta = t1 - t0
print('[speed]', time_delta, last_len, last_len / time_delta)
