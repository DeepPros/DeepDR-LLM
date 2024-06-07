import argparse
import json, os

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default="DeepDR-LLM/Module1/llama-7b-weights", type=str, required=False)
parser.add_argument('--lora_model', default="DeepDR-LLM/Module1/lora-adapter-weights", type=str, help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path', default="DeepDR-LLM/Module1/lora-adapter-weights", type=str)
parser.add_argument('--with_prompt', default=True, help="wrap the input with the prompt automatically")
parser.add_argument('--interactive', default=True, help="run in the instruction mode (single-turn)")
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--system_prompt', type=str, default=DEFAULT_SYSTEM_PROMPT, help="The system prompt of the prompt template.")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from peft import  PeftModel
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

generation_config = GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=512
)

def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})

if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True
    )
    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size!=tokenizer_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenizer_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',).half()
    else:
        model = base_model
    if device==torch.device('cpu'):
        model.float()
    model.eval()

    with torch.no_grad():
        while True:
            raw_input_text = input("Input:")
            if len(raw_input_text.strip())==0:
                break
            input_text = generate_prompt(instruction=raw_input_text, system_prompt=args.system_prompt)
            inputs = tokenizer(input_text,return_tensors="pt")
            generation_output = model.generate(
                    input_ids = inputs["input_ids"].to(device),
                    attention_mask = inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    generation_config = generation_config
                )
            s = generation_output[0]
            output = tokenizer.decode(s,skip_special_tokens=True)
            response = output.split("[/INST]")[-1].strip()
            print("Response: ",response)
            print("\n")
