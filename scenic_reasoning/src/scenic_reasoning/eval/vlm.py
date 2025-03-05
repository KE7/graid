# import outlines
# from transformers import LlavaNextForConditionalGeneration
# from PIL import Image


# model = outlines.models.transformers_vision(
#     "llava-hf/llava-v1.6-mistral-7b-hf",
#     model_class=LlavaNextForConditionalGeneration,
# 	device="cuda",
# )

# description_generator = outlines.generate.text(model)
# description_generator(
#     "<image> detailed description:",
#     [Image.open('../demo/demo.jpg')]
# )



import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-base-490-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to('cuda').eval()

image = Image.open("../demo/demo.jpg").convert('RGB')
inputs = model.build_conversation_input_ids(tokenizer, query='', images=[image])
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))
