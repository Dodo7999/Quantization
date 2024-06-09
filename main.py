import logging

import numpy as np
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer

login(
    token="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",
)

device = torch.device(
    f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')  # the device to load the model onto

model_id = "CohereForAI/aya-23-35B"
quant_path = "aya23_35B_gptq"

quantize_config = BaseQuantizeConfig(
    bits=4,  # 4 or 8
    group_size=128,
    damp_percent=0.01,
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    static_groups=False,
    sym=True,
    true_sequential=True,
    model_name_or_path=None,
    model_file_base_name="model"
)
max_len = 8192

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config, torch_dtype=torch.bfloat16,
                                            low_cpu_mem_usage=True, )


def check_role(x):
    new = []
    for i in x:
        if i['role'] == 'bot':
            i['role'] = 'assistant'
        if (i['role'] == 'assistant') or (i['role'] == 'user'):
            new.append(i)
    return np.array(new)


def chat_template(x):
    try:
        s = tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=False)
    except:
        s = '0'
    return s


def make_chat_aya(x):
    messages = [{'role': 'user', 'content': x.inputs},
                {'role': 'assistant', 'content': x.targets}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return prompt


def make_chat_orca(x):
    messages = [{'role': 'user', 'content': x.question},
                {'role': 'assistant', 'content': x.response}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return prompt


def load_saiga():
    data = load_dataset("IlyaGusev/saiga_scored")
    data = data[data.language.isin(['English', 'Russian'])]
    data = data[~data['source'].isin(['saiga_bot_gpt_4o', 'saiga'])]
    data['messages'] = data['messages'].apply(check_role)
    data['prompts'] = data['messages'].apply(chat_template)
    data = data[data.prompts != '0'].sample(5000)
    x = [prompt for prompt in data["prompts"]]
    res_data = []
    for text in x:
        model_inputs = tokenizer([text])
        input_ids = torch.tensor(model_inputs.input_ids[:max_len], dtype=torch.int)
        res_data.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id)))
    return res_data


data = load_saiga()

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
model.quantize(data, cache_examples_on_gpu=False, batch_size=1, use_triton=False)

model.save_quantized(quant_path, use_safetensors=True)
tokenizer.save_pretrained(quant_path)
