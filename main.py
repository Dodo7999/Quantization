from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
login(token="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")
import torch

device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu') # the device to load the model onto

model_id = "CohereForAI/aya-23-35B"
tokenizer = AutoTokenizer.from_pretrained(model_id).to(device)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# Format message with the command-r-plus chat template
messages = [{"role": "user", "content": "Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)