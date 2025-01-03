import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from load_token import load_token

HF_TOKEN = load_token()


def load_quantized_model(model_name, hf_token, llm_int8_threshold=6.0):
    print("Loading quantized model...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization
        llm_int8_threshold=llm_int8_threshold,  # threshold for outlier features
    )
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        token=hf_token
    )
    return quantized_model


def load_full_model(model_name, hf_token, device):
    print("Loading non-quantized model...")
    full_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        token=hf_token
    ).to(device)
    return full_model


if __name__ == "__main__":    
    # model_name = "meta-llama/Llama-3.1-8B"
    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    prompt = "Machine Learning is"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quantized_model=load_quantized_model(model_name, HF_TOKEN, device)
    full_model=load_full_model(model_name, HF_TOKEN, device)