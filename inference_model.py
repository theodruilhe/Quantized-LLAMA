from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time


def load_token(file_path="credentials.txt"):
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("HUGGINGFACE_TOKEN="):
                return line.strip().split("=")[1]
    raise ValueError("HUGGINGFACE_TOKEN not found in the file")


def load_model_and_tokenizer(model_path, model_name, hf_token, device="cuda"):
    if model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_8bit=(model_path == "./quantized_model/"),
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def generate_text(model, tokenizer, prompt, device="cuda", max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Clamp logits to avoid inf/nan issues
    def modify_logits(logits):
        logits[logits != logits] = -float("inf")  # Replace NaN with -inf
        logits = torch.clamp(logits, min=-1e9, max=1e9)  # Clamp values
        return logits

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./quantized_model/"
    model_name = None  # Use `None` if loading from model_path
    prompt = "Machine Learning is "
    max_length = 50
    hf_token = load_token()

    model, tokenizer = load_model_and_tokenizer(model_path, model_name, hf_token, device)
    generated_text = generate_text(model, tokenizer, prompt, device, max_length)

    print("\n--- Generated Text ---")
    print(generated_text)