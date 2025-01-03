import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

# Function to evaluate a model
def evaluate_model(model, tokenizer, device, prompt, description):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move inputs to the correct device
    
    # Measure generation time
    start_time = time.time()
    output = model.generate(**inputs, max_length=50)
    end_time = time.time()
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Measure GPU memory usage
    gpu_memory = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else None
    
    # Print results
    print(f"=== {description} ===")
    print(f"Generated Text: {generated_text}")
    print(f"Generation Time: {end_time - start_time:.2f} seconds")
    if gpu_memory is not None:
        print(f"GPU Memory Allocated: {gpu_memory:.2f} MB")
    print("\n")

if __name__ == "__main__":    
    model_name = "meta-llama/Llama-3.1-8B"  # Replace with your model identifier
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Machine Learning is"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate quantized model
    print("Loading quantized model...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization
        llm_int8_threshold=6.0,  # Adjust threshold for outlier features
    )
    quantized_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    evaluate_model(quantized_model, tokenizer, device, prompt, "Quantized Model")

    # Evaluate full model (non-quantized)
    print("Loading non-quantized model...")
    full_model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16).to("cuda")
    evaluate_model(full_model, tokenizer, device, prompt, "Non-Quantized Model")