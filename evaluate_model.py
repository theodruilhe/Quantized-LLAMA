import torch
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM

# Load your Hugging Face token from a credentials file
def load_token(file_path="credentials.txt"):
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("HUGGINGFACE_TOKEN="):
                return line.strip().split("=")[1]
    raise ValueError("HUGGINGFACE_TOKEN not found in the file")

# Function to measure memory usage
def get_memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 ** 2)  # Memory in MB
    return mem

# Function to measure inference time
def measure_inference_time(model, tokenizer, prompt, num_tokens=50, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=num_tokens)
    end_time = time.time()
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return end_time - start_time, generated_text

# Function to evaluate a model
def evaluate_model(model, tokenizer, prompt, model_type, device="cuda"):
    model.to(device)
    memory_usage = get_memory_usage()
    inference_time, generated_text = measure_inference_time(model, tokenizer, prompt, device=device)
    print(f"\n[{model_type} Model]")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    print(f"Inference Time: {inference_time:.2f} seconds")
    print(f"Generated Text: {generated_text}")
    return memory_usage, inference_time, generated_text

# Main comparison function
def compare_models(original_model_name, quantized_model_path, tokenizer, prompt, hf_token, device="cuda"):
    # Load and evaluate the original model
    print("\nLoading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(original_model_name, token=hf_token).to(device)
    original_memory, original_time, original_text = evaluate_model(original_model, tokenizer, prompt, "Original", device=device)

    # Load and evaluate the quantized model
    print("\nLoading quantized model...")
    quantized_model = ORTModelForCausalLM.from_pretrained(
        quantized_model_path,
        provider="CUDAExecutionProvider"
    )
    quantized_memory, quantized_time, quantized_text = evaluate_model(quantized_model, tokenizer, prompt, "Quantized", device=device)

    # Collect results
    results = {
        "Model": ["Original", "Quantized"],
        "Memory (MB)": [original_memory, quantized_memory],
        "Inference Time (s)": [original_time, quantized_time],
        "Generated Text": [original_text, quantized_text],
    }

    return results

# Save results and generate plots
def save_and_visualize_results(results, report_path="REPORT.md", plot_path="performance_comparison.png"):
    # Create a DataFrame
    df = pd.DataFrame(results)

    # Save results to a markdown file
    with open(report_path, "w") as file:
        file.write(df.to_markdown(index=False))
    print(f"\n[INFO] Results saved to {report_path}")

    # Plotting the results
    plt.figure(figsize=(10, 5))

    # Memory usage comparison
    plt.subplot(1, 2, 1)
    plt.bar(results["Model"], results["Memory (MB)"], color=["blue", "orange"])
    plt.title("Memory Usage Comparison")
    plt.ylabel("Memory (MB)")

    # Inference time comparison
    plt.subplot(1, 2, 2)
    plt.bar(results["Model"], results["Inference Time (s)"], color=["blue", "orange"])
    plt.title("Inference Time Comparison")
    plt.ylabel("Time (s)")

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"[INFO] Performance comparison plot saved as {plot_path}")
    plt.show()

if __name__ == "__main__":
    # Configurations
    HF_TOKEN = load_token()
    original_model_name = "meta-llama/Llama-3.1-8B"  # Hugging Face model path
    quantized_model_path = "./quantized_model"       # Path to quantized model
    prompt = "Once upon a time"
    tokenizer = AutoTokenizer.from_pretrained(original_model_name, token=HF_TOKEN)

    # Compare models
    results = compare_models(original_model_name, quantized_model_path, tokenizer, prompt, HF_TOKEN, device="cuda")

    # Save and visualize results
    save_and_visualize_results(results)