from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import torch
import time
import psutil
import warnings
import pandas as pd
import matplotlib.pyplot as plt

# Suppress TracerWarnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

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
def measure_inference_time(model, tokenizer, prompt, num_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=num_tokens)
    end_time = time.time()
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return end_time - start_time, generated_text

def evaluate_model(model, tokenizer, prompt, model_type):
    memory_usage = get_memory_usage()
    inference_time, generated_text = measure_inference_time(model, tokenizer, prompt)
    print(f"\n[{model_type} Model]")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    print(f"Inference Time: {inference_time:.2f} seconds")
    print(f"Generated Text: {generated_text}")
    return memory_usage, inference_time, generated_text

if __name__ == "__main__":
    # Set paths and configurations
    quantized_model_path = "./quantized_model/"
    model_name = "meta-llama/Llama-3.2-1B"
    HF_TOKEN = load_token()
    prompt = "Once upon a time"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

    # Load and evaluate the full-precision model
    print("\nLoading full-precision model...")
    full_precision_model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN).to("cuda")
    full_memory, full_time, full_text = evaluate_model(full_precision_model, tokenizer, prompt, "Full-Precision")

    # Load and evaluate the quantized ONNX model
    print("\nLoading quantized model...")
    quantized_model = ORTModelForCausalLM.from_pretrained(
        quantized_model_path,
        quantization_config=AutoQuantizationConfig.arm64(is_static=False),
        provider="CUDAExecutionProvider",
        use_cache=False,
        use_io_binding=False
    )
    quantized_memory, quantized_time, quantized_text = evaluate_model(quantized_model, tokenizer, prompt, "Quantized")

    # Collect results for comparison
    results = {
        "Model": ["Full-Precision", "Quantized"],
        "Memory (MB)": [full_memory, quantized_memory],
        "Inference Time (s)": [full_time, quantized_time],
        "Generated Text": [full_text, quantized_text],
    }

    # Create a DataFrame for analysis
    df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df)

    # Save results to a markdown file
    # df.to_markdown("REPORT.md", index=False)
    # print("\nResults saved to REPORT.md")

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
    plt.savefig("performance_comparison.png")
    print("\nPerformance comparison plot saved as performance_comparison.png")
    plt.show()