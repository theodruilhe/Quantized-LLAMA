import os
import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd


def evaluate_model(model, tokenizer, prompt, max_length=50, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start_time = time.time()
    outputs = model.generate(**inputs, max_length=max_length)
    latency = time.time() - start_time

    memory_info = psutil.Process(os.getpid()).memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # Convert to MB

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text, latency, memory_usage


def load_model_and_tokenizer(model_path, device="cuda"):
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def main():
    model_paths = {
        "original": "./original_model",
        "quantized": "./quantized_model"
    }
    prompt = "Once upon a time, in a distant galaxy,"
    max_length = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    for model_type, model_path in model_paths.items():
        model, tokenizer = load_model_and_tokenizer(model_path, device)
        generated_text, latency, memory_usage = evaluate_model(
            model, tokenizer, prompt, max_length, device
        )
        results.append({
            "Model Type": model_type,
            "Generated Text": generated_text,
            "Latency (s)": latency,
            "Memory Usage (MB)": memory_usage
        })

    # Save results to a DataFrame
    df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df)

    # Visualization
    df.set_index("Model Type", inplace=True)
    df[["Latency (s)", "Memory Usage (MB)"]].plot(kind="bar", figsize=(10, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Values")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()