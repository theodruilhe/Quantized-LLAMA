import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from quantize_model2 import load_full_model, load_quantized_model
from load_token import load_token
import logging
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

logging.getLogger("transformers").setLevel(logging.ERROR)
HF_TOKEN = load_token()


def evaluate_model_on_piqa(model, tokenizer, device, description,  max_examples = 100):
    """
    Evaluates a model on the PIQA dataset for accuracy, latency, and memory usage.
    """
    correct_predictions = 0
    total_predictions = 0
    latencies = []
    memory_usages = []

    print(f"Evaluating {description} on PIQA dataset...")
    # Load the PIQA validation dataset
    dataset = load_dataset("piqa", split="validation", trust_remote_code=True)

    # Initialize the progress bar
    with tqdm(total=max_examples, desc="Processing examples", unit="example") as progress_bar:
        for i, example in enumerate(dataset):
            if i >= max_examples:
                break  # Stop after processing max_examples

            # Prepare the question and answers
            question = example["goal"]
            choice1 = example["sol1"]
            choice2 = example["sol2"]

            # Create inputs for the model
            input1 = tokenizer(f"Question: {question} Answer: {choice1}", return_tensors="pt").to(device)
            input2 = tokenizer(f"Question: {question} Answer: {choice2}", return_tensors="pt").to(device)

            # Measure latency and memory usage
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            with torch.no_grad():
                output1 = model(**input1)
                output2 = model(**input2)
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)

            memory_usage = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
            memory_usages.append(memory_usage)

            # Use the sum of logits as the score for each choice
            score1 = output1.logits.sum().item()
            score2 = output2.logits.sum().item()

            # Determine the model's predicted answer
            predicted_choice = 1 if score1 > score2 else 2
            correct_choice = 1 if example["label"] == 0 else 2

            if predicted_choice == correct_choice:
                correct_predictions += 1

            total_predictions += 1

            # Update the progress bar
            progress_bar.update(1)

    accuracy = correct_predictions / total_predictions
    avg_latency = sum(latencies) / len(latencies)
    avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else None

    print(f"\n=== {description} ===")
    print(f"Accuracy on PIQA: {accuracy:.4f}")
    print(f"Average Latency: {avg_latency:.4f} seconds")
    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB")

    return {
        "description": description,
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "avg_memory_usage": avg_memory_usage,
    }


def compare_models(results):
    """
    Compare models and plot results.
    """
    # Create a DataFrame for results
    df = pd.DataFrame(results)

    # Print summary table
    print("\n=== Performance Comparison ===")
    print(df)

    # Plot accuracy
    plt.figure()
    plt.bar(df["description"], df["accuracy"])
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.show()

    # Plot latency
    plt.figure()
    plt.bar(df["description"], df["avg_latency"])
    plt.title("Latency Comparison")
    plt.ylabel("Latency (seconds)")
    plt.xlabel("Model")
    plt.show()

    # Plot memory usage
    plt.figure()
    plt.bar(df["description"], df["avg_memory_usage"])
    plt.title("Memory Usage Comparison")
    plt.ylabel("Memory Usage (MB)")
    plt.xlabel("Model")
    plt.show()


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-70B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    full_model = load_full_model(model_name, HF_TOKEN)
    quantized_model = load_quantized_model(model_name, HF_TOKEN)

    # Evaluate models on PIQA
    full_results = evaluate_model_on_piqa(full_model, tokenizer, device, "Non-Quantized Model")
    quantized_results = evaluate_model_on_piqa(quantized_model, tokenizer, device, "Quantized Model")

    # Compare results
    compare_models([quantized_results, full_results])