import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import pandas as pd

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

# Function to measure inference time and quality
def evaluate_model(model, tokenizer, prompts, num_tokens=50, device="cuda"):
    results = []
    model.to(device)
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=num_tokens)
        end_time = time.time()
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference_time = end_time - start_time
        memory_usage = get_memory_usage()
        results.append((prompt, generated_text, inference_time, memory_usage))
    return results

# Evaluate quality metrics
def evaluate_quality(generated, references):
    bleu_scores = [sentence_bleu([ref.split()], gen.split()) for gen, ref in zip(generated, references)]
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [rouge.score(ref, gen) for gen, ref in zip(generated, references)]
    return bleu_scores, rouge_scores

if __name__ == "__main__":
    # Load dataset and prompts
    dataset = load_dataset("EleutherAI/the_pile", split="validation[:100]")
    prompts = [example["text"][:100] for example in dataset if "text" in example]  # Use first 100 chars as prompt

    # Load models
    models = {
        "Original": "meta-llama/Llama-3.1-8B",
        "Quantized": "./quantized_model"
    }
    HF_TOKEN = load_token()

    results = []
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", token=HF_TOKEN)
    for model_name, model_path in models.items():
        print(f"\nEvaluating {model_name}...")
        if model_name == "Quantized":
            model = torch.jit.load(model_path).to("cuda")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, token=HF_TOKEN).to("cuda")

        # Evaluate
        model_results = evaluate_model(model, tokenizer, prompts[:10], device="cuda")
        for prompt, generated, time, memory in model_results:
            results.append({
                "Model": model_name,
                "Prompt": prompt,
                "Generated Text": generated,
                "Inference Time (s)": time,
                "Memory Usage (MB)": memory
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("\nEvaluation Results saved to evaluation_results.csv")

    # Analyze quality metrics
    generated_texts = [res["Generated Text"] for res in results]
    references = [example["text"] for example in dataset][:len(generated_texts)]
    bleu_scores, rouge_scores = evaluate_quality(generated_texts, references)

    # Summarize quality
    quality_df = pd.DataFrame({
        "Model": [res["Model"] for res in results],
        "BLEU Score": bleu_scores,
        "ROUGE-1": [score["rouge1"].fmeasure for score in rouge_scores],
        "ROUGE-2": [score["rouge2"].fmeasure for score in rouge_scores],
        "ROUGE-L": [score["rougeL"].fmeasure for score in rouge_scores],
    })
    quality_df.to_csv("quality_metrics.csv", index=False)
    print("\nQuality Metrics saved to quality_metrics.csv")