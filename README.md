# Optimizing Large Language Model Inference with 8-bit Quantization

## Overview

This project focuses on optimizing the inference process of Large Language Models (LLMs), particularly LLAMA, through the implementation and analysis of 8-bit quantization techniques. By exploring hardware and computational aspects of LLMs, the goal is to reduce resource usage while maintaining model performance.

## Methodology

### Tools and Resources
- **Model:** LLAMA and its derivatives.
- **Quantization:** Techniques from [Bitsandbytes GitHub Repository](https://github.com/timdettmers/bitsandbytes).
- **Datasets:** PIQA dataset for performance evaluation.
- **Evaluation Metrics:**
  - **Accuracy:** Predictive performance on the PIQA dataset.
  - **Latency:** Inference time per task.
  - **Memory Usage:** GPU memory consumption during inference.

### Implementation Steps
1. Load pre-trained LLAMA models using Hugging Face's `transformers` library.
2. Apply 8-bit quantization using `BitsAndBytesConfig` from the `bitsandbytes` library.
3. Evaluate both quantized and full-precision models on the PIQA dataset.
4. Analyze the trade-offs between performance, memory usage, and inference speed.

## Key Results

### Performance Summary
| Model          | Configuration        | Accuracy (%) | Avg Latency (s) | Avg Memory Usage (MB) |
|-----------------|----------------------|--------------|------------------|------------------------|
| Llama-3.2-1B   | Quantized Model      | 49.00        | 0.1500          | 7000.00               |
| Llama-3.2-1B   | Non-Quantized Model  | 50.50        | 0.0500          | 10200.00              |
| Llama-3.2-3B   | Quantized Model      | 50.00        | 0.1440          | 7500.00               |
| Llama-3.2-3B   | Non-Quantized Model  | 51.00        | 0.0459          | 10500.00              |
| Llama-3.1-8B   | Quantized Model      | 50.50        | 0.1300          | 7600.00               |
| Llama-3.1-8B   | Non-Quantized Model  | 51.20        | 0.0475          | 10800.00              |

### Key Observations
- Quantized models reduce memory usage significantly (e.g., up to 30% reduction) while maintaining comparable accuracy to full-precision models.
- Full-precision models exhibit lower latency, making them more suitable for real-time applications.

## Deliverables

### 1. Codebase
- Python scripts for 8-bit quantization and performance evaluation.

### 2. Documentation
- `README.md` (this file): Highlights objectives, methodology, and results.
- `REPORT.md`: Detailed technical explanation (uploaded separately).

### 3. Results
- Performance metrics and figures saved in the `results/` and `figures/` directories.

## How to Run

1. Clone the repository and install the required dependencies.
2. Prepare the virtual machine with suitable specifications (e.g., NVIDIA A40 GPU, 50 GB RAM).
3. Run the evaluation script:
   ```bash
   python evaluate_model.py
   ```
4. Review the generated results and figures.

## References

- [LLAMA Paper](https://arxiv.org/abs/2302.13971)
- [LLAMA 2 Paper](https://arxiv.org/abs/2307.09288)
- [8-bit LLM Paper](https://arxiv.org/abs/2208.07339)
- [Bitsandbytes GitHub Repository](https://github.com/timdettmers/bitsandbytes)