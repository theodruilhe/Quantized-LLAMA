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

### Observations and Summary

1. **Accuracy:**
   - Full-precision models outperform quantized models by a small margin (~1.5%), with `Llama-3.1-8B` achieving the highest accuracy (51.20%).
   - Quantization retains most of the model's prediction capabilities.

2. **Latency:**
   - Quantized models show higher latency on GPU due to dequantization overhead (e.g., `Llama-3.2-1B` quantized: 0.1500s vs. non-quantized: 0.0500s).
   - Full-precision models benefit from GPU optimization and are faster.

3. **Memory Usage:**
   - Quantized models reduce memory usage by ~31% (e.g., `Llama-3.2-1B` uses 7000 MB compared to 10200 MB for the non-quantized version).
   - Ideal for memory-constrained environments.

4. **Scalability:**
   - Larger models improve accuracy but increase memory usage, presenting challenges for deployment in resource-constrained scenarios.

---

### Potential Use on CPU

Quantized models hold significant promise for CPU-based inference, particularly in scenarios where GPU resources are unavailable or cost-prohibitive. The reduced memory footprint of quantized models could make them highly efficient for inference on devices with limited resources, such as laptops, edge devices, or standard servers.

However, as the current implementation relies on the `bitsandbytes` library, which supports GPU-only operations, I could not test the quantized model on my Apple M1 CPU. This limitation prevented direct evaluation of the model's performance on CPU hardware.

---

### Hypothesis: Quantized Models on CPU

I suppose that quantized models would exhibit **significant latency improvements on CPUs** compared to full-precision models. The reasons for this include:

1. **Smaller Model Size:** The reduced memory footprint of quantized models would result in faster memory access and reduced data movement, which is particularly advantageous on CPUs.
2. **Integer Arithmetic:** Quantized models leverage integer arithmetic instead of floating-point computations, which is generally more efficient on CPUs that are optimized for such operations.
3. **Scalability for Edge Devices:** With lower memory requirements, quantized models could be deployed on edge devices, offering faster inference without requiring high-end hardware.

Testing this hypothesis would involve adapting the quantization process for CPU compatibility and measuring metrics like latency, memory usage, and accuracy on representative hardware. Unfortunately, due to time constraints, I was unable to explore this avenue in the current project.

Future work should focus on adapting and evaluating the quantized model for CPU environments to validate this hypothesis and unlock broader deployment possibilities.

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
