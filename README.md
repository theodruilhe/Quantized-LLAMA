
# Optimizing Large Language Model Inference

## Overview
This project explores the optimization of inference processes for Large Language Models (LLMs), focusing on LLAMA models. The primary objective is to implement and analyze **8-bit quantization techniques** to reduce computational resource usage while maintaining model performance.

Additionally, the project investigates **Flash Attention mechanisms** (optional) for further efficiency improvements.

---

## Objectives
1. **Understanding LLAMA Models**
   - Explain the architecture and functionality of transformers and causal transformers (e.g., GPT, LLAMA).
   - Analyze the training methodologies of LLAMA using the following papers:
     - [LLAMA Paper](https://arxiv.org/abs/2302.13971)
     - [LLAMA 2 Paper](https://arxiv.org/abs/2307.09288)

2. **8-bit Quantization**
   - Implement 8-bit quantization for LLAMA models.
   - Evaluate the performance of the quantized models in text generation tasks.

3. **Optional: Flash Attention**
   - Explore the Flash Attention mechanism and its role in improving LLM efficiency using resources such as:
     - [Flash Attention v1](https://arxiv.org/abs/2205.14135)
     - [Flash Attention v2](https://arxiv.org/abs/2307.08691)

4. **Documentation**
   - Provide a detailed technical report in `REPORT.md` and summarize the project in `README.md`.

---

## Implementation

### Environment Setup
1. Install the required libraries:
   ```bash
   pip install torch transformers bitsandbytes
   ```
2. Clone the Bitsandbytes repository:
   ```bash
   git clone https://github.com/timdettmers/bitsandbytes.git
   cd bitsandbytes
   ```
3. Download the pre-trained LLAMA model:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_name = "meta-llama/Llama-2-7b-hf"  # Example model
   model = AutoModelForCausalLM.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   ```

4. Test the baseline model with a simple text generation task:
   ```python
   input_text = "Once upon a time"
   inputs = tokenizer(input_text, return_tensors="pt")
   outputs = model.generate(**inputs)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

---

### 8-bit Quantization
1. Load the LLAMA model with 8-bit quantization:
   ```python
   quantized_model = AutoModelForCausalLM.from_pretrained(
       model_name, 
       load_in_8bit=True,
       device_map="auto"
   )
   ```

2. Perform inference with the quantized model:
   ```python
   inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
   outputs = quantized_model.generate(**inputs)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

---

## Results
### Performance Metrics
| Metric            | Original Model | 8-bit Quantized Model |
|--------------------|----------------|------------------------|
| Latency (ms)      | TBD            | TBD                    |
| Memory Usage (GB) | TBD            | TBD                    |
| Text Quality      | TBD            | TBD                    |

---

## Resources and References
- [LLAMA Paper](https://arxiv.org/abs/2302.13971)
- [LLAMA 2 Paper](https://arxiv.org/abs/2307.09288)
- [8-bit LLM Paper](https://arxiv.org/abs/2208.07339)
- [Bitsandbytes Repository](https://github.com/timdettmers/bitsandbytes)
- [Flash Attention v1 Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention v2 Paper](https://arxiv.org/abs/2307.08691)
- [Hugging Face Flash Attention Documentation](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention)

---

## Future Work
- Integrate **Flash Attention** for improved efficiency (if time permits).
- Test the quantized model on larger datasets and additional text generation benchmarks.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
