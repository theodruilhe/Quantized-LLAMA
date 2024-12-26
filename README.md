# Optimizing Large Language Model Inference

## Overview

This project explores the optimization of inference processes for Large Language Models (LLMs), focusing on LLAMA models. The primary objective is to implement and analyze **8-bit quantization techniques** to reduce computational resource usage while maintaining model performance.

Additionally, the project investigates **Flash Attention mechanisms** (optional) for further efficiency improvements.

------------------------------------------------------------------------

## Objectives

1.  **Understanding LLAMA Models**
    -   Explain the architecture and functionality of transformers and causal transformers (e.g., GPT, LLAMA).
    -   Analyze the training methodologies of LLAMA using the following papers:
        -   [LLAMA Paper](https://arxiv.org/abs/2302.13971)
        -   [LLAMA 2 Paper](https://arxiv.org/abs/2307.09288)
2.  **8-bit Quantization**
    -   Implement 8-bit quantization for LLAMA models.
    -   Evaluate the performance of the quantized models in text generation tasks.
3.  **Optional: Flash Attention**
    -   Explore the Flash Attention mechanism and its role in improving LLM efficiency using resources such as:
        -   [Flash Attention v1](https://arxiv.org/abs/2205.14135)
        -   [Flash Attention v2](https://arxiv.org/abs/2307.08691)
4.  **Documentation**
    -   Provide a detailed technical report in `REPORT.md` and summarize the project in `README.md`.

## Transformers: The Foundation of Modern NLP

### Overview

Transformers are a deep learning architecture introduced in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). They have become foundational in modern Natural Language Processing (NLP) due to their efficiency in processing sequential data and their capability to capture long-range dependencies in text.

**What is a Transformer?**

A Transformer is a model architecture that relies on a mechanism called **self-attention** to compute representations of its input and output without using sequence-aligned RNNs or convolution. This design allows for greater parallelization and efficiency in training and inference.

#### Key Features:

- **Self-Attention Mechanism**: Enables the model to weigh the importance of different words in a sentence when encoding a particular word, allowing it to capture dependencies regardless of their distance in the sequence.

- **Positional Encoding**: Since Transformers do not inherently process data in sequence, positional encodings are added to input embeddings to provide information about the position of a word in a sentence.

- **Encoder-Decoder Structure**: The architecture is typically divided into an encoder that processes the input and a decoder that generates the output, both utilizing layers of self-attention and feed-forward neural networks.


### Visualization of Transformer Architecture

The diagram below provides a detailed visualization of the Transformer architecture as described in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). It illustrates the two main components of the Transformer model: the encoder and the decoder.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*BHzGVskWGS_3jEcYYi6miQ.png" alt="Transformer Architecture" width="600" />

#### Description:
1. **Encoder (Left):**
   - The encoder consists of multiple identical layers (denoted as \( N \)).
   - Each layer contains two main sub-layers:
     - **Multi-Head Attention:** Allows the model to focus on different parts of the input sequence.
     - **Feed Forward:** A position-wise fully connected network that processes each token independently.
   - Positional encodings are added to the input embeddings to provide sequential order information.

2. **Decoder (Right):**
   - The decoder also consists of multiple identical layers (denoted as \( N \)).
   - Each layer has three sub-layers:
     - **Masked Multi-Head Attention:** Prevents the decoder from "seeing" future tokens during training.
     - **Multi-Head Attention:** Attends to the encoder’s outputs to learn contextual relationships.
     - **Feed Forward:** A fully connected network applied position-wise.
   - Positional encodings are added to output embeddings to provide sequence order information for the output.

3. **Final Output:**
   - The decoder generates the output probabilities for the next token using a linear layer followed by a softmax function.



### Key Components of Transformers:

#### **1. Self-Attention Mechanism**

-   The self-attention mechanism is the core of the transformer architecture. It allows the model to dynamically focus on relevant parts of the input sequence when generating predictions.
-   For each token in the input, the self-attention mechanism calculates a weighted sum of all other tokens in the sequence, determining their importance for the current token.

**Mathematical Formulation:** For a given input sequence, the self-attention mechanism computes the output as: $$ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Where

-   $Q$ (Query), $K$ (Key), and $V$ (Value) are matrices derived from the input embeddings:
$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$ Here, $W^Q, W^K, W^V$ are learnable weight matrices.

-   $d_k$: Dimensionality of the key vectors, used to scale the dot product for numerical stability.

-   The softmax function ensures that attention weights sum to 1 across the sequence.

This mechanism allows the model to assign attention weights dynamically, learning which tokens are most relevant to each other.

------------------------------------------------------------------------

#### **2. Multi-Head Attention**

-   Instead of performing a single self-attention calculation, the transformer uses **multi-head attention**, where attention is computed in parallel across multiple "heads."
-   Each head learns to focus on different parts of the input sequence, capturing diverse relationships and dependencies.

**Formula for Multi-Head Attention:** $$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W^O
$$ Where each attention head is computed as: $$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

-   $W_i^Q, W_i^K, W_i^V$: Learnable projection matrices for the $i$-th head.

-    $W^O$: Output projection matrix.

By concatenating and projecting the results, the model integrates diverse attention patterns effectively.

------------------------------------------------------------------------

#### **3. Positional Encoding**

-   Transformers lack inherent sequential order due to their fully parallelized processing. To inject order into the input, positional encodings are added to the input embeddings.
-   These encodings are designed to help the model distinguish the position of each token in the sequence.

**Formula for Positional Encoding:** 
$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$ 

Where:
- $pos$: Position index of the token in the sequence.
- $i$: Dimension index of the embedding.
- $d$: Dimensionality of the embedding.

The use of sin and cos functions at different frequencies ensures unique encodings for each position, allowing the model to differentiate positions effectively.

------------------------------------------------------------------------

#### **4. Feedforward Layers**

-   After the self-attention mechanism, the transformer applies a position-wise feedforward network (FFN) to add non-linearity and model complex patterns.
-   Each FFN consists of two linear transformations with a ReLU activation: 
    $$\text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2$$ 

Where:
-   $W_1, W_2$: Weight matrices.
-   $b_1, b_2$: Bias terms.

This FFN is applied independently to each position in the sequence.

------------------------------------------------------------------------

#### **5. Layer Normalization and Residual Connections**

-   **Residual Connections:** Add the input of a layer directly to its output to improve gradient flow: $$
    \text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))
    $$
-   **Layer Normalization:** Stabilizes training by normalizing the outputs of each sublayer (e.g., attention, FFN) across the feature dimension.

These components help mitigate the issues of vanishing/exploding gradients and improve convergence.

------------------------------------------------------------------------

#### **6. Encoder-Decoder Architecture**

-   Transformers are often structured into two main components:
    -   **Encoder:**
        -   Processes the input sequence into a set of contextual embeddings.
        -   Consists of multiple identical layers, each with a multi-head self-attention mechanism and a feedforward network.
    -   **Decoder:**
        -   Generates the output sequence one token at a time.
        -   Consists of:
            -   Multi-head self-attention (with a causal mask to prevent access to future tokens).
            -   Cross-attention to link input (encoder outputs) and output tokens.

**Workflow:** 

1. The **encoder** processes the input sequence. 
2. The **decoder** uses the encoder’s output and previously generated tokens to produce the next token.

------------------------------------------------------------------------

## Causal Transformers

Causal transformers, such as GPT (Generative Pre-trained Transformer) and LLAMA (Large Language Model Meta AI), are specialized transformer architectures designed to handle autoregressive tasks, such as text generation, story completion, and code synthesis. Unlike standard bidirectional transformers (e.g., BERT), causal transformers rely on unidirectional processing, ensuring that the model predicts tokens sequentially while respecting the order of the input.

**What Makes a Transformer "Causal"?**

In causal transformers, the self-attention mechanism is modified to enforce a strict left-to-right, sequential processing order. This prevents the model from accessing "future" tokens when predicting the next word, a requirement for autoregressive tasks where outputs depend only on past context.


### Key Characteristics of Causal Transformers:

#### **1. Autoregressive Modeling**
- **Autoregressive models** generate text sequentially, one token at a time. The model predicts the next token \( x_t \) based on all previously generated tokens \( x_1, x_2, \dots, x_{t-1} \). This process is fundamental for tasks like text generation, where output coherence depends on maintaining sequential dependencies.
  
- The conditional probability of generating a sequence \( x = [x_1, x_2, \dots, x_T] \) is factorized as:
  \[
  P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t | x_1, x_2, \dots, x_{t-1})
  \]
  This factorization ensures that the prediction of each token is explicitly conditioned on all preceding tokens, enabling the model to generate contextually relevant outputs.

- **Core Idea:** The model attends only to previous tokens (i.e., left-to-right attention), preventing it from accessing future information during both training and inference. This constraint is enforced using a **causal attention mask**.

##### **Causal Attention Mask**
- The causal attention mask is a binary matrix used in the self-attention mechanism to mask out future tokens during training and inference. 
- It ensures that token \( x_t \) only attends to tokens \( x_1 \) through \( x_t \), blocking access to \( x_{t+1}, x_{t+2}, \dots \) in the sequence.
  
- The attention mask is implemented as an **upper triangular matrix** with values:
  - `1` (allows attention) for tokens at or before the current position \( t \).
  - `0` (blocks attention) for tokens after \( t \).

**Mathematical Representation:**
For a sequence of 4 tokens, the causal attention mask \( M \) is:
\[
M =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{bmatrix}
\]

**Intuition:**
- At position \( t = 1 \), the first token \( x_1 \) can only attend to itself.
- At position \( t = 2 \), the second token \( x_2 \) can attend to \( x_1 \) and \( x_2 \).
- At position \( t = 3 \), the third token \( x_3 \) can attend to \( x_1, x_2, \) and \( x_3 \).
- This process continues sequentially until the end of the sequence.


##### **Mask Application in Attention Calculation**
The causal attention mask is applied directly to the scaled dot-product attention scores during the self-attention calculation:

1. **Self-Attention Scores**:
   For queries \( Q \), keys \( K \), and values \( V \), the attention weights are computed as:
   \[
   A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M \right)
   \]
   Here:
   - \( \frac{QK^T}{\sqrt{d_k}} \) are the raw dot-product attention scores scaled by \( d_k \), the dimensionality of the keys.
   - \( M \) is the causal attention mask (with `-∞` values for masked positions, which effectively sets the corresponding attention weights to 0 after the softmax).

2. **Resulting Attention**:
   The attention weights ensure that each token only attends to tokens up to its position in the sequence. For example:
   - Token \( x_1 \) attends only to itself.
   - Token \( x_2 \) attends to \( x_1 \) and \( x_2 \), and so on.

3. **Final Weighted Sum**:
   The attention scores are multiplied with \( V \) (the values) to compute the output for each position:
   \[
   \text{Attention Output} = A \cdot V
   \]


#### **2. Pre-training and Fine-tuning**

Causal transformers undergo two distinct stages of development: **pre-training** and **fine-tuning**. These stages leverage large datasets and task-specific data to make the models general-purpose and highly adaptable to various applications.

---

##### **Pre-training**
- **Objective**: Causal transformers are initially trained on massive, unlabelled datasets (e.g., books, web data, or research papers) using an **autoregressive language modeling objective**. The goal is to predict the next token \( x_t \) given the previous tokens \( x_1, x_2, \dots, x_{t-1} \).

- **Learning Process**:
  - The model maximizes the likelihood of the correct token at each position:
    \[
    P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t | x_1, x_2, \dots, x_{t-1})
    \]
  - Training involves minimizing the **cross-entropy loss** between the predicted probabilities and the ground truth:
    \[
    \mathcal{L}_{\text{pre-train}} = - \sum_{t=1}^T \log P(x_t | x_1, x_2, \dots, x_{t-1})
    \]
    Here, \( T \) is the total number of tokens in the sequence.

- **What the Model Learns**:
  - **Syntax and Grammar**: The structure of language (e.g., sentence formation, punctuation).
  - **Semantics**: Meaning and relationships between words, phrases, and contexts.
  - **World Knowledge**: Patterns and associations from the large-scale corpus (e.g., common facts, cultural references).

- **Data Considerations**:
  - Large, diverse, and high-quality datasets ensure that the model generalizes well across domains.
  - Examples of pre-training datasets include OpenWebText, The Pile, and CCNet.

##### **Fine-tuning**
- **Objective**: After pre-training, causal transformers are adapted to specific tasks using labeled or curated datasets. Fine-tuning aligns the model's general knowledge with task-specific requirements, such as summarization, dialogue generation, or code completion.

- **Techniques**:
  1. **Supervised Fine-tuning**:
     - The model is trained on a labeled dataset where input-output pairs are explicitly provided (e.g., questions paired with answers).
     - Example: Fine-tuning a GPT model for a customer support chatbot by training it on a dataset of questions and corresponding responses.

  2. **Reinforcement Learning with Human Feedback (RLHF)**:
     - RLHF is often used in fine-tuning tasks where human preferences are involved (e.g., generating text that is factual and polite).
     - The process involves:
       - Collecting human feedback on generated outputs.
       - Training a reward model to predict human preferences.
       - Fine-tuning the transformer using reinforcement learning to optimize the reward signal.
     - Example: GPT models use RLHF to align outputs with user preferences and avoid generating harmful or biased text.


#### **3. Efficiency Through Decoding**

During inference, causal transformers generate text **autoregressively**, meaning that tokens are produced one at a time based on the preceding context. The decoding process can be optimized using various strategies to balance **efficiency**, **quality**, and **diversity**.

##### **Decoding Strategies**
1. **Greedy Decoding**:
   - The simplest decoding method.
   - At each step, the model selects the token with the highest probability:
     \[
     x_t = \arg \max P(x_t | x_1, x_2, \dots, x_{t-1})
     \]
   - Advantage: Computationally efficient and deterministic.
   - Limitation: May produce repetitive or suboptimal results because it does not explore alternative sequences.

2. **Beam Search**:
   - Explores multiple candidate sequences (beams) simultaneously to find the most likely overall sequence.
   - At each step, it keeps the top \( k \) beams with the highest cumulative probabilities.
   - Advantage: Produces higher-quality outputs compared to greedy decoding.
   - Limitation: Computationally expensive and may still favor generic sequences.

3. **Sampling Methods**:
   - Introduces randomness to encourage diverse and creative outputs:
     - **Top-k Sampling**: Samples from the \( k \) most probable tokens at each step, reducing low-probability noise.
     - **Nucleus Sampling (Top-p Sampling)**: Samples from the smallest set of tokens whose cumulative probability exceeds a threshold \( p \), dynamically adjusting the candidate pool size.
   - Advantage: Balances quality and diversity, making it ideal for tasks like creative writing.
   - Limitation: May occasionally generate incoherent outputs if randomness is too high.

---

### Advantages of Causal Transformers:

1. **Sequential Text Generation**:
   - Ensures coherent and contextually accurate outputs for tasks like text generation and dialogue.

2. **Adaptability Across Domains**:
   - Generalizes well across diverse domains with minimal task-specific fine-tuning.

3. **Simplicity in Objective**:
   - Predicting the next token is straightforward, making pre-training scalable and efficient.

4. **Rich Contextual Understanding**:
   - Builds detailed context by attending to all preceding tokens in a sequence.

5. **Optimizable**:
   - Compatible with techniques like quantization and caching for improved efficiency.

---

### Limitations of Causal Transformers:

1. **Unidirectional Context**:
   - Can only use past tokens, limiting tasks requiring bidirectional understanding.

2. **Sequential Inference**:
   - Token-by-token generation slows inference for long outputs.

3. **Resource Intensive**:
   - Demands significant computational resources for training and fine-tuning.

4. **Repetition Issues**:
   - May generate repetitive sequences without advanced decoding strategies.

5. **Long-Term Dependencies**:
   - Struggles with very long contexts despite self-attention optimizations.


------------------------------------------------------------------------

## LLAMA: Large Language Model Meta AI

LLAMA (Large Language Model Meta AI) is a family of causal transformers developed by Meta, designed to optimize performance and efficiency in natural language understanding and generation tasks.

### Key Features of LLAMA:

1.  **Lightweight Design:**
    -   LLAMA is built to be computationally efficient, reducing resource requirements while maintaining high performance.
    -   This makes it accessible for research and smaller-scale deployments.
2.  **Training on Diverse Data:**
    -   LLAMA models are trained on a mix of publicly available datasets to generalize well across a variety of tasks.
3.  **Variants of LLAMA:**
    -   LLAMA comes in different sizes (e.g., LLAMA-7B, LLAMA-13B) to cater to various hardware and task requirements.
4.  **Advanced Optimizations:**
    -   Techniques like rotary positional embeddings and tokenization optimizations make LLAMA models faster and more robust.

### LLAMA vs. Other Transformers:

| Feature          | LLAMA              | GPT (OpenAI)       | BERT                |
|-------------------|------------------|------------------|------------------|
| Architecture     | Causal Transformer | Causal Transformer | Encoder Transformer |
| Usage            | Text Generation    | Text Generation    | Classification, QA  |
| Efficiency Focus | High               | Moderate           | Moderate            |

### LLAMA Applications:

-   Chatbots
-   Summarization
-   Question Answering
-   Creative Writing

### LLAMA Papers:

-  [LLAMA Paper](https://arxiv.org/abs/2302.13971)
- [LLAMA 2 Paper](https://arxiv.org/abs/2307.09288)

## Implementation

### Environment Setup

1.  Install the required libraries:

    ``` bash
    pip install torch transformers bitsandbytes
    ```

2.  Clone the Bitsandbytes repository:

    ``` bash
    git clone https://github.com/timdettmers/bitsandbytes.git
    cd bitsandbytes
    ```

3.  Download the pre-trained LLAMA model:

    ``` python
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-2-7b-hf"  # Example model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ```

4.  Test the baseline model with a simple text generation task:

    ``` python
    input_text = "Once upon a time"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    ```

------------------------------------------------------------------------

### 8-bit Quantization

1.  Load the LLAMA model with 8-bit quantization:

    ``` python
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        load_in_8bit=True,
        device_map="auto"
    )
    ```

2.  Perform inference with the quantized model:

    ``` python
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = quantized_model.generate(**inputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    ```

------------------------------------------------------------------------

## Results

### Performance Metrics

| Metric            | Original Model | 8-bit Quantized Model |
|-------------------|----------------|-----------------------|
| Latency (ms)      | TBD            | TBD                   |
| Memory Usage (GB) | TBD            | TBD                   |
| Text Quality      | TBD            | TBD                   |

------------------------------------------------------------------------

## Resources and References

-   [LLAMA Paper](https://arxiv.org/abs/2302.13971)
-   [LLAMA 2 Paper](https://arxiv.org/abs/2307.09288)
-   [8-bit LLM Paper](https://arxiv.org/abs/2208.07339)
-   [Bitsandbytes Repository](https://github.com/timdettmers/bitsandbytes)
-   [Flash Attention v1 Paper](https://arxiv.org/abs/2205.14135)
-   [Flash Attention v2 Paper](https://arxiv.org/abs/2307.08691)
-   [Hugging Face Flash Attention Documentation](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention)

------------------------------------------------------------------------

## Future Work

-   Integrate **Flash Attention** for improved efficiency (if time permits).
-   Test the quantized model on larger datasets and additional text generation benchmarks.

------------------------------------------------------------------------

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
