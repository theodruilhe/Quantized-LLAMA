# Optimizing Large Language Model Inference

## Overview

This project explores the optimization of inference processes for Large Language Models (LLMs), focusing on LLAMA models. The objective is to implement and analyze **8-bit quantization techniques** to reduce computational resource usage while maintaining model performance.

------------------------------------------------------------------------

## Transformers: The Foundation of Modern NLP

### Overview

Transformers are a deep learning architecture introduced in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). They have become foundational in modern Natural Language Processing (NLP) due to their efficiency in processing sequential data and their capability to capture long-range dependencies in text.

**What is a Transformer?**

A Transformer is a model architecture that relies on a mechanism called **self-attention** to compute representations of its input and output without using sequence-aligned RNNs or convolution. This design allows for greater parallelization and efficiency in training and inference.

#### Key Features:

-   **Self-Attention Mechanism**: Enables the model to weigh the importance of different words in a sentence when encoding a particular word, allowing it to capture dependencies regardless of their distance in the sequence.

-   **Positional Encoding**: Since Transformers do not inherently process data in sequence, positional encodings are added to input embeddings to provide information about the position of a word in a sentence.

-   **Encoder-Decoder Structure**: The architecture is typically divided into an encoder that processes the input and a decoder that generates the output, both utilizing layers of self-attention and feed-forward neural networks.

### Visualization of Transformer Architecture

The diagram below provides a detailed visualization of the Transformer architecture as described in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). It illustrates the two main components of the Transformer model: the encoder and the decoder.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*BHzGVskWGS_3jEcYYi6miQ.png" alt="Transformer Architecture" width="600"/>

#### Description:

1.  **Encoder (Left):**
    -   The encoder consists of multiple identical layers (denoted as $N$).
    -   Each layer contains two main sub-layers:
        -   **Multi-Head Attention:** Allows the model to focus on different parts of the input sequence.
        -   **Feed Forward:** A position-wise fully connected network that processes each token independently.
    -   Positional encodings are added to the input embeddings to provide sequential order information.
2.  **Decoder (Right):**
    -   The decoder also consists of multiple identical layers (denoted as $N$).
    -   Each layer has three sub-layers:
        -   **Masked Multi-Head Attention:** Prevents the decoder from "seeing" future tokens during training.
        -   **Multi-Head Attention:** Attends to the encoder’s outputs to learn contextual relationships.
        -   **Feed Forward:** A fully connected network applied position-wise.
    -   Positional encodings are added to output embeddings to provide sequence order information for the output.
3.  **Final Output:**
    -   The decoder generates the output probabilities for the next token using a linear layer followed by a softmax function.

### Key Components of Transformers:

#### **1. Self-Attention Mechanism**

-   The self-attention mechanism is the core of the transformer architecture. It allows the model to dynamically focus on relevant parts of the input sequence when generating predictions.
-   For each token in the input, the self-attention mechanism calculates a weighted sum of all other tokens in the sequence, determining their importance for the current token.

**Mathematical Formulation:** For a given input sequence, the self-attention mechanism computes the output as: 
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where

-   $Q$(Query), $K$(Key), and $V$(Value) are matrices derived from the input embeddings: 
    $$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

    Here, $W^Q$, $W^K$, $W^V$ are learnable weight matrices.

-   $d_k$: Dimensionality of the key vectors, used to scale the dot product for numerical stability.

-   The softmax function ensures that attention weights sum to 1 across the sequence.

This mechanism allows the model to assign attention weights dynamically, learning which tokens are most relevant to each other.

------------------------------------------------------------------------

#### **2. Multi-Head Attention**

Instead of performing a single self-attention calculation, the transformer uses **multi-head attention**, where attention is computed in parallel across multiple "heads." Each head learns to focus on different parts of the input sequence, capturing diverse relationships and dependencies.

**Formula for Multi-Head Attention:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W^O$$

Where each attention head is computed as: $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

-   $W_i^$, $W_i^K$, $W_i^V$: Learnable projection matrices for the $i$-th head.

-   $W^O$: Output projection matrix.

By concatenating and projecting the results, the model integrates diverse attention patterns effectively.

------------------------------------------------------------------------

#### **3. Positional Encoding**

Transformers lack inherent sequential order due to their fully parallelized processing. To inject order into the input, positional encodings are added to the input embeddings. These encodings are designed to help the model distinguish the position of each token in the sequence.

**Formula for Positional Encoding:** 
$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Where:

-   $pos$: Position index of the token in the sequence.

-   $i$: Dimension index of the embedding.

-   $d$: Dimensionality of the embedding.

The use of sin and cos functions at different frequencies ensures unique encodings for each position, allowing the model to differentiate positions effectively.

------------------------------------------------------------------------

#### **4. Feedforward Layers**

After the self-attention mechanism, the transformer applies a position-wise feedforward network (FFN) to add non-linearity and model complex patterns.Each FFN consists of two linear transformations with a ReLU activation: $$\text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2$$

Where:

-   $W_1$, $W_2$: Weight matrices.

-   $b_1$, $b_2$: Bias terms.

This FFN is applied independently to each position in the sequence.

------------------------------------------------------------------------

#### **5. Layer Normalization and Residual Connections**

**Residual Connections:** Add the input of a layer directly to its output to improve gradient flow: 
$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Layer Normalization:** Stabilizes training by normalizing the outputs of each sublayer (e.g., attention, FFN) across the feature dimension.

These components help mitigate the issues of vanishing/exploding gradients and improve convergence.

------------------------------------------------------------------------

#### **6. Encoder-Decoder Architecture**

Transformers are often structured into two main components:

-   **Encoder:** Processes the input sequence into a set of contextual embeddings. Consists of multiple identical layers, each with a multi-head self-attention mechanism and a feedforward network.
-   **Decoder:** Generates the output sequence one token at a time. Consists of:
    -   Multi-head self-attention (with a causal mask to prevent access to future tokens).
    -   Cross-attention to link input (encoder outputs) and output tokens.

**Workflow:**

1.  The **encoder** processes the input sequence.
2.  The **decoder** uses the encoder’s output and previously generated tokens to produce the next token.

------------------------------------------------------------------------

## Causal Transformers

Causal transformers, such as GPT (Generative Pre-trained Transformer) and LLAMA (Large Language Model Meta AI), are specialized transformer architectures designed to handle autoregressive tasks, such as text generation. Unlike standard bidirectional transformers (e.g., BERT), causal transformers rely on unidirectional processing, ensuring that the model predicts tokens sequentially while respecting the order of the input.

**What Makes a Transformer "Causal"?**

In causal transformers, the self-attention mechanism is modified to enforce a strict left-to-right, sequential processing order. This prevents the model from accessing "future" tokens when predicting the next word, a requirement for autoregressive tasks where outputs depend only on past context.

### Key Characteristics of Causal Transformers:

------------------------------------------------------------------------

#### **1. Autoregressive Modeling**

**Autoregressive models** generate text sequentially, one token at a time. The model predicts the next token $x_t$based on all previously generated tokens $x_1$ , $x_2$ , $\dots$ , $x_{t-1}$ . This process is fundamental for tasks like text generation, where output coherence depends on maintaining sequential dependencies.

The conditional probability of generating a sequence $$x = [x_1, x_2, \dots, x_T]$$ is factorized as:

$$
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t | x_1, x_2, \dots, x_{t-1})
$$

**Core Idea:** The model attends only to previous tokens (i.e., left-to-right attention), preventing it from accessing future information during both training and inference. This constraint is enforced using a **causal attention mask**.

**Causal Attention Mask**

The causal attention mask is a binary matrix used in the self-attention mechanism to mask out future tokens during training and inference. It ensures that token $x_t$only attends to tokens $x_1$through $x_t$, blocking access to $x_{t+1}$ , $x_{t+2}$ ,  $...$ in the sequence. The attention mask is implemented as an **upper triangular matrix** with values:

-   `1` (allows attention) for tokens at or before the current position $t$ .
-   `0` (blocks attention) for tokens after $t$ .

**Mathematical Representation:** For a sequence of 4 tokens, the causal attention mask $M$is:

$$
M =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{bmatrix}
$$

**Intuition:**

-   At position $t = 1$, the first token $x_1$ can only attend to itself.

-   At position $t = 2$, the second token $x_2$ can attend to $x_1$and $x_2$.

-   At position $t = 3$, the third token $x_3$ can attend to $x_1$, $x_2$, and $x_3$.

This process continues sequentially until the end of the sequence.

**Mask Application in Attention Calculation**

The causal attention mask is applied directly to the scaled dot-product attention scores during the self-attention calculation:

1.  **Self-Attention Scores**: For queries $Q$, keys $K$, and values $V$, the attention weights are computed as:

    $$A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M \right)$$

    Here:

    -   $\frac{QK^T}{\sqrt{d_k}}$ are the raw dot-product attention scores scaled by $d_k$ , the dimensionality of the keys.
    -   $M$ is the causal attention mask (with $-∞$ values for masked positions, which effectively sets the corresponding attention weights to 0 after the softmax).

2.  **Resulting Attention**: The attention weights ensure that each token only attends to tokens up to its position in the sequence. For example:

    -   Token $x_1$ attends only to itself.
    -   Token $x_2$ attends to $x_1$ and $x_2$, and so on.

3.  **Final Weighted Sum**: The attention scores are multiplied with $V$(the values) to compute the output for each position: 
$$\text{Attention Output} = A \cdot V$$

------------------------------------------------------------------------

#### **2. Pre-training and Fine-tuning**

Causal transformers undergo two distinct stages of development: **pre-training** and **fine-tuning**. These stages leverage large datasets and task-specific data to make the models general-purpose and highly adaptable to various applications.

**Pre-training**

**Objective** Causal transformers are initially trained on massive, unlabelled datasets (e.g., books, web data, or research papers) using an **autoregressive language modeling objective**. The goal is to predict the next token $x_t$ given the previous tokens $x_1$ , $x_2$ , $\dots$, x_{t-1}$.

**Learning Process**:

-   The model maximizes the likelihood of the correct token at each position: 
    $$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t | x_1, x_2, \dots, x_{t-1})$$

-   Training involves minimizing the cross-entropy loss between the predicted probabilities and the ground truth:
$$\mathcal{L}_{\text{pre-train}} = - \sum_{t=1}^T \log P(x_t | x_1, x_2, \dots, x_{t-1})$$ 

Here, $T$ is the total number of tokens in the sequence.

**What the Model Learns**

1.  **Syntax and Grammar**: The structure of language (e.g., sentence formation, punctuation).
2.  **Semantics**: The meaning and relationships between words, phrases, and contexts.
3.  **World Knowledge**: Patterns and associations derived from the large-scale corpus (e.g., common facts, cultural references).

**Data Considerations** Large, diverse, and high-quality datasets are crucial to ensure that the model generalizes well across domains. Examples of pre-training datasets:

-   **OpenWebText**: A curated dataset of web text.

-   **The Pile**: A 800GB diverse dataset including academic papers, books, and more.

-   **CCNet**: A cleaned version of Common Crawl data optimized for training language models.

------------------------------------------------------------------------

**Fine-Tuning**

**Objective**: After pre-training, causal transformers are adapted to specific tasks using labeled or curated datasets. Fine-tuning aligns the model's general knowledge with task-specific requirements, such as summarization, dialogue generation, or code completion.

**Techniques**:

1.  **Supervised Fine-tuning**:
    -   The model is trained on a labeled dataset where input-output pairs are explicitly provided (e.g., questions paired with answers).
    -   Example: Fine-tuning a GPT model for a customer support chatbot by training it on a dataset of questions and corresponding responses.
2.  **Reinforcement Learning with Human Feedback (RLHF)**:
    -   RLHF is often used in fine-tuning tasks where human preferences are involved (e.g., generating text that is factual and polite).
    -   The process involves:
        -   Collecting human feedback on generated outputs.
        -   Training a reward model to predict human preferences.
        -   Fine-tuning the transformer using reinforcement learning to optimize the reward signal.
    -   Example: GPT models use RLHF to align outputs with user preferences and avoid generating harmful or biased text.

------------------------------------------------------------------------

#### **3. Efficiency Through Decoding**

During inference, causal transformers generate text **autoregressively**, meaning that tokens are produced one at a time based on the preceding context. The decoding process can be optimized using various strategies to balance **efficiency**, **quality**, and **diversity**.

**Decoding Strategies**

1.  **Greedy Decoding**: The simplest decoding method. At each step, the model selects the token with the highest probability: $$x_t = \arg \max P(x_t | x_1, x_2, \dots, x_{t-1})$$
    -   Advantage: Computationally efficient and deterministic.
    -   Limitation: May produce repetitive or suboptimal results because it does not explore alternative sequences.
2.  **Beam Search**: Explores multiple candidate sequences (beams) simultaneously to find the most likely overall sequence. At each step, it keeps the top $k$ beams with the highest cumulative probabilities.
    -   Advantage: Produces higher-quality outputs compared to greedy decoding.
    -   Limitation: Computationally expensive and may still favor generic sequences.
3.  **Sampling Methods**: Introduces randomness to encourage diverse and creative outputs:
    -   **Top-k Sampling**: Samples from the $k$most probable tokens at each step, reducing low-probability noise.
    -   **Nucleus Sampling (Top-p Sampling)**: Samples from the smallest set of tokens whose cumulative probability exceeds a threshold $p$, dynamically adjusting the candidate pool size.
    -   Advantage: Balances quality and diversity, making it ideal for tasks like creative writing.
    -   Limitation: May occasionally generate incoherent outputs if randomness is too high.

------------------------------------------------------------------------

### Advantages of Causal Transformers:

1.  **Sequential Text Generation**: Ensures coherent and contextually accurate outputs for tasks like text generation and dialogue.
2.  **Adaptability Across Domains**: Generalizes well across diverse domains with minimal task-specific fine-tuning.
3.  **Simplicity in Objective**: Predicting the next token is straightforward, making pre-training scalable and efficient.
4.  **Rich Contextual Understanding**: Builds detailed context by attending to all preceding tokens in a sequence.
5.  **Optimizable**: Compatible with techniques like quantization and caching for improved efficiency.

------------------------------------------------------------------------

### Limitations of Causal Transformers:

1.  **Unidirectional Context**: Can only use past tokens, limiting tasks requiring bidirectional understanding.
2.  **Sequential Inference**: Token-by-token generation slows inference for long outputs.
3.  **Resource Intensive**: Demands significant computational resources for training and fine-tuning.
4.  **Repetition Issues**: May generate repetitive sequences without advanced decoding strategies.
5.  **Long-Term Dependencies**: Struggles with very long contexts despite self-attention optimizations.

------------------------------------------------------------------------

## LLAMA: Large Language Model Meta AI

LLAMA (Large Language Model Meta AI) is a family of causal transformers developed by Meta, designed to optimize performance and efficiency in natural language understanding and generation tasks.

### Key Features of LLAMA:

1.  **Lightweight Design:** LLAMA is built to be computationally efficient, reducing resource requirements while maintaining high performance. This makes it accessible for research and smaller-scale deployments.
2.  **Training on Diverse Data:** LLAMA models are trained on a mix of publicly available datasets to generalize well across a variety of tasks.
3.  **Variants of LLAMA:** LLAMA comes in different sizes (e.g., LLAMA-7B, LLAMA-13B) to cater to various hardware and task requirements.
4.  **Advanced Optimizations:** Techniques like rotary positional embeddings and tokenization optimizations make LLAMA models faster and more robust.

### LLAMA vs. Other Transformers:

| Feature          | LLAMA              | GPT (OpenAI)       | BERT                |
|------------------|-------------------|------------------|-----------------------------------------------------|
| Architecture     | Causal Transformer | Causal Transformer | Encoder Transformer |
| Usage            | Text Generation    | Text Generation    | Classification, QA  |
| Efficiency Focus | High               | Moderate           | Moderate            |

### LLAMA Applications:

-   Chatbots
-   Summarization
-   Question Answering
-   Creative Writing

## Training Methodologies of LLaMA

The **LLaMA (Large Language Model Meta AI)** project introduced a family of foundation models ranging from 7 billion to 65 billion parameters. These models are optimized for efficient training and inference using publicly available datasets. Below is a detailed breakdown of the methodologies used to train the LLaMA models as described in the paper.

### Pre-Training Data

LLaMA models were trained on **1.4 trillion tokens**, leveraging a diverse mix of publicly available datasets to ensure generalizability across tasks while adhering to open-source compatibility.

#### Dataset Composition

| **Dataset**               | **Proportion** | **Epochs** | **Disk Size** |
|---------------------------|----------------|------------|---------------|
| CommonCrawl               | 67.0%          | 1.10       | 3.3 TB        |
| C4                        | 15.0%          | 1.06       | 783 GB        |
| GitHub                    | 4.5%           | 0.64       | 328 GB        |
| Wikipedia                 | 4.5%           | 2.45       | 83 GB         |
| Books (Books3, Gutenberg) | 4.5%           | 2.23       | 85 GB         |
| ArXiv                     | 2.5%           | 1.06       | 92 GB         |
| StackExchange             | 2.0%           | 1.03       | 78 GB         |

#### Preprocessing Steps

1.  **Deduplication**: Applied both line-level and file-level deduplication to eliminate redundant data.
2.  **Language Filtering**: Retained only English content using FastText classifiers.
3.  **Domain-Specific Filters**:
    -   **GitHub**: Included repositories under permissive licenses and filtered out low-quality or boilerplate code.
    -   **ArXiv**: Removed headers, bibliographies, and comments for cleaner data.
4.  **Tokenization**: Utilized Byte Pair Encoding (BPE) with SentencePiece, with special handling for numeric sequences and UTF-8 characters.

### Architecture and Enhancements

The LLaMA architecture builds upon the Transformer model with several enhancements for stability, efficiency, and scalability:

1.  **Pre-Normalization**: Normalizes the input to each transformer sub-layer using RMSNorm to improve stability during training.
2.  **SwiGLU Activation**: Replaces ReLU activation with SwiGLU for better performance and reduced computational cost.
3.  **Rotary Positional Embeddings (RoPE)**: Uses rotational embeddings to capture long-range dependencies, replacing absolute positional encodings.

#### Model Configurations

| **Model** | **Parameters** | **Dimensions** | **Heads** | **Layers** | **Learning Rate** | **Batch Size** | **Tokens (Trillions)** |
|---------|---------|---------|---------|---------|---------|---------|---------|
| LLaMA-7B | 6.7B | 4096 | 32 | 32 | 3.0e-4 | 4M | 1.0T |
| LLaMA-13B | 13.0B | 5120 | 40 | 40 | 3.0e-4 | 4M | 1.0T |
| LLaMA-33B | 32.5B | 6656 | 52 | 60 | 1.5e-4 | 4M | 1.4T |
| LLaMA-65B | 65.2B | 8192 | 64 | 80 | 1.5e-4 | 4M | 1.4T |

### Optimization

LLaMA models were trained using the **AdamW optimizer**, with the following configurations:

**Hyperparameters**:

-   $β₁ = 0.9$
-   $β₂ = 0.95$
-   Weight Decay = 0.1
-   Gradient Clipping = 1.0

**Learning Rate Schedule**:

-   Cosine learning rate schedule with a final learning rate set to 10% of the maximum.
-   Warmup steps: 2000.

### Efficiency Optimizations

1.  **Memory-Efficient Attention**: Implemented memory-efficient causal multi-head attention to reduce computational overhead.
2.  **Activation Checkpointing**: Saved intermediate activations during backpropagation to minimize recomputation and reduce memory usage.
3.  **Parallelism**: Combined model and sequence parallelism to distribute workloads across GPUs, optimizing utilization.

## Training Timeline

Models were trained on **2048 NVIDIA A100 GPUs** over approximately 5 months.

Throughput: **380 tokens/sec/GPU** for LLaMA-65B.

### Performance Benchmarks

LLaMA demonstrated competitive performance across various benchmarks, often outperforming much larger models:

| **Model** | **MMLU (5-shot)** | **TriviaQA** | **PIQA** |
|-----------|-------------------|--------------|----------|
| LLaMA-7B  | 35.1%             | 50.0%        | 76.5%    |
| LLaMA-13B | 46.9%             | 56.6%        | 78.1%    |
| LLaMA-33B | 57.8%             | 65.1%        | 83.1%    |
| LLaMA-65B | 63.4%             | 68.2%        | 85.3%    |

### Key Innovations

1.  **Data Efficiency**: Trained on publicly available datasets, ensuring reproducibility and transparency.
2.  **Scalability**: Maintains robust performance as models scale from 7B to 65B parameters.
3.  **Accessibility**: By prioritizing open-source datasets and efficient architecture, LLaMA democratizes access to high-quality LLMs.

### References

-   Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," [arXiv:2302.13971](https://arxiv.org/abs/2302.13971).

------------------------------------------------------------------------

## 8-bit Quantization for Large Language Models

The representation of neural network weights using 8 bits instead of the standard 32 bits has emerged as a groundbreaking method to reduce memory requirements and computational overhead for inference, particularly in Large Language Models (LLMs). The methodology, described in the paper "[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)", introduces techniques to achieve performance parity with 16-bit precision models while significantly lowering resource consumption.

### Motivation for 8-bit Quantization

#### Challenges of Large LLMs

-   Large Language Models (LLMs) like LLAMA often exceed billions of parameters (from 7 to 405 billion parameters), demanding enormous computational resources for inference.
-   Traditional inference in 16-bit or 32-bit precision requires substantial GPU memory, making such models inaccessible to researchers or organizations with limited resources.

#### Benefits of 8-bit Quantization

-   **Memory Efficiency:** Reduces memory usage by up to 50% compared to 16-bit models, enabling deployment on consumer-grade GPUs.
-   **Inference Accessibility:** Makes inference for multi-billion-parameter models feasible on a single server with consumer GPUs.
-   **Computational Savings:** Decreases latency and improves throughput without sacrificing accuracy.

### Core Methodology of 8-bit Quantization

#### Quantization Basics

Quantization involves reducing the precision of numerical representations. In the case of 8-bit quantization:

-   Neural network weights and activations are represented with 8 bits instead of 16 or 32 bits.

-   Integer arithmetic replaces floating-point arithmetic, improving efficiency.

**Mathematical Definition**

Given a tensor $X_{f16}$ in 16-bit floating-point precision: $$X_{i8} = \left\lfloor \frac{127 \cdot X_{f16}}{\max |X_{f16}|} \right\rfloor$$ 

where:

-   $X_{i8}$: Quantized tensor in 8-bit integers.

-   $\max |X_{f16}|$: Maximum absolute value in the 16-bit tensor.

-   $\lfloor \cdot \rfloor$: Floor operation.

#### LLM.int8() Methodology

The paper introduces a two-part quantization strategy called **LLM.int8()**, which addresses key challenges at scale:

**1. Vector-wise Quantization**

-   Assigns separate scaling constants to each row and column of matrices during matrix multiplications.
-   Reduces precision loss by normalizing individual components of the computation.
-   Denormalization occurs after matrix multiplication to recover scaled outputs: 
$$C_{f16} \approx \frac{C_{i32}}{c_{x} \otimes c_{w}}$$

where:

-   $c_x$: Row-wise scaling constants.

-   $c_w$: Column-wise scaling constants.

**2. Mixed-Precision Decomposition**

-   Isolates outlier dimensions (rare, high-magnitude features) into a 16-bit matrix multiplication, while the remaining 99.9% of dimensions use 8-bit precision.
-   Ensures precision for critical computations without sacrificing memory efficiency.

### Overcoming Challenges at Scale

#### Emergence of Outliers

-   Large-scale LLMs (e.g., \>6.7B parameters) exhibit "outlier features," dimensions with exceptionally high magnitude values.
-   Outliers disrupt quantization precision, causing severe performance degradation.

**Solutions for Outliers**

-   **Mixed-Precision Handling:** High-magnitude outliers are isolated and processed using 16-bit precision, while the rest of the tensor is quantized to 8 bits.
-   **Dynamic Thresholding:** The threshold for detecting outliers is adaptively set based on observed magnitude distributions.

### Experimental Results of the LLM.int8() Methodology Paper

#### Performance Metrics

The methodology was evaluated on LLMs ranging from 125M to 175B parameters across various tasks (e.g., language modeling, zero-shot classification).

##### Perplexity Results

| Model Size (Params) | 32-bit Baseline | Int8 AbsMax | LLM.int8() |
|---------------------|-----------------|-------------|------------|
| 125M                | 25.65           | 87.76       | 25.83      |
| 6.7B                | 13.30           | 14.59       | 13.24      |
| 13B                 | 12.45           | 19.08       | 12.45      |

**Observation:** Standard quantization methods fail to maintain precision as model size increases. LLM.int8() preserves baseline performance.

### Advantages of LLM.int8()

1.  **Scalability:** Enables inference for models up to 175B parameters on consumer-grade GPUs.
2.  **No Performance Degradation:** Maintains full precision performance for most tasks.
3.  **Cost Reduction:** Reduces the financial and environmental costs of operating LLMs at scale.

### References

-   Tim Dettmers et al., "[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)"

## Implementation:

### Overview

The file `quantize_model.py` provides a streamlined way to load and compare pre-trained language models in both quantized and full-precision formats. This implementation leverages Hugging Face's `transformers` library, along with `BitsAndBytesConfig` for efficient 8-bit quantization.

### Key Components

#### 1. Loading Quantized Models

-   **Function:** `load_quantized_model`
-   **Purpose:** Reduces memory usage and improves efficiency by converting model weights to 8-bit integers.
-   **Implementation:**
    -   Uses `BitsAndBytesConfig` to enable 8-bit loading.
    -   Maintains accuracy by handling outlier features through the `llm_int8_threshold` parameter.
    -   Returns a lightweight, memory-efficient model suitable for resource-constrained environments.

#### 2. Loading Full-Precision Models

-   **Function:** `load_full_model`
-   **Purpose:** Loads the full-precision version of the model for high-fidelity tasks requiring maximum accuracy.
-   **Implementation:**
    -   Configured for FP16 to optimize memory usage on GPUs.
    -   Uses `low_cpu_mem_usage` to minimize memory overhead during model loading.
    -   Allows flexible deployment by supporting both CPU and GPU devices.

### Usage

#### 1. Run the Script

Execute the script to load both quantized and full-precision versions of a specified model:

``` bash
python quantize_model.py
```

#### 2. Specify the Model

Update the `model_name` variable to choose a pre-trained model:

``` python
model_name = "meta-llama/Llama-3.1-8B"
```

#### 3. Customize Device

Automatically detects and uses GPU if available, falling back to CPU otherwise:

``` python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Example Workflow

### Load Models

Both quantized and full models are loaded into memory:

``` python
quantized_model = load_quantized_model(model_name, HF_TOKEN)
full_model = load_full_model(model_name, HF_TOKEN, device)
```

### Generate Text

Use the loaded models for tasks like text generation:

``` python
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("Once upon a time", return_tensors="pt").to(device)
outputs = full_model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## Evaluate Performance

### Overview

The `evaluate_model.py` file provides a systematic framework for evaluating the performance of quantized and full-precision models on the **PIQA (Physical Interaction Question Answering)** dataset. The script measures critical metrics such as accuracy, latency, and memory usage and generates visual comparisons for better insights.

### Key Components

#### 1. Evaluating a Model on PIQA

-   **Function:** `evaluate_model_on_piqa`
-   **Purpose:** This function evaluates a given model on the PIQA dataset, tracking its performance and resource usage.
-   **Key Features:**
    -   **Accuracy Calculation:** Determines the percentage of correctly predicted answers.
    -   **Latency Measurement:** Measures the time taken to process each question.
    -   **Memory Usage Tracking:** Records the peak memory utilized during inference.
    -   **Progress Tracking:** Displays progress using a visual progress bar with `tqdm`.
-   **Implementation Details:**
    -   Loads the PIQA validation dataset using the Hugging Face `datasets` library.
    -   For each question in the dataset, the model compares scores for two provided answers and predicts the most plausible one.
    -   Returns a summary of metrics including accuracy, average latency, and memory usage.

#### 2. Comparing Models

-   **Function:** `compare_models`
-   **Purpose:** Compares the performance of multiple models and generates visualizations.
-   **Key Features:**
    -   **Summary Table:** Outputs a performance table for all models evaluated.
    -   **Visualization:** Generates bar charts for:
        -   Accuracy comparison.
        -   Latency comparison.
        -   Memory usage comparison.
    -   **File Saving:** Saves results and plots in `figures/` directory for reporting purposes.

#### Usage

**1. Evaluate a Model** The script evaluates both quantized and full-precision models:

``` python
full_results = evaluate_model_on_piqa(full_model, model_name, tokenizer, device, "Non-Quantized Model")
quantized_results = evaluate_model_on_piqa(quantized_model, model_name, tokenizer, device, "Quantized Model")
```

**2. Compare Models** The comparison is made using the results collected from both models:

``` python
compare_models([quantized_results, full_results])
```

**3. Run the Script** Execute the script to load models, evaluate them on PIQA, and generate visual comparisons:

``` bash
python evaluate_model.py
```

## PIQA Dataset and Accuracy Computation

### Overview of the PIQA Dataset

PIQA (Physical Interaction Question Answering) is a dataset designed to evaluate a model's ability to reason about physical interactions in the real world. It focuses on common-sense knowledge related to how objects are used and how physical processes occur.

### Dataset Details

**Structure:**

Each entry in the dataset contains:

-   **Goal:** A description of a physical situation or task (e.g., "How do you light a match?").
-   **Solution 1 (sol1):** A possible answer to the goal.
-   **Solution 2 (sol2):** Another possible answer to the goal.
-   **Label:** The correct answer, represented as `0` (sol1) or `1` (sol2).

**Size:**

-   Validation set contains 1,838 examples.
-   Designed to challenge a model’s ability to perform reasoning beyond surface-level text matching.

### Objective

The task is to select the most plausible solution (`sol1` or `sol2`) for a given goal based on common-sense reasoning.

------------------------------------------------------------------------

### Accuracy Computation

#### Process

Accuracy measures how often the model selects the correct solution from the two provided options.

1.  **Score Computation:**
    -   For each example, the model generates scores for both solutions (`sol1` and `sol2`) based on its internal representations.
    -   Scores are typically derived from the logits of the model’s output.
2.  **Prediction:**
    -   The solution with the higher score is selected as the model’s prediction.

    -   Predicted choice:

        $$text_{predicted\_choice} = \text{argmax}(\text{score1}, \text{score2})$$

3.  **Correctness Check:**
    -   Compare the predicted choice with the label (`0` or `1`) in the dataset:
        -   If the predicted choice matches the label, it is counted as correct.
4.  **Accuracy Formula:**
    -   Accuracy is the ratio of correct predictions to total predictions:
    $$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}}$$

### Example

**Input:**

-   Goal: "How do you light a candle?"
-   sol1: "Use a lighter to ignite the wick."
-   sol2: "Blow air onto the wick."
-   Label: `0` (sol1 is correct).

**Model Output:**

-   Score1: 0.85 (higher score for sol1).
-   Score2: 0.45.
-   Predicted Choice: `0`.
-   Result: Correct prediction.

### Significance of PIQA Accuracy

-   **Evaluates Common-Sense Reasoning:** Tests the model's ability to understand physical interactions and causal relationships.
-   **Task-Oriented:** Goes beyond generic text understanding to address practical, real-world scenarios.
-   **Comparison Metric:** Accuracy provides a straightforward metric to compare the performance of different models.

### Limitations

-   **Ambiguity:** Some examples may have subjective interpretations of the correct answer.
-   **Context:** The dataset assumes basic knowledge of physical processes, which models might lack without pre-training on relevant data.

By measuring accuracy on the PIQA dataset, we assess a model's capability to perform physical reasoning and generalize its understanding of real-world tasks.

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