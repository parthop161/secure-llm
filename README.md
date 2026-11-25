# 🔐 Secure LLM — Quantization Techniques on Large Language Models

> Democratizing access to large language models by implementing **Mistral-7B-Instruct-v0.2** on a free T4 GPU using 4-bit quantization, achieving **4x size reduction** and **2x memory savings** while maintaining **98% accuracy**.

---

## 🎯 Objective

Large language models like Mistral-7B require expensive GPU hardware to run in full precision (`~28 GB` in FP32, `~14 GB` in FP16). This project implements **4-bit NF4 quantization** using the `bitsandbytes` library to make 7B-parameter models accessible on free-tier cloud GPUs (Google Colab T4 / Kaggle 15 GB VRAM) with minimal quality loss.

---

## 📊 Results Summary

| Metric | Full Precision (FP32) | Baseline (FP16) | Quantized (4-bit NF4) | Improvement |
|---|---|---|---|---|
| **Model Size** | ~28 GB | ~14 GB | ~7 GB | **4x reduction (vs FP32)** |
| **GPU Memory Usage** | OOM on T4 | ~14 GB | ~7 GB | **2x reduction (vs FP16)** |
| **Accuracy (Perplexity)** | Baseline | Baseline | 98% retained | **~2% loss** |
| **Hardware** | A100 / V100 required | Tight fit on T4 | ✅ Free T4 GPU | **Democratized** |

---

## 📖 Quantization Theory

### What is Quantization?
Quantization reduces the **numerical precision** of model weights from higher bit-widths (FP32 / FP16) to lower bit-widths (INT8 / NF4), dramatically shrinking the memory footprint and bandwidth requirements while keeping inference fast.

### Precision Formats Comparison

| Format | Bits | Range | Model Size (7B params) | Use Case |
|---|---|---|---|---|
| **FP32** | 32 | ±3.4×10³⁸ | ~28 GB | Full-precision training |
| **FP16** | 16 | ±65,504 | ~14 GB | Mixed-precision training & inference |
| **INT8** | 8 | -128 to 127 | ~7 GB | Standard inference optimization |
| **NF4** | 4 | Normalized Float | ~3.5–7 GB | QLoRA / BitsAndBytes 4-bit inference & fine-tuning |

### Why BitsAndBytes?
The `bitsandbytes` library implements three key optimizations used in this project:
1. **NF4 (NormalFloat 4-bit):** Information-theoretically optimal data type for normally distributed neural network weights.
2. **Double Quantization (`bnb_4bit_use_double_quant=True`):** Quantizes the quantization constants themselves, saving an extra `~0.37` bits per parameter.
3. **Mixed-Precision Compute (`bnb_4bit_compute_dtype=torch.float16`):** Stores weights compactly in 4-bit NF4 while performing matrix multiplications in FP16 for numerical stability.

---

## 🔬 How It Works

```
Full Precision (FP32, ~28 GB)
        │
        ▼
  4-bit NF4 Quantization (BitsAndBytes)
  ├── NormalFloat 4-bit encoding
  ├── Double quantization (saves ~0.37 bits/param)
  └── Mixed-precision compute (FP16)
        │
        ▼
Quantized Model (~7 GB) → Fits comfortably on free T4 GPU (15 GB VRAM) ✅
```

### Quantization Configuration
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

---

## 📋 Step-by-Step Pipeline Explanation

The notebook [`Secure_LLM_Quantization.ipynb`](Secure_LLM_Quantization.ipynb) is kept strictly code-only and executes the following pipeline:

1. **Environment Setup & Dependencies:** Installs `transformers`, `accelerate`, `bitsandbytes`, `torch`, `datasets`, `sentencepiece`, `protobuf`, and plotting libraries (`matplotlib`, `seaborn`).
2. **GPU Verification & Memory Utilities:** Runs `nvidia-smi` and defines `print_gpu_memory()` and `clear_gpu()` helpers to track exact VRAM allocation and free memory between runs.
3. **Full-Precision Baseline (FP16):** Loads `mistralai/Mistral-7B-Instruct-v0.2` in `torch.float16` with `low_cpu_mem_usage=True` and profiles parameter counts, dtype breakdown, model size, and GPU memory allocation (also computing the theoretical FP32 size at `2x` FP16).
4. **4-bit Quantized Loading:** Unloads the FP16 model, clears CUDA cache, configures `BitsAndBytesConfig` (NF4 + double quantization + FP16 compute), and loads the 4-bit quantized model.
5. **Memory & Size Profiling:** Profiles the quantized model and builds a side-by-side comparison table and bar charts (`comparison_chart.png`) across model size, GPU memory allocated, and load time.
6. **Inference Speed & Quality Test:** Runs chat-formatted prompts through `generate_response()` and measures generation latency and throughput (`tokens/sec`).
7. **Perplexity Benchmark (WikiText-2):** Evaluates both the 4-bit quantized model and the FP16 baseline on `wikitext-2-raw-v1` (test split) using `evaluate_perplexity()`. Lower perplexity indicates better next-token prediction; accuracy retained is computed as `(fp16_ppl / quant_ppl) * 100`, verifying `≥98%` retention.
8. **Results Visualization:** Generates a 3-panel summary chart (`results_summary.png`) comparing Model Size, GPU Memory Usage against the 15 GB T4 VRAM limit, and Perplexity retention.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Quantization | BitsAndBytes (4-bit NF4 + Double Quantization) |
| Framework | HuggingFace Transformers & Accelerate |
| Benchmarking | WikiText-2 (Perplexity) |
| Environment | Google Colab (T4 GPU) / Kaggle Notebooks |
| Visualization | Matplotlib, Seaborn, Pandas |

---

## 📂 Project Structure

```
secure-llm/
├── Secure_LLM_Quantization.ipynb   # Code-only experiment notebook
├── README.md                        # Full project documentation & theory
└── requirements.txt                 # Python dependencies
```

---

## 🚀 Quick Start

### Option A: Google Colab (Recommended)
1. Upload `Secure_LLM_Quantization.ipynb` to Google Colab
2. Set **Runtime → Change runtime type → T4 GPU**
3. Run all cells

### Option B: Kaggle
1. Create a new Kaggle notebook and import `Secure_LLM_Quantization.ipynb`
2. Enable **GPU T4** accelerator in settings
3. Run all cells

### Option C: Local
```bash
git clone https://github.com/parthop161/secure-llm.git
cd secure-llm
pip install -r requirements.txt
jupyter notebook Secure_LLM_Quantization.ipynb
```

---

## 📝 Key Takeaways

- **Democratized Access:** A 7B-parameter LLM that previously required high-end GPUs runs smoothly on free cloud T4 GPUs.
- **Minimal Quality Loss:** 4-bit NF4 + double quantization retains **~98%** of full-precision perplexity accuracy.
- **Practical Deployment:** At `~7 GB`, the quantized model fits comfortably inside a 15 GB T4 GPU with plenty of headroom for inference batches.

## 📚 References

- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized Large Language Models.* NeurIPS.
- Dettmers, T., & Zettlemoyer, L. (2023). *The case for 4-bit precision: k-bit Inference Scaling Laws.*
- Mistral AI. (2023). *Mistral-7B.* https://mistral.ai

## 👨‍💻 Author

**Parth Sethi**
- 📧 parthsethi161@gmail.com
- 💻 [GitHub](https://github.com/parthop161)
- 🔗 [LinkedIn](https://linkedin.com/in/parth-sethi)
