# 🔐 Secure LLM — Quantization Techniques on Large Language Models

> Democratizing access to large language models by implementing **Mistral-7B** on a free T4 GPU using 4-bit quantization, achieving **4x size reduction** and **2x memory savings** while maintaining **98% accuracy**.

---

## 🎯 Objective

Large language models like Mistral-7B require expensive GPU hardware to run. This project implements **4-bit NF4 quantization** using the BitsAndBytes library to make these models accessible on free-tier cloud GPUs (Google Colab T4 / Kaggle).

## 📊 Results

| Metric | Full Precision (FP32) | Quantized (4-bit NF4) | Improvement |
|---|---|---|---|
| **Model Size** | ~28 GB | ~7 GB | **4x reduction** |
| **Memory Usage** | ~14 GB (FP16) | ~7 GB | **2x reduction** |
| **Accuracy** | Baseline | 98% of baseline | **~2% loss** |
| **Hardware** | A100/V100 required | ✅ Free T4 GPU | **Democratized** |

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Model | Mistral-7B-Instruct-v0.2 |
| Quantization | BitsAndBytes (4-bit NF4) |
| Framework | HuggingFace Transformers |
| Benchmarking | WikiText-2 (Perplexity) |
| Environment | Google Colab (T4 GPU) / Kaggle |
| Visualization | Matplotlib, Seaborn |

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
Quantized Model (~7 GB) → Runs on free T4 GPU ✅
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

## 📂 Project Structure

```
secure-llm/
├── Secure_LLM_Quantization.ipynb   # Complete notebook with all experiments
├── README.md                        # This file
└── requirements.txt                 # Dependencies
```

## 🚀 Quick Start

### Option A: Google Colab (Recommended)
1. Upload `Secure_LLM_Quantization.ipynb` to Google Colab
2. Set Runtime → T4 GPU
3. Run all cells

### Option B: Kaggle
1. Create a new Kaggle notebook
2. Import the notebook
3. Enable GPU accelerator (T4)
4. Run all cells

### Option C: Local
```bash
git clone https://github.com/parthop161/secure-llm.git
cd secure-llm
pip install -r requirements.txt
jupyter notebook Secure_LLM_Quantization.ipynb
```

## 📋 Notebook Sections

1. **Environment Setup** — Install dependencies
2. **GPU Verification** — Confirm T4 hardware
3. **Quantization Theory** — NF4, double quant explained
4. **Full-Precision Baseline** — Load FP16 model, measure size/memory
5. **4-bit Quantization** — Apply BitsAndBytes config
6. **Comparative Analysis** — Side-by-side size & memory charts
7. **Inference Test** — Generate responses, measure speed
8. **Perplexity Benchmark** — WikiText-2 accuracy evaluation
9. **Results Visualization** — Publication-ready charts
10. **Conclusion** — Summary of findings

## 📚 References

- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized Large Language Models.* NeurIPS.
- Dettmers, T. & Zettlemoyer, L. (2023). *The case for 4-bit precision: k-bit Inference Scaling Laws.*
- Mistral AI. (2023). *Mistral-7B.* https://mistral.ai

## 👨‍💻 Author

**Parth Sethi**
- 📧 parthsethi161@gmail.com
- 💻 [GitHub](https://github.com/parthop161)
- 🔗 [LinkedIn](https://linkedin.com/in/parth-sethi)
