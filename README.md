# EpiRNAT
<img width="1399" height="628" alt="image" src="https://github.com/user-attachments/assets/e6fba7c2-d31e-41fa-bf83-390e1a64de6a" />

# EpiRNAT – Transformer‑Biophysical Fusion Validator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20676854.svg)](https://doi.org/10.5281/zenodo.20676854)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformer%20Validator-blue)](https://huggingface.co/spaces/supzammy/epiRNAT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EpiRNAT** is the **high‑fidelity validator** component of the EpiRNA suite.  
It uses a frozen **DNA‑BERT** transformer fused with a biophysical CNN to provide **accurate binary classification** of candidate m⁶A sites.

> *First scan with the [EpiRNAC CNN scanner](https://huggingface.co/spaces/supzammy/EpiRNAC) (real‑time, any length), then validate individual windows here.*

---

## 📊 Performance (real miCLIP data)

| Metric | Value |
|--------|-------|
| **AUROC** | **0.8034** |
| **Specificity** | **0.8272** |
| **Sensitivity** | 0.6145 |
| **F1 Score** | 0.6877 |

*Trained on 11,844 real human m⁶A sites (GSE63753) + equal number of DRACH‑containing negatives.*

---

## 🔬 When to use this model

- You have a **candidate window** (41–200 bp) from a CNN scan and need **high‑confidence validation**.
- You want to suppress false positives with **specificity > 0.82**.
- You want to test a single DRACH motif and get a **classification score**.

> For **whole‑transcript profiling** or **batch scanning**, use the [CNN scanner](https://huggingface.co/spaces/supzammy/EpiRNAC) — it processes sequences up to 200 kb in seconds.

---

## 🧬 Architecture

A frozen **DNA‑BERT‑6** backbone (768‑dim CLS token) is combined with a **biophysical CNN** (three dilated paths + Cross‑Scale Fusion Gates).  
The two branches are projected and concatenated into a 192‑dim vector, then classified by a single linear layer.  
Only the CNN, fusion gates, and projector are trained; BERT weights remain frozen.

---

## ⚙️ Detection Modes

Like the CNN scanner, this tool offers four threshold levels:

- 🔍 **Discovery** (τ = 0.0) – show all DRACH windows  
- ⚖️ **Standard** (τ = 0.45)  
- 🔬 **Strict** (τ = 0.70)  
- 🏥 **Clinical** (τ = 0.90) – only the most confident sites

---

## 🚀 Live Demo

**[huggingface.co/spaces/supzammy/epiRNAT](https://huggingface.co/spaces/supzammy/epiRNAT)**  

Paste a short RNA/DNA sequence (≥ 41 bp) or a single DRACH window, select a threshold, and get a classification score.

---

## 📦 Installation

    ```bash
    git clone https://github.com/supzammy/EpiRNAT
    cd EpiRNAT
    pip install -r requirements.txt

## 🧪 Reproducibility & Testing
Run individual window validation:
      ```bash
      # Validate a candidate DRACH window
      python epirna_engine.py --sequence "GGGGGGGGGGGGGGGGGGGGGGACTGGGGGGGGGGGGGGGG"
      # Batch validation from a FASTA file
      python epirna_engine.py --fasta test_data/candidates.fasta
      ```bash


## 📄 Citation
If you use EpiRNA, please cite:

> ** Zaeem Ahmad Mansoori et al. (2026). “EpiRNA: Single‑Nucleotide Resolution Mapping of RNA Catalytic Boundaries Using Biophysical Tensor Fusion.” InCoB 2025.

** Transformer checkpoint: https://zenodo.org/badge/DOI/10.5281/zenodo.20676854.svg

** CNN checkpoint & benchmarks: https://zenodo.org/badge/DOI/10.5281/zenodo.20615778.svg

## 📜 License
This project is licensed under the MIT License – see the MIT LICENSE file.

    ''' bash

    This README follows the exact structure you provided, includes the transformer‑specific metrics and architecture, and clearly positions the tool as the validator component. It’s ready to be dropped into your `epiRNAT` repository.

