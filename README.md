# EpiRNAT
<img width="1399" height="628" alt="image" src="https://github.com/user-attachments/assets/e6fba7c2-d31e-41fa-bf83-390e1a64de6a" />

# EpiRNAT – Transformer‑Biophysical Fusion Validator

[![DOI (Transformer)](https://zenodo.org/badge/DOI/10.5281/zenodo.20676854.svg)](https://doi.org/10.5281/zenodo.20676854)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformer%20Validator-blue)](https://huggingface.co/spaces/supzammy/epiRNAT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EpiRNAT** is the **high‑fidelity validator** of the EpiRNA suite.  
It fuses a frozen **DNA‑BERT** transformer with a biophysical CNN to deliver **accurate, high‑specificity classification** of candidate m⁶A sites.

The EpiRNA project provides **two complementary models** — use the CNN scanner for fast whole‑transcript profiling, and this transformer variant to validate individual windows with confidence.

| Model | Backbone | AUROC (real miCLIP) | Inference | Use case |
|-------|----------|---------------------|-----------|----------|
| **Biophysical Tensor Fusion (CNN)** | Cross‑Scale CNN with biophysical embeddings | 0.68 | < 5 s per sequence | Live interactive EBCS visualisation |
| **Transformer‑Biophysical Fusion** | Frozen DNA‑BERT + biophysical CNN | **0.80** | ~60 s per sequence (single‑window) | Offline, high‑specificity validation |

Both models were trained on **11,844 experimentally verified human m⁶A sites (GSE63753, miCLIP)** and an equal number of DRACH‑containing negative windows.

---

## 🚀 Live Demos

- **Transformer validator (this tool):**  
  [huggingface.co/spaces/supzammy/epiRNAT](https://huggingface.co/spaces/supzammy/epiRNAT)  
- **Real‑time CNN scanner (for whole transcripts):**  
  [huggingface.co/spaces/supzammy/EpiRNAC](https://huggingface.co/spaces/supzammy/EpiRNAC)

---

## ✨ Key Features

- **High‑specificity classification** – Achieves **specificity 0.83**, minimising false positives.
- **Frozen DNA‑BERT backbone** – Pre‑trained on the human genome, providing rich contextual features.
- **Biophysical fusion** – Same 3‑path dilated CNN and Cross‑Scale Fusion Gates as the CNN model.
- **Tunable detection modes** – Discovery, Standard, Strict, and Clinical thresholds (τ = 0.0–0.90).
- **Sliding‑window profiling** – Processes sequences up to ~200 bp; for longer sequences use the CNN scanner first.
- **Batch processing** – Upload CSV/FASTA files for multi‑window validation.
- **NCBI streaming** – Fetch sequences directly from NCBI via accession number.

---

## 📊 Performance on real miCLIP data

| Metric | Value |
|--------|-------|
| **AUROC** | **0.8034** |
| **Specificity** | **0.8272** |
| **Sensitivity** | 0.6145 |
| **F1 Score** | 0.6877 |

*Evaluation on held‑out test set (GSE63753, n = 3,554 sequences).*

---

## 🧬 Architecture

### Transformer‑Biophysical Fusion
- **Backbone:** Frozen `armheb/DNA_bert_6` (768‑dim CLS token).
- **Biophysical branch:** Three dilated 1D‑convolutional paths (local, flank, structure) with Cross‑Scale Fusion Gates, max‑pooled to 96‑dim.
- **Fusion:** CLS token projected to 96‑dim, concatenated with CNN features → 192‑dim vector → linear classifier.
- **Training:** Only the biophysical embedding, CNN paths, fusion gates, BERT projector, and classifier head are trained; BERT weights remain frozen.

### EBCS – Epitranscriptomic Boundary Contrast Scoring (CNN only)
The CNN model uses a sliding‑window approach with a local‑global variance blender to generate per‑nucleotide contrast profiles. This transformer variant outputs a single classification score per window.

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
    

## 📄 Citation
If you use EpiRNA, please cite:

> Zaeem Ahmad Mansoori et al. (2026). “EpiRNA: Single‑Nucleotide Resolution Mapping of RNA Catalytic Boundaries Using Biophysical Tensor Fusion.” InCoB 2025.

 * Transformer checkpoint: https://zenodo.org/badge/DOI/10.5281/zenodo.20676854.svg

* CNN checkpoint & benchmarks: https://zenodo.org/badge/DOI/10.5281/zenodo.20615778.svg

## 📜 License
This project is licensed under the MIT License – see the MIT LICENSE file.

    ''' bash

    This README follows the exact structure you provided, includes the transformer‑specific metrics and architecture, and clearly positions the tool as the validator component. 
    It’s ready to be dropped into your `epiRNAT` repository.

