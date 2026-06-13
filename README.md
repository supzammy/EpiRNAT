# EpiRNAT
<img width="1412" height="646" alt="image" src="https://github.com/user-attachments/assets/08137b0d-6a9b-4e65-bce8-8292cfedeeab" />

# EpiRNA – Single‑Nucleotide m⁶A Boundary Detection

**Biophysical Tensor Fusion with Cross‑Scale Attention for Real‑Time Transcriptome Scanning**

[![DOI (CNN)](https://zenodo.org/badge/DOI/10.5281/zenodo.20615778.svg)](https://doi.org/10.5281/zenodo.20615778)
[![DOI (Transformer)](https://zenodo.org/badge/DOI/10.5281/zenodo.20676854.svg)](https://doi.org/10.5281/zenodo.20676854)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CNN%20(Real‑time)-blue)](https://huggingface.co/spaces/supzammy/EpiRNAh)
[![Live Demo (Transformer)](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformer%20Variant-lightgrey)](https://huggingface.co/spaces/supzammy/epiRNAT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EpiRNA is a fast, interpretable tool for mapping N⁶‑methyladenosine (m⁶A) sites at **single‑nucleotide resolution** across RNA sequences of any length.  
It introduces **Epitranscriptomic Boundary Contrast Scoring (EBCS)**, a novel metric that quantifies the local change in a model’s confidence as a synthetic mask slides across the sequence.  

The project provides **two complementary models**:

| Model | Backbone | AUROC (real miCLIP) | Inference | Use case |
|-------|----------|---------------------|-----------|----------|
| **Biophysical Tensor Fusion (CNN)** | Cross‑Scale CNN with biophysical embeddings | 0.68 | < 5 s per sequence | Live interactive EBCS visualisation |
| **Transformer‑Biophysical Fusion** | Frozen DNA‑BERT + biophysical CNN | **0.80** | ~60 s per sequence | Offline, genome‑wide classification |

Both models were trained and evaluated on **11,844 experimentally verified human m⁶A sites (GSE63753, miCLIP)** and an equal number of DRACH‑containing negative windows.

---

## 🚀 Live Demos

- **Real‑time CNN interface (recommended for exploration):**  
  [huggingface.co/spaces/supzammy/EpiRNAh](https://huggingface.co/spaces/supzammy/EpiRNAh)  
- **Transformer variant interface (slower, for batch/offline use):**  
  [huggingface.co/spaces/supzammy/epiRNAT](https://huggingface.co/spaces/supzammy/epiRNAT)

---

## ✨ Key Features

- **Length‑agnostic** – Processes sequences from 41 nt to >100 kb without manual windowing or memory errors.
- **Single‑nucleotide resolution** – EBCS pinpoints the exact catalytic adenosine within DRACH motifs.
- **Tunable detection modes** – Choose Discovery, Standard, Strict, or Clinical thresholds to control sensitivity/specificity.
- **Interpretable profiles** – Per‑nucleotide contrast plots with GC‑content overlay and DRACH alignment markers.
- **Pre‑trained transformer variant** – Hybrid architecture combining frozen DNA‑BERT with the biophysical CNN, achieving AUROC 0.80 and specificity 0.83.
- **Batch processing** – Upload CSV/FASTA files and download scored results.
- **Explainability (CNN only)** – Integrated Gradients via Captum highlights which bases influence the model’s decision.

---

## 🧬 Architecture

### Biophysical Embedding
Each nucleotide is represented by a fixed 3‑dimensional vector encoding hydrogen‑bond potential, base‑stacking energy, and solvent accessibility.

### Multi‑Path Dilated Convolution with Cross‑Scale Fusion
Three parallel 1D‑convolutional paths (local, flank, structure) process the sequence at different scales. Cross‑Scale Fusion Gates allow the shorter‑range paths to dynamically attend to the long‑range structural path before pooling.

### Transformer‑Biophysical Fusion (Advanced)
A frozen DNA‑BERT transformer (pre‑trained on the human genome) provides a 768‑dimensional contextual representation. This is projected and concatenated with the max‑pooled biophysical features, then passed through a single classification head. The BERT backbone remains frozen; only the biophysical CNN, fusion gates, and projector are trained.

### EBCS – Epitranscriptomic Boundary Contrast Scoring
The model slides a 41‑nt window across the input sequence. Raw outputs are stabilised by a local‑global variance blender (α = 0.3) and passed through a noise gate. The resulting per‑nucleotide contrast profile is plotted alongside GC content, and DRACH motifs are automatically annotated.

---

## 📦 Installation

    ```bash
    git clone https://github.com/supzammy/EpiRNAT
    cd EpiRNAT
    python -m venv epirna_env
    source epirna_env/bin/activate   # Windows: epirna_env\Scripts\activate
    pip install -r requirements.txt
    
    ```bash

🧪 Reproducibility & Testing
    Run the included stress tests:
    
    # Test motif disruption
    python epirna_engine.py --fasta test_data/MYC_wildtype.fasta
    python epirna_engine.py --fasta test_data/MYC_mutant_U135C.fasta
    
    # Homopolymer stability test
    python epirna_engine.py --fasta test_data/TEST_01_HOMOPOLYMER_STABILITY.fasta
    
    # Direct sequence input
    python epirna_engine.py --sequence "UCCGGCUCCGCUUCGGCGGACUCCGGCUUCGGC"

## 📄 Citation
If you use EpiRNA, please cite both the software and the corresponding manuscript:

###Zaeem Ahmad Mansoori et al. (2026). “EpiRNA: Single‑Nucleotide Resolution Mapping of RNA Catalytic Boundaries Using Biophysical Tensor Fusion.” InCoB 2025.

  * - CNN model checkpoint: https://zenodo.org/badge/DOI/10.5281/zenodo.20615778.svg

  * - Transformer model checkpoint: https://zenodo.org/badge/DOI/10.5281/zenodo.20676854.svg

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

The transformer Space also requires a `requirements.txt` with `gradio`, `torch`, `transformers`, `matplotlib`, `numpy`, `pandas`, `captum`.
