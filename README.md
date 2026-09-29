# 🔩 MTAM-HG: A Mixture-of-Experts Heterogeneous Graph Network with Agent-Regulated Diffusion Augmentation for Strip Yield Strength Prediction

<p align="center">
  <b>Mechanism-Prior Diffusion Augmentation · CBTG-Agent · Mixture-of-experts · Heterogeneous Graph Network </b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/MTAM--HG-Yield%20Strength%20Prediction-blue">
  <img src="https://img.shields.io/badge/MP--TabDiff-Mechanism--Prior%20Augmentation-green">
  <img src="https://img.shields.io/badge/CBTG--Agent-Dynamic%20Sample%20Regulation-purple">
  <img src="https://img.shields.io/badge/MoE--IPOHGN-Heterogeneous%20Graph%20MoE-orange">
</p>

<p align="center">
  <a href="https://github.com/WeizhiZhang051029/MTAM-HG-A-Mixture-of-Experts-Heterogeneous-Graph-Network-with-Agent-Regulated-Diffusion-Augmentation">
    <b>Project Page</b>
  </a>
    |  
  <a href="#citation">
    <b>Paper</b>
  </a>
</p>

## 📌 Overview

This repository provides the official implementation of **MTAM-HG: A Mixture-of-Experts Heterogeneous Graph Network with Agent-Regulated Diffusion Augmentation for Strip Yield Strength Prediction**.

<p align="center">
  <img src="images/framework.jpg" width="100%">
</p>

<p align="center">
  <em>Overall framework of MTAM-HG for data augmentation and strip yield strength prediction in continuous annealing production lines.</em>
</p>

Yield strength is a key quality indicator in continuous annealing production lines (CAPLs), but its accurate prediction remains challenging when production records are limited and process variables are strongly coupled. Existing data-driven approaches also rarely incorporate process-mechanism constraints or explicitly account for differences among operating conditions.

To address these challenges, **MTAM-HG** integrates mechanism-prior diffusion augmentation, feedback-driven sample regulation, and heterogeneous graph mixture-of-experts prediction within a unified framework.

The MTAM-HG framework comprises two main modules:

* **Data augmentation:** MP-TabDiff embeds furnace-temperature trajectories, production-window constraints, and an empirical yield-strength prior into tabular diffusion to generate process-consistent samples. CBTG-Agent dynamically perceives operating conditions and training states, and regulates synthetic samples through iterative decision-making, feedback, selection, and reweighting.

* **MoE-IPOHGN prediction:** MoE-IPOHGN represents CAPL variables within an implicit process-order heterogeneous graph and captures condition-dependent variable interactions. A Hard Sparse Gate (HSG) adaptively activates specialized experts for different operating conditions, enabling sample-dependent yield-strength prediction.
  
Experiments on real CAPL production data demonstrate that MTAM-HG improves prediction accuracy and cross-condition stability over competitive baselines, while maintaining reliable performance under data scarcity.

---

## 🔥 Highlights

* Yield strength prediction in the continuous annealing line is investigated.
* A novel MTAM-HG framework is proposed for prediction under data scarcity.
* MP-TabDiff and CBTG-Agent are designed for mechanism-prior diffusion augmentation.
* A MoE heterogeneous graph network (MoE-IPOHGN) predicts yield strength.
* Experiments on real industrial data verify the effectiveness of MTAM-HG.

---

## 🧩 Framework

The training workflow of MTAM-HG is organized as follows:

```text
Real CAPL production data
        |
        v
Training / validation / test partition
        |
        v
MP-TabDiff training on real training data
        |
        v
Mechanism-prior synthetic sample generation
        |
        v
CBTG-Agent dynamic sample regulation
        |
        v
Selected and reweighted synthetic samples
        |
        v
MoE-IPOHGN synthetic-data pretraining
        |
        v
Real-domain LoRA calibration
        |
        v
Validation-based model selection
        |
        v
Strip yield strength prediction
```

The test set is isolated throughout model development and is used only for final evaluation.

---

## 📊 Experimental Protocol

Experiments use **600 real CAPL production records** with **21 process variables** and yield strength as the prediction target. The raw industrial data cannot be publicly released due to confidentiality.

For each of **10 independent runs**, the data are stratified by yield strength and split into training/validation/test sets at **70%/15%/15%**, with the run seed controlling both data partitioning and model initialization.

All preprocessing, clustering, and synthetic-data generation are fitted only on the corresponding real training set. CBTG-Agent uses training-set feedback, while validation is used for model selection and early stopping; the test set is reserved for final evaluation.

Evaluation metrics and prediction results are saved separately for each run.

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/WeizhiZhang051029/MTAM-HG-A-Mixture-of-Experts-Heterogeneous-Graph-Network-with-Agent-Regulated-Diffusion-Augmentation.git
cd MTAM-HG-A-Mixture-of-Experts-Heterogeneous-Graph-Network-with-Agent-Regulated-Diffusion-Augmentation
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the package:

```bash
python -m pip install --upgrade pip
pip install -e .
```

The default main experiment requires **Linux and an NVIDIA GPU with CUDA support**. Install a PyTorch build compatible with the CUDA environment. The environment-activation commands above are platform-specific examples. They do not imply that the default training configuration supports Windows or macOS.

---

## 🚀 Running

### Complete MTAM-HG Experiment

To reproduce the complete experiment:

```bash
python run_experiment.py \
  --data_path data/CAPL.xlsx \
  --config configs/mtam_hg.yaml \
  --tabdiff_num_samples 5000
```

The pipeline sequentially:

1. partitions and preprocesses the real CAPL data.
2. trains MP-TabDiff on the training partition.
3. generates mechanism-constrained candidate samples.
4. regulates synthetic samples using CBTG-Agent.
5. pretrains MoE-IPOHGN on the selected synthetic samples.
6. performs real-domain calibration.
7. selects the model according to validation performance.
8. evaluates the final model on the held-out test set.

### Data Requirement

Formal training requires an authorized CAPL dataset. This release does not include private data, synthetic smoke-test data, or a dry-run validation path.

---

## 📏 Evaluation Metrics

Yield strength prediction is evaluated using four regression metrics:

* **Root Mean Squared Error (RMSE)**
* **Mean Absolute Error (MAE)**
* **Mean Absolute Percentage Error (MAPE)**
* **Coefficient of Determination (R²)**

Lower RMSE, MAE, and MAPE values indicate smaller prediction errors, while a higher R² indicates stronger agreement between predicted and measured yield strength.

---

## 📰 News

* **July 2026** — MTAM-HG framework completed.
* **July 2026** — Manuscript completed and submitted.
* **August 2026** — Source code released.

---

## 🙏 Acknowledgements

This project builds upon **PyTorch**, **scikit-learn**, and the open-source **TabDiff** implementation.

We thank the open-source community for the tools and resources that support research in tabular diffusion modeling, heterogeneous graph learning, mixture-of-experts architectures, and parameter-efficient adaptation.

---

## 📖 Citation

If you find this repository useful in your research, please consider citing our paper:

```bibtex
@article{zhang2026mtamhg,
  title   = {MTAM-HG: A Mixture-of-Experts Heterogeneous Graph Network with Agent-Regulated Diffusion Augmentation for Strip Yield Strength Prediction},
  journal = {Expert Systems with Applications},
  year    = {2026},
}
```

The citation information will be updated after the paper is officially published.

---

## 📬 Contact

For questions regarding the implementation, experimental configuration, or reproducibility of MTAM-HG, please open an issue in this repository.
