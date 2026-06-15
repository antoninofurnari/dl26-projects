> **NOTE: This file is the official template for the technical README of your repository.**  
> Before starting, make sure you have carefully read the **[INSTRUCTIONS.md](INSTRUCTIONS.md)**.  
> This file must contain **exclusively the technical aspects** of the project (Setup, Run, baseline Results). The textual and theoretical report should be placed in the **[`docs/REPORT.md`](docs/REPORT.md)** file.
> *Delete this note block before submission.*

# Multi-source Domain Adaptation for Action Recognition

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information
- **Group ID**: DataLost
- **Project ID**: Track 9

## 📝 Project Description
A brief paragraph (3-4 lines) that visually and concisely describes the project, the main implemented model, and the task addressed. 
*(Imagine this is the technical Abstract of your GitHub repo).*

> 📖 **Official Report**: For all theoretical details, performance analysis, the architecture used, and group contributions, please refer to our formal paper: **[REPORT.md](docs/REPORT.md)**.

## 🛠 Technical Reproducibility

### 1. Data and Environment Setup

**Prerequisites:**

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
conda env create -f environment.yml
conda activate dl-project
```

**Dataset:**
Explain in 2 lines where to download the data from and in which folder it needs to reside (e.g., `data/raw/`).

### 2. Network Training
Provide the **exact commands** to start the training.

**Baseline Training:**
```bash
python -m src.training.train --config experiments/configs/baseline.yaml
```

**Improved Model Training:**
```bash
python -m src.training.train --config experiments/configs/model_v1.yaml
```

### 3. Evaluation
Provide the commands to reproduce the numbers in your summary table.

```bash
python -m src.evaluation.evaluate --config experiments/configs/model_v1.yaml
```

---

*For the declaration of individual tasks and the use of AI, refer to `docs/REPORT.md`.*
