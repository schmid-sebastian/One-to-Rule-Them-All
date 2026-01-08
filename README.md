# One to Rule Them All  
### Large Language Models for Multi-Imperfection Business Process Event Log Repair

This repository contains the **software artifact** accompanying the paper:

> **One to Rule Them All: Large Language Models for Multi-Imperfection Business Process Event Log Repair**

The artifact implements a **unified, LLM-based framework** for detecting and repairing **multiple interdependent event log imperfections** in a single repair step.

---

## Table of Contents
- [Overview](#overview)
- [Artifact Description](#artifact-description)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [License](#license)

---

## Overview

Business process event logs often suffer from **multiple co-occurring imperfections**, such as missing case identifiers, incorrect timestamps, or distorted activity labels.  
Existing repair approaches typically address these issues **in isolation**, resulting in complex and fragile toolchains.

This project explores **instruction-following large language models (LLMs)** as a **single, adaptable repair mechanism** capable of:

- Diagnosing multiple imperfection patterns simultaneously  
- Proposing imperfection-specific mitigation strategies  
- Generating repaired event logs in a structured format  

---

## Artifact Description

The framework is instantiated as a **Python-based prototype** consisting of:

- Data preparation and imperfection injection routines  
- Instruction-based fine-tuning of a foundational LLM  
- An automated repair routine for imperfect event logs  
- Evaluation scripts for detection and repair quality  

The implementation is **independent of a specific process domain** and can be adapted to different event log formats and imperfection patterns.

---

## Repository Structure

```text
One-to-Rule-Them-All/
├── notebooks/                          # Data preparation, instruction design, and evaluation
│   ├── data_exploration.ipynb
│   ├── data_prep.ipynb
│   ├── evaluation.ipynb
│   ├── instruction_dataset.ipynb
│   └── train_test_split.ipynb
├── scripts/                            # Instruction data generation, fine-tuning, and inference
│   ├── batch_predictions_instruct.py
│   ├── fine_tuning_instruct.py
│   └── generate_instruction_data.py
├── LICENSE
├── README.md                           # This file
└── requirements.txt
```

---

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks to reproduce experiments and evaluations, or use the scripts for batch-oriented event log repair.

---

## Evaluation

The artifact is evaluated on eight real-life event logs with multiple imperfection patterns injected simultaneously, following a purely technical evaluation strategy within the Design Science Research paradigm.

Detailed results and metrics are reported in the accompanying paper.

---

## License

MIT License. See [LICENSE](LICENSE.md) for details.