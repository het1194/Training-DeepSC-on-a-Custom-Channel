# Training DeepSC on a Custom Channel

This repository implements a transition-aware custom channel for the DeepSC semantic communication framework. The project extends DeepSC to operate under realistic, non-stationary satellite channel conditions while preserving the original semantic transceiver architecture.

The focus of this work is on **channel realism and training-time robustness**, not model compression or architectural modification.

---

## Project Overview

The implementation introduces a unified satellite channel model with deterministic state persistence and transitions, enabling DeepSC to learn robustness against gradual channel degradation and recovery.

Key features include:
- Distance-dependent free-space path loss (FSPL)
- Rician fading with configurable K-factor
- Deterministic LOS–shadowed–LOS channel transitions
- Transition-aware training within each epoch
- Controlled evaluation of semantic degradation and recovery

---


---

## Channel Model Description

A single unified Land Mobile Satellite (LMS) channel replaces the default AWGN, Rayleigh, and Rician channels. The channel jointly models:
- path loss scaling based on satellite distance,
- small-scale fading via Rician distribution,
- perfect CSI at the receiver.

Channel parameters are updated dynamically during training to simulate persistent propagation states.

---

## Transition-Aware Training

Training is performed under a deterministic, state-persistent channel process:
- **LOS state:** low path loss, high K-factor  
- **Shadowed state:** higher path loss, Rayleigh fading  

Each state persists for a fixed number of consecutive mini-batches before transitioning within a single epoch. This ensures the model is exposed to realistic channel evolution rather than abrupt, unseen changes during evaluation.

---

## Execution Pipeline

### 1. Train DeepSC with Custom Channel
```bash
python main.py
```
**Note**: You can directly test the checkpoint given after training for 500 epochs and skip this step if you want to. 
### 2. Evaluate Under Stationary Channel Conditions
```bash
python performance.py
```
### 3. Evaluate Channel Transitions (LOS–Shadowed–LOS)
```bash
python markovperform.py
```

