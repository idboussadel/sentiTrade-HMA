# SentiTrade-HMA

<div align="center">
  
<img alt="FreeSample-Vectorizer-io-image-removebg-preview (1)" src="https://github.com/user-attachments/assets/f71d0ae6-2f2e-47d5-aecb-2ee540e91e7a" />

**A Synergistic Framework for Algorithmic Trading via Fusion of Large Language Models and Temporal Fusion Transformers**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>


https://github.com/user-attachments/assets/12cf1a57-e920-4863-b2f8-67a4d9b53f96


---

## Overview

SentiTrade-HMA is a production-ready algorithmic trading system that combines **context-aware sentiment analysis** from fine-tuned Llama-2-7B with **Temporal Fusion Transformers** for multi-horizon stock price forecasting. The system achieves a **Sharpe Ratio of 5.17** and **5% MAE reduction** over price-only baselines.

### Key Features

- 🧠 **LLM-Powered Sentiment Analysis**: QLoRA fine-tuned Llama-2-7B for financial news
- 📈 **Multi-Horizon Forecasting**: Temporal Fusion Transformer with quantile predictions
- 🔍 **Model Interpretability**: Attention weights and feature importance visualization
- ⚡ **Production-Ready**: Optimized API with caching and async processing
- 📊 **Real-Time Signals**: Trading signals with confidence intervals and risk metrics

---

## 🏗️ Architecture

<div align="center">
  
<img width="1360" height="784" alt="image" src="https://github.com/user-attachments/assets/ca2e0686-2ab5-4fb1-ab9a-4d7f58268659" />

**System Architecture: Multi-Source Data → LLM Sentiment Analysis → TFT Forecasting → Trading Signals**

</div>

---

## 🚀 Quick Start

### Prerequisites

- Conda (Miniconda or Anaconda)
- Python 3.11+
- CUDA (optional, for GPU acceleration)
