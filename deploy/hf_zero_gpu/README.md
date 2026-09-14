---
title: IQ Remote Inference Prototype
emoji: 🧠
sdk: gradio
app_file: app.py
python_version: 3.12
---

# IQ remote inference prototype

This directory is a Hugging Face Space scaffold for running inference without a local GPU.

For the infrastructure smoke test it serves `microsoft/Phi-4-mini-instruct`. Once MOHAWK Stage 3 produces a complete IQ PyTorch checkpoint, the model loader should be replaced with the IQ reference architecture and checkpoint while keeping the same remote API surface.

## Recommended hardware

Use **ZeroGPU large (48 GB)** for the first deployment. Current Hugging Face documentation allows free personal accounts in good standing to host up to two ZeroGPU Spaces, subject to daily GPU quotas. The Space must use the Gradio SDK.

## Deployment

Create a Gradio Space, select ZeroGPU in the Space hardware settings, and copy these files into the Space repository:

- `app.py`
- `requirements.txt`
- this `README.md`

The Space then exposes both an interactive chat UI and a Gradio API endpoint. No local inference hardware is required.

## Scope

This solves **inference hosting**, not MOHAWK training. The transfer experiment runs separately in Colab because MOHAWK needs longer continuous GPU access than a request-scoped ZeroGPU function is designed to provide.
