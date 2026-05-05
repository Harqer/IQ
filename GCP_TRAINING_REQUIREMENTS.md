# Google Cloud Training Requirements for IQ LLM
# Proprietary NIF Sovereign Architecture Training Dependencies
# Copyright © 2026 [Your Name]. All Rights Reserved.

## 📋 Overview

This document outlines the complete requirements for training the IQ LLM on Google Cloud Platform using the proprietary NIF Sovereign architecture.

## 🏗️ Architecture Requirements

### Core Components
- **NIF Sovereign Architecture**: Physics-based transformer with quantum integration
- **Mojo SDK v0.26.2+**: Primary programming language for implementation
- **CUDA-Q v0.3.0+**: Quantum computation backend
- **Thunder Compute**: Remote quantum execution platform
- **Gemma 4 (26B A4B)**: Base model for structural grafting

### Physics-Based Modules
1. **Riemannian Manifold Embedding**: Hyperbolic manifold weight initialization
2. **Neutrino Oscillation Block**: Recurrent logic loops with particle physics dynamics
3. **Ising Hamiltonian Gate**: Quantum spin system mapping for logical reasoning
4. **Heterogeneous MoE Router**: Three expert routing (Linguistic, Physics, Diffusion)

## 💻 Software Dependencies

### Core Language & Runtime
```bash
# Mojo SDK
curl --proto '=https' --tlsv1.2 -sSf https://get.modular.com | sh
mojo setup

# Verify installation
mojo --version  # Should be 0.26.2 or higher
```

### Python Dependencies
```bash
# Required Python packages
pip install torch==2.4.0
pip install torchvision==0.19.0
pip install torchaudio==2.4.0
pip install numpy==1.26.0
pip install scipy==1.13.0
pip install pandas==2.2.0
pip install matplotlib==3.8.0
pip install seaborn==0.13.0
pip install wandb==0.17.0
pip install tensorboard==2.16.0
pip install cuda-quantum==0.3.0
pip install nvidia-ml-py==12.535.133
pip install accelerate==0.28.0
pip install transformers==4.40.0
pip install datasets==2.19.0
pip install huggingface-hub==0.23.0
pip install safetensors==0.4.0
pip install bitsandbytes==0.43.0
pip install peft==0.10.0
```

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    python3-dev \
    python3-pip \
    cuda-toolkit-12-4 \
    cuda-libraries-12-4 \
    libcudnn8=8.9.7.29-1+cuda12.2
```

## 🖥️ Hardware Requirements

### Minimum Requirements
- **GPU**: 8x NVIDIA H200 (192GB VRAM total)
- **CPU**: 96+ vCPUs
- **RAM**: 1TB+ system memory
- **Storage**: 2TB+ SSD for checkpoints and data
- **Network**: 100Gbps+ for distributed training

### Recommended Configuration
- **Machine Type**: `a3-highgpu-8g` (8x H200 GPUs)
- **GPU Memory**: 192GB H200 per GPU
- **Total GPU Memory**: 1.5TB across cluster
- **CPU**: 96 vCPUs per node
- **System Memory**: 1.5TB per node
- **Storage**: 2TB NVMe SSD per node

## 📊 Data Requirements

### Training Data
```bash
# Data sources
gs://iq-training-data/gneissweb-2026/     # 70% weight
gs://iq-training-data/stack-v3/          # 30% weight

# Data formats
- Text: Parquet files with tokenized sequences
- Code: Parquet files with source code
- Mixed: Combined text and code datasets

# Data preprocessing
- Tokenization with custom vocabulary
- Sequence length: 8192 tokens
- Batch size: 32 sequences
- Sharding: 1000+ files for parallel loading
```

### Validation Data
```bash
# Validation split: 10% of training data
gs://iq-training-data/validation/
- Mixed format validation set
- 10,000 sequences for evaluation
- Balanced text/code ratio
```

### Test Data
```bash
# Test split: 5% of training data
gs://iq-training-data/test/
- Final evaluation set
- 5,000 sequences
- Held-out for final metrics
```

## 🔧 Configuration Requirements

### Model Configuration
```yaml
model:
  name: "IQ-NIF-Sovereign"
  base_model: "Gemma-4-26B-A4B"
  hidden_dim: 4096
  num_layers: 32
  num_heads: 32
  num_experts: 3
  vocab_size: 50000
  max_sequence_length: 8192
```

### Physics Configuration
```yaml
physics:
  riemannian_curvature: 0.1
  neutrino_oscillation_depth: 8
  ising_iterations: 10
  quantum_augmentation: true
  manifold_dimension: 512
```

### Training Configuration
```yaml
training:
  batch_size: 32
  learning_rate: 1e-4
  max_steps: 1000000
  warmup_steps: 10000
  save_interval: 10000
  eval_interval: 5000
  log_interval: 100
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
```

### Optimization Configuration
```yaml
optimization:
  optimizer: "muon"
  muon_momentum: 0.9
  galore_rank: 64
  vera_rank: 64
  weight_decay: 0.01
  scheduler: "cosine"
  min_lr: 1e-6
```

## 🔐 Security & Access Requirements

### Service Account Permissions
```json
{
  "required_permissions": [
    "aiplatform.customJobs.create",
    "aiplatform.customJobs.get",
    "aiplatform.customJobs.list",
    "aiplatform.customJobs.delete",
    "storage.objects.get",
    "storage.objects.create",
    "storage.objects.update",
    "storage.objects.delete",
    "storage.buckets.get",
    "storage.buckets.create",
    "storage.buckets.update",
    "logging.logEntries.create",
    "monitoring.timeSeries.create",
    "monitoring.metrics.list"
  ]
}
```

### API Keys & Endpoints
```bash
# Required environment variables
export THUNDER_COMPUTE_ENDPOINT="https://api.thundercompute.com/v1"
export NVIDIA_H200_TARGET="nvidia-h200-remote"
export CUDA_Q_API_KEY="your-cuda-q-api-key"
export THUNDER_API_KEY="your-thunder-api-key"
export WANDB_API_KEY="your-wandb-api-key"
export GCP_PROJECT_ID="iq-llm-training"
export GCP_REGION="us-central1"
```

## 💰 Cost Estimation

### Compute Costs
```bash
# H200 GPU pricing (us-central1)
- a3-highgpu-8g: $32.00/hour
- 8 GPUs × $32.00 = $256.00/hour per node
- 30 days training: $184,320 per node
- Multi-node training: $500,000 - $1,000,000 total
```

### Storage Costs
```bash
# Storage pricing
- Training data: 10TB × $0.026 = $260/month
- Checkpoints: 5TB × $0.026 = $130/month
- Logs: 1TB × $0.026 = $26/month
- Total storage: ~$416/month
```

### Network Costs
```bash
# Network egress
- Data transfer: $0.12/GB
- 100TB transfer: $12,000
- Total network: ~$15,000
```

### Total Estimated Cost
```bash
# 30-day training run
- Compute: $500,000 - $1,000,000
- Storage: $416
- Network: $15,000
- Total: $515,416 - $1,015,416
```

## 📈 Performance Targets

### Training Performance
```yaml
performance_targets:
  throughput: 1000 tokens/second/GPU
  convergence: < 1M steps
  final_loss: < 0.5
  accuracy: > 90% on validation
```

### Resource Utilization
```yaml
resource_targets:
  gpu_utilization: > 90%
  memory_utilization: > 85%
  network_bandwidth: > 80%
  storage_iops: > 10,000
```

## 🚀 Deployment Requirements

### Container Requirements
```dockerfile
FROM us-docker.pkg.dev/deeplearning-platform-release/gpu.mojo:latest

# Runtime requirements
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    cuda-toolkit-12-4

# Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Application code
COPY . /workspace
WORKDIR /workspace

# Runtime configuration
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_DEBUG=INFO
```

### Monitoring Requirements
```yaml
monitoring:
  metrics:
    - gpu_utilization
    - gpu_memory_used
    - gpu_temperature
    - training_loss
    - validation_loss
    - learning_rate
    - gradient_norm
    - throughput

  alerts:
    - gpu_failure
    - memory_exhaustion
    - training_stall
    - cost_overrun
```

## 🔧 Setup Instructions

### 1. Project Setup
```bash
# Create GCP project
gcloud projects create iq-llm-training
gcloud config set project iq-llm-training

# Enable APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com
```

### 2. Service Account Setup
```bash
# Create service account
gcloud iam service-accounts create iq-training \
    --display-name="IQ Training Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding iq-llm-training \
    --member="serviceAccount:iq-training@iq-llm-training.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

### 3. Storage Setup
```bash
# Create buckets
gsutil mb -l us-central1 gs://iq-training-data
gsutil mb -l us-central1 gs://iq-model-checkpoints
gsutil mb -l us-central1 gs://iq-training-logs
gsutil mb -l us-central1 gs://iq-model-artifacts
```

### 4. Data Upload
```bash
# Upload training data
gsutil -m rsync -r data/training/ gs://iq-training-data/training/
gsutil -m rsync -r data/validation/ gs://iq-training-data/validation/
gsutil -m rsync -r data/test/ gs://iq-training-data/test/
```

### 5. Training Job Submission
```bash
# Submit training job
./gcp_training_script.sh --train-only

# Monitor job
./gcp_training_script.sh --monitor-only
```

## 📋 Checklist

### Pre-Training Checklist
- [ ] GCP project created and configured
- [ ] All required APIs enabled
- [ ] Service account with proper permissions
- [ ] Storage buckets created
- [ ] Training data uploaded and validated
- [ ] Container built and pushed
- [ ] Configuration files validated
- [ ] API keys and endpoints configured
- [ ] Budget alerts set up
- [ ] Monitoring and logging configured

### Post-Training Checklist
- [ ] Model checkpoints downloaded
- [ ] Training logs collected
- [ ] Performance metrics analyzed
- [ ] Model evaluation completed
- [ ] Results documented
- [ ] Resources cleaned up
- [ ] Cost review completed

## 🚨 Common Issues & Solutions

### GPU Memory Issues
```bash
# Reduce batch size
export BATCH_SIZE=16

# Enable gradient checkpointing
export GRADIENT_CHECKPOINTING=true

# Use mixed precision
export MIXED_PRECISION=true
```

### Network Issues
```bash
# Check network connectivity
ping -c 3 8.8.8.8

# Test GCP connectivity
gcloud auth list
gcloud config list
```

### Storage Issues
```bash
# Check bucket permissions
gsutil ls gs://iq-training-data/

# Test data access
gsutil cat gs://iq-training-data/sample.txt
```

### Training Issues
```bash
# Check training logs
gcloud ai-platform custom-jobs describe $JOB_ID --region=us-central1

# Monitor GPU usage
nvidia-smi

# Check memory usage
free -h
```

---

**IMPORTANT**: This is proprietary software. All training configurations, model weights, and generated artifacts are confidential and belong to [Your Name]. Unauthorized distribution or modification is strictly prohibited.
