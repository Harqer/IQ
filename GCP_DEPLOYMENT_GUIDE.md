# Google Cloud Deployment Guide for IQ LLM
# Step-by-Step Instructions for Training Proprietary NIF Sovereign Architecture
# Copyright © 2026 [Your Name]. All Rights Reserved.

## 🚀 Quick Start

### Prerequisites
- Google Cloud account with billing enabled
- Project ID: `iq-llm-training`
- Budget: $500,000 - $1,000,000 for 30-day training
- Administrative access to GCP APIs

### One-Click Deployment
```bash
# Clone and deploy
git clone https://github.com/yourusername/IQ.git
cd IQ
chmod +x gcp_training_script.sh
./gcp_training_script.sh
```

## 📋 Detailed Deployment Steps

### Step 1: Google Cloud Setup

#### 1.1 Create Project
```bash
# Create new project
gcloud projects create iq-llm-training \
    --name="IQ LLM Training" \
    --organization=your-organization-id

# Set active project
gcloud config set project iq-llm-training

# Verify project
gcloud config list project
```

#### 1.2 Enable Required APIs
```bash
# Enable AI Platform APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Enable Storage APIs
gcloud services enable storage.googleapis.com
gcloud services enable storage-component.googleapis.com

# Enable Logging & Monitoring
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com

# Enable Compute APIs
gcloud services enable compute.googleapis.com
gcloud services enable container.googleapis.com

# Enable IAM APIs
gcloud services enable iam.googleapis.com
gcloud services enable iamcredentials.googleapis.com

# Verify all APIs
gcloud services list --enabled | grep -E "(aiplatform|storage|logging|monitoring|compute|container|iam)"
```

#### 1.3 Set Up Billing
```bash
# Link billing account
gcloud beta billing projects link iq-llm-training \
    --billing-account=your-billing-account-id

# Verify billing
gcloud beta billing projects describe iq-llm-training
```

### Step 2: Service Account & Permissions

#### 2.1 Create Service Account
```bash
# Create service account
gcloud iam service-accounts create iq-training \
    --display-name="IQ Training Service Account" \
    --description="Service account for IQ LLM training on GCP" \
    --project=iq-llm-training

# Get service account email
SA_EMAIL="iq-training@iq-llm-training.iam.gserviceaccount.com"
echo "Service account email: $SA_EMAIL"
```

#### 2.2 Grant Required Permissions
```bash
# AI Platform permissions
gcloud projects add-iam-policy-binding iq-llm-training \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/aiplatform.user"

# Storage permissions
gcloud projects add-iam-policy-binding iq-llm-training \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/storage.admin"

# Logging permissions
gcloud projects add-iam-policy-binding iq-llm-training \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/logging.logWriter"

# Monitoring permissions
gcloud projects add-iam-policy-binding iq-llm-training \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/monitoring.metricWriter"

# Compute permissions
gcloud projects add-iam-policy-binding iq-llm-training \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/compute.admin"

# IAM permissions
gcloud projects add-iam-policy-binding iq-llm-training \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/iam.serviceAccountUser"

# Verify permissions
gcloud projects get-iam-policy iq-llm-training \
    --flatten="bindings[].members" \
    --format="table(bindings.role,bindings.members)" |
    grep "$SA_EMAIL"
```

#### 2.3 Create Service Account Key
```bash
# Create key for service account
gcloud iam service-accounts keys create ~/iq-training-key.json \
    --iam-account="$SA_EMAIL" \
    --project=iq-llm-training

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=~/iq-training-key.json
```

### Step 3: Storage Setup

#### 3.1 Create Storage Buckets
```bash
# Set region
REGION="us-central1"

# Create buckets
gsutil mb -l $REGION gs://iq-training-data
gsutil mb -l $REGION gs://iq-model-checkpoints
gsutil mb -l $REGION gs://iq-training-logs
gsutil mb -l $REGION gs://iq-model-artifacts

# Verify buckets
gsutil ls
```

#### 3.2 Configure Bucket Lifecycle
```bash
# Create lifecycle configuration
cat > lifecycle_config.json << EOF
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {
        "age": 30,
        "isLive": false
      }
    }
  ]
}
EOF

# Apply lifecycle to buckets
for bucket in gs://iq-training-data gs://iq-model-checkpoints gs://iq-training-logs gs://iq-model-artifacts; do
    gsutil lifecycle set lifecycle_config.json $bucket
done

# Verify lifecycle
gsutil lifecycle get gs://iq-training-data
```

#### 3.3 Set Bucket Permissions
```bash
# Grant service account access to buckets
for bucket in gs://iq-training-data gs://iq-model-checkpoints gs://iq-training-logs gs://iq-model-artifacts; do
    gsutil iam ch serviceAccount:$SA_EMAIL:objectAdmin $bucket
done

# Verify permissions
gsutil iam get gs://iq-training-data
```

### Step 4: Data Preparation

#### 4.1 Create Data Directory Structure
```bash
# Create local data directories
mkdir -p data/{training,validation,test}
mkdir -p data/{gneissweb,stack,mixed}

# Create sample data structure
cat > data/sample_config.json << EOF
{
  "datasets": {
    "gneissweb_2026": {
      "path": "data/gneissweb/",
      "format": "parquet",
      "weight": 0.7,
      "type": "text"
    },
    "stack_v3": {
      "path": "data/stack/",
      "format": "parquet",
      "weight": 0.3,
      "type": "code"
    }
  },
  "preprocessing": {
    "tokenization": true,
    "sequence_length": 8192,
    "batch_size": 32,
    "vocab_size": 50000
  }
}
EOF
```

#### 4.2 Upload Training Data
```bash
# Upload sample data (replace with actual data)
if [ -d "data/training" ] && [ "$(ls -A data/training)" ]; then
    echo "Uploading training data..."
    gsutil -m rsync -r data/training/ gs://iq-training-data/training/
    gsutil -m rsync -r data/validation/ gs://iq-training-data/validation/
    gsutil -m rsync -r data/test/ gs://iq-training-data/test/

    echo "Data upload completed"
    echo "Training data: $(gsutil du -sh gs://iq-training-data/training/ | cut -f1)"
    echo "Validation data: $(gsutil du -sh gs://iq-training-data/validation/ | cut -f1)"
    echo "Test data: $(gsutil du -sh gs://iq-training-data/test/ | cut -f1)"
else
    echo "Warning: No training data found in data/ directory"
    echo "Please upload your training data before proceeding"
fi
```

### Step 5: Container Build

#### 5.1 Create Dockerfile
```bash
# Create optimized Dockerfile
cat > Dockerfile << EOF
FROM us-docker.pkg.dev/deeplearning-platform-release/gpu.mojo:latest

# System dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    curl \\
    build-essential \\
    python3 \\
    python3-pip \\
    python3-dev \\
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Mojo SDK (if not in base image)
RUN curl --proto '=https' --tlsv1.2 -sSf https://get.modular.com | sh \\
    && /root/.modular/bin/modular auth mut_$(cat ~/.modular/token) \\
    && /root/.modular/bin/modular install mojo

# Set working directory
WORKDIR /workspace

# Copy source code
COPY . /workspace/

# Set permissions
RUN chmod +x /workspace/*.sh
RUN chmod +x /workspace/nif_sovereign/core/*.mojo

# Environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_DEBUG=INFO
ENV MOJO_PATH=/root/.modular/pkg/modular_mojo

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Default command
CMD ["python3", "/workspace/nif_sovereign/core/custom_training_logic.py"]
EOF
```

#### 5.2 Create Requirements File
```bash
# Create Python requirements
cat > requirements.txt << EOF
torch==2.4.0
torchvision==0.19.0
torchaudio==2.4.0
numpy==1.26.0
scipy==1.13.0
pandas==2.2.0
matplotlib==3.8.0
seaborn==0.13.0
wandb==0.17.0
tensorboard==2.16.0
cuda-quantum==0.3.0
nvidia-ml-py==12.535.133
accelerate==0.28.0
transformers==4.40.0
datasets==2.19.0
huggingface-hub==0.23.0
safetensors==0.4.0
bitsandbytes==0.43.0
peft==0.10.0
tqdm==4.66.0
requests==2.31.0
pyyaml==6.0.1
EOF
```

#### 5.3 Build and Push Container
```bash
# Build container
IMAGE_NAME="us-docker.pkg.dev/iq-llm-training/containers/iq-training:latest"
docker build -t $IMAGE_NAME .

# Configure Docker for GCP
gcloud auth configure-docker us-central1-docker.pkg.dev

# Push container
docker push $IMAGE_NAME

# Verify container
gcloud artifacts docker images list us-central1-docker.pkg.dev/iq-llt-training/containers/iq-training
```

### Step 6: Training Job Configuration

#### 6.1 Create Training Job Request
```bash
# Create training job configuration
cat > training_request.json << EOF
{
  "displayName": "IQ-NIF-Sovereign-Training",
  "jobSpec": {
    "workerPoolSpecs": [
      {
        "machineSpec": {
          "machineType": "a3-highgpu-8g",
          "acceleratorType": "NVIDIA_H200",
          "acceleratorCount": 8
        },
        "replicaCount": 1,
        "containerSpec": {
          "imageUri": "us-docker.pkg.dev/iq-llm-training/containers/iq-training:latest",
          "args": [
            "--config-path=/workspace/config/nif_config.mojo",
            "--data-path=/data/training",
            "--output-path=/output/model",
            "--checkpoint-path=/output/checkpoints",
            "--log-path=/output/logs",
            "--tensorboard-path=/output/tensorboard",
            "--wandb-project=IQ-NIF-Training",
            "--max-steps=1000000",
            "--batch-size=32",
            "--learning-rate=1e-4",
            "--save-interval=10000",
            "--eval-interval=5000"
          ],
          "env": [
            {"name": "MOJO_VERSION", "value": "0.26.2"},
            {"name": "CUDA_VERSION", "value": "12.4"},
            {"name": "CUDA_Q_VERSION", "value": "0.3.0"},
            {"name": "THUNDER_COMPUTE_ENDPOINT", "value": "https://api.thundercompute.com/v1"},
            {"name": "NVIDIA_H200_TARGET", "value": "nvidia-h200-remote"},
            {"name": "MODEL_NAME", "value": "IQ-NIF-Sovereign"},
            {"name": "BASE_MODEL", "value": "Gemma-4-26B-A4B"},
            {"name": "HIDDEN_DIM", "value": "4096"},
            {"name": "NUM_LAYERS", "value": "32"},
            {"name": "NUM_EXPERTS", "value": "3"},
            {"name": "BATCH_SIZE", "value": "32"},
            {"name": "SEQUENCE_LENGTH", "value": "8192"},
            {"name": "LEARNING_RATE", "value": "1e-4"},
            {"name": "MUON_MOMENTUM", "value": "0.9"},
            {"name": "GALORE_RANK", "value": "64"},
            {"name": "VERA_RANK", "value": "64"},
            {"name": "RIEMANNIAN_CURVATURE", "value": "0.1"},
            {"name": "NEUTRINO_OSCILLATION_DEPTH", "value": "8"},
            {"name": "ISING_ITERATIONS", "value": "10"},
            {"name": "MAX_STEPS", "value": "1000000"},
            {"name": "SAVE_INTERVAL", "value": "10000"},
            {"name": "EVAL_INTERVAL", "value": "5000"},
            {"name": "LOG_INTERVAL", "value": "100"},
            {"name": "WANDB_API_KEY", "value": "your-wandb-api-key"},
            {"name": "THUNDER_API_KEY", "value": "your-thunder-api-key"},
            {"name": "CUDA_Q_API_KEY", "value": "your-cuda-q-api-key"}
          ]
        }
      }
    ]
  },
  "serviceAccount": "iq-training@iq-llm-training.iam.gserviceaccount.com",
  "network": "projects/iq-llm-training/global/networks/default"
}
EOF
```

#### 6.2 Submit Training Job
```bash
# Submit training job
REGION="us-central1"
gcloud ai-platform custom-jobs create \
    --region=$REGION \
    --display-name="IQ-NIF-Sovereign-Training" \
    --config=training_request.json \
    --project=iq-llm-training

# Get job ID
JOB_ID=$(gcloud ai-platform custom-jobs list \
    --region=$REGION \
    --filter="displayName:IQ-NIF-Sovereign-Training" \
    --format="value(name)" \
    --limit=1 | cut -d'/' -f6)

echo "Training job submitted with ID: $JOB_ID"
echo "Monitor job: gcloud ai-platform custom-jobs describe $JOB_ID --region=$REGION"
```

### Step 7: Monitoring & Management

#### 7.1 Monitor Training Progress
```bash
# Monitor job status
while true; do
    STATUS=$(gcloud ai-platform custom-jobs describe $JOB_ID \
        --region=$REGION \
        --format="value(state)")

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job status: $STATUS"

    if [ "$STATUS" = "SUCCEEDED" ]; then
        echo "Training completed successfully!"
        break
    elif [ "$STATUS" = "FAILED" ]; then
        echo "Training failed!"
        gcloud ai-platform custom-jobs describe $JOB_ID --region=$REGION
        break
    elif [ "$STATUS" = "CANCELLED" ]; then
        echo "Training cancelled!"
        break
    fi

    sleep 300  # Check every 5 minutes
done
```

#### 7.2 View Logs
```bash
# View training logs
gcloud ai-platform custom-jobs stream-logs $JOB_ID --region=$REGION

# View logs in Cloud Logging
gcloud logging read "resource.type=aiplatform.googleapis.com/custom_job" \
    --limit=50 \
    --format="table(timestamp,textPayload)"
```

#### 7.3 Monitor Resources
```bash
# Monitor GPU utilization
gcloud compute instances describe iq-training-node \
    --format="table(name,machineType,status)" \
    --zone=$ZONE

# Monitor storage usage
gsutil du -sh gs://iq-model-checkpoints/
gsutil du -sh gs://iq-training-logs/

# Monitor costs
gcloud billing budgets list --billing-account=your-billing-account-id
```

### Step 8: Results & Download

#### 8.1 Download Model Checkpoints
```bash
# Create local directories
mkdir -p output/{checkpoints,logs,model}

# Download checkpoints
gsutil -m rsync -r gs://iq-model-checkpoints/nif-sovereign/ output/checkpoints/

# Download logs
gsutil -m rsync -r gs://iq-training-logs/nif-sovereign/ output/logs/

# Download final model
gsutil -m rsync -r gs://iq-model-artifacts/nif-sovereign/ output/model/

# Verify downloads
echo "Checkpoints: $(ls -la output/checkpoints/ | wc -l) files"
echo "Logs: $(ls -la output/logs/ | wc -l) files"
echo "Model: $(ls -la output/model/ | wc -l) files"
```

#### 8.2 Evaluate Model Performance
```bash
# Run evaluation
python3 -c "
import torch
from nif_sovereign.core.custom_llm_architecture import NIFCustomLLM
from nif_sovereign.config.nif_config import NIFConfig

# Load model
config = NIFConfig()
model = NIFCustomLLM(config)
model.load_state_dict(torch.load('output/model/final_model.pth'))
model.eval()

# Test inference
input_text = 'Hello, I am IQ, your personal multimodal LLM.'
output = model.generate(input_text, max_length=100)
print('Generated text:', output)
"
```

### Step 9: Cleanup & Cost Management

#### 9.1 Clean Up Resources
```bash
# Delete training job
gcloud ai-platform custom-jobs delete $JOB_ID --region=$REGION

# Clean up temporary files
rm -f training_request.json lifecycle_config.json Dockerfile requirements.txt

# Optional: Delete service account key
rm -f ~/iq-training-key.json
```

#### 9.2 Cost Review
```bash
# Review cost breakdown
gcloud billing budgets list --billing-account=your-billing-account-id

# View cost breakdown by service
gcloud billing accounts get-usage-export \
    --billing-account=your-billing-account-id \
    --start-date=2026-01-01 \
    --end-date=2026-01-31

# Generate cost report
echo "Training Cost Summary:"
echo "======================"
echo "Compute: ~\$500,000 - \$1,000,000"
echo "Storage: ~\$416"
echo "Network: ~\$15,000"
echo "Total: ~\$515,416 - \$1,015,416"
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### GPU Memory Issues
```bash
# Reduce batch size in training_request.json
"args": ["--batch-size=16", "--gradient-accumulation-steps=8"]

# Enable gradient checkpointing
"args": ["--gradient-checkpointing=true"]

# Use mixed precision
"args": ["--mixed-precision=true"]
```

#### Network Issues
```bash
# Check network connectivity
ping -c 3 8.8.8.8

# Test GCP connectivity
gcloud auth list
gcloud config list

# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

#### Storage Issues
```bash
# Check bucket permissions
gsutil iam get gs://iq-training-data

# Test data access
gsutil cat gs://iq-training-data/sample.txt

# Re-upload data if needed
gsutil -m rsync -r data/training/ gs://iq-training-data/training/
```

#### Training Issues
```bash
# Check training logs
gcloud ai-platform custom-jobs stream-logs $JOB_ID --region=$REGION

# Get detailed job information
gcloud ai-platform custom-jobs describe $JOB_ID --region=$REGION

# Restart job if needed
gcloud ai-platform custom-jobs delete $JOB_ID --region=$REGION
# Then resubmit with updated configuration
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] GCP project created and configured
- [ ] All required APIs enabled
- [ ] Service account with proper permissions
- [ ] Storage buckets created and configured
- [ ] Training data uploaded and validated
- [ ] Container built and pushed
- [ ] Training job configuration validated
- [ ] Budget alerts configured
- [ ] Monitoring and logging set up

### Post-Deployment
- [ ] Training job submitted successfully
- [ ] Training progress monitored
- [ ] Model checkpoints downloaded
- [ ] Training logs collected
- [ ] Model evaluation completed
- [ ] Results documented
- [ ] Resources cleaned up
- [ ] Cost review completed

## 🚀 Next Steps

After successful deployment and training:

1. **Model Evaluation**: Test the trained model on validation data
2. **Performance Optimization**: Fine-tune hyperparameters if needed
3. **Deployment**: Deploy model to production environment
4. **Monitoring**: Set up ongoing monitoring for model performance
5. **Maintenance**: Regular updates and retraining as needed

---

**IMPORTANT**: This is proprietary software. All training configurations, model weights, and generated artifacts are confidential and belong to [Your Name]. Unauthorized distribution or modification is strictly prohibited.

For support, contact: [your.email@example.com]
