#!/bin/bash

# Google Cloud Training Script for IQ LLM
# Proprietary NIF Sovereign Architecture Training
# Copyright © 2026 [Your Name]. All Rights Reserved.

set -e

# Configuration
PROJECT_ID="iq-llm-training"
REGION="us-central1"
ZONE="us-central1-a"
JOB_NAME="iq-nif-sovereign-training"
CONFIG_FILE="gcp_training_config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if gcloud is installed
check_gcloud() {
    log "Checking Google Cloud SDK installation..."
    if ! command -v gcloud &> /dev/null; then
        error "Google Cloud SDK not found. Please install gcloud CLI."
    fi
    success "Google Cloud SDK found"
}

# Authenticate with Google Cloud
authenticate() {
    log "Authenticating with Google Cloud..."
    gcloud auth login
    gcloud config set project $PROJECT_ID
    success "Authentication completed"
}

# Enable required APIs
enable_apis() {
    log "Enabling required Google Cloud APIs..."

    apis=(
        "aiplatform.googleapis.com"
        "storage.googleapis.com"
        "logging.googleapis.com"
        "monitoring.googleapis.com"
        "iam.googleapis.com"
        "compute.googleapis.com"
        "container.googleapis.com"
    )

    for api in "${apis[@]}"; do
        log "Enabling $api..."
        gcloud services enable $api --project=$PROJECT_ID
    done

    success "All APIs enabled"
}

# Create service account
create_service_account() {
    log "Creating service account..."

    SA_NAME="iq-training"
    SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

    # Create service account if it doesn't exist
    if ! gcloud iam service-accounts describe $SA_EMAIL --project=$PROJECT_ID &> /dev/null; then
        gcloud iam service-accounts create $SA_NAME \
            --display-name="IQ Training Service Account" \
            --description="Service account for IQ LLM training" \
            --project=$PROJECT_ID
        success "Service account created"
    else
        warning "Service account already exists"
    fi

    # Grant required permissions
    log "Granting permissions to service account..."

    roles=(
        "roles/aiplatform.user"
        "roles/storage.admin"
        "roles/logging.logWriter"
        "roles/monitoring.metricWriter"
        "roles/iam.serviceAccountUser"
        "roles/compute.admin"
    )

    for role in "${roles[@]}"; do
        gcloud projects add-iam-policy-binding $PROJECT_ID \
            --member="serviceAccount:$SA_EMAIL" \
            --role="$role" \
            --condition=None
    done

    success "Service account permissions granted"
}

# Create storage buckets
create_storage_buckets() {
    log "Creating storage buckets..."

    buckets=(
        "gs://iq-training-data"
        "gs://iq-model-checkpoints"
        "gs://iq-training-logs"
        "gs://iq-model-artifacts"
    )

    for bucket in "${buckets[@]}"; do
        if ! gsutil ls $bucket &> /dev/null; then
            log "Creating bucket: $bucket"
            gsutil mb -l $REGION $bucket
            gsutil uniformbucketlevelaccess set on $bucket
            gsutil lifecycle set lifecycle_config.json $bucket
        else
            warning "Bucket $bucket already exists"
        fi
    done

    success "Storage buckets ready"
}

# Upload training data
upload_training_data() {
    log "Uploading training data..."

    # Create data directories
    mkdir -p data/{training,validation,test}

    # Upload sample data (replace with actual data)
    if [ -d "data/training" ] && [ "$(ls -A data/training)" ]; then
        gsutil -m rsync -r data/training/ gs://iq-training-data/training/
        gsutil -m rsync -r data/validation/ gs://iq-training-data/validation/
        gsutil -m rsync -r data/test/ gs://iq-training-data/test/
        success "Training data uploaded"
    else
        warning "No training data found in data/ directory"
        log "Please upload your training data to data/ directory"
    fi
}

# Create lifecycle configuration
create_lifecycle_config() {
    log "Creating lifecycle configuration..."

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

    success "Lifecycle configuration created"
}

# Build training container
build_container() {
    log "Building training container..."

    cat > Dockerfile << EOF
FROM us-docker.pkg.dev/deeplearning-platform-release/gpu.mojo:latest

# Install dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    curl \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install \\
    torch \\
    torchvision \\
    torchaudio \\
    numpy \\
    scipy \\
    pandas \\
    matplotlib \\
    seaborn \\
    wandb \\
    tensorboard \\
    cuda-quantum

# Set working directory
WORKDIR /workspace

# Copy source code
COPY . /workspace/

# Set permissions
RUN chmod +x /workspace/*.sh

# Environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_DEBUG=INFO

# Entry point
ENTRYPOINT ["python3", "/workspace/nif_sovereign/core/custom_training_logic.py"]
EOF

    # Build and push container
    IMAGE_NAME="us-docker.pkg.dev/$PROJECT_ID/containers/iq-training:latest"
    docker build -t $IMAGE_NAME .
    docker push $IMAGE_NAME

    success "Container built and pushed: $IMAGE_NAME"
}

# Submit training job
submit_training_job() {
    log "Submitting training job..."

    # Create training job request
    cat > training_request.json << EOF
{
  "displayName": "$JOB_NAME",
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
          "imageUri": "us-docker.pkg.dev/$PROJECT_ID/containers/iq-training:latest",
          "args": [
            "--config-path=/workspace/config/nif_config.mojo",
            "--data-path=/data/training",
            "--output-path=/output/model",
            "--checkpoint-path=/output/checkpoints",
            "--log-path=/output/logs",
            "--tensorboard-path=/output/tensorboard"
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
            {"name": "LOG_INTERVAL", "value": "100"}
          ]
        }
      }
    ]
  },
  "serviceAccount": "iq-training@$PROJECT_ID.iam.gserviceaccount.com"
}
EOF

    # Submit job
    gcloud ai-platform custom-jobs create \
        --region=$REGION \
        --display-name="$JOB_NAME" \
        --config=training_request.json \
        --project=$PROJECT_ID

    success "Training job submitted"
}

# Monitor training job
monitor_job() {
    log "Monitoring training job..."

    # Get job ID
    JOB_ID=$(gcloud ai-platform custom-jobs list \
        --region=$REGION \
        --filter="displayName:$JOB_NAME" \
        --format="value(name)" \
        --limit=1 | cut -d'/' -f6)

    if [ -z "$JOB_ID" ]; then
        error "Could not find training job"
    fi

    log "Job ID: $JOB_ID"

    # Monitor job status
    while true; do
        STATUS=$(gcloud ai-platform custom-jobs describe \
            $JOB_ID \
            --region=$REGION \
            --format="value(state)")

        log "Job status: $STATUS"

        if [ "$STATUS" = "SUCCEEDED" ]; then
            success "Training completed successfully"
            break
        elif [ "$STATUS" = "FAILED" ]; then
            error "Training failed"
        elif [ "$STATUS" = "CANCELLED" ]; then
            error "Training cancelled"
        fi

        sleep 60
    done
}

# Download results
download_results() {
    log "Downloading training results..."

    # Download checkpoints
    mkdir -p output/checkpoints
    gsutil -m rsync -r gs://iq-model-checkpoints/nif-sovereign/ output/checkpoints/

    # Download logs
    mkdir -p output/logs
    gsutil -m rsync -r gs://iq-training-logs/nif-sovereign/ output/logs/

    # Download model artifacts
    mkdir -p output/model
    gsutil -m rsync -r gs://iq-model-artifacts/nif-sovereign/ output/model/

    success "Results downloaded to output/ directory"
}

# Cleanup
cleanup() {
    log "Cleaning up resources..."

    # Remove temporary files
    rm -f training_request.json lifecycle_config.json Dockerfile

    success "Cleanup completed"
}

# Main execution
main() {
    log "Starting IQ LLM training setup..."

    # Check prerequisites
    check_gcloud

    # Setup
    authenticate
    enable_apis
    create_service_account
    create_storage_buckets
    create_lifecycle_config

    # Data preparation
    upload_training_data

    # Container build
    build_container

    # Training
    submit_training_job
    monitor_job

    # Results
    download_results

    # Cleanup
    cleanup

    success "IQ LLM training pipeline completed!"
}

# Help function
help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -s, --setup-only        Run setup only (no training)"
    echo "  -t, --train-only        Run training only (skip setup)"
    echo "  -m, --monitor-only      Monitor existing job only"
    echo "  -d, --download-only     Download results only"
    echo "  -c, --cleanup-only      Cleanup resources only"
    echo ""
    echo "Examples:"
    echo "  $0                      # Run full pipeline"
    echo "  $0 --setup-only         # Setup only"
    echo "  $0 --train-only         # Training only"
    echo "  $0 --monitor-only       # Monitor existing job"
    echo "  $0 --download-only      # Download results only"
    echo "  $0 --cleanup-only       # Cleanup resources only"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        help
        exit 0
        ;;
    -s|--setup-only)
        check_gcloud
        authenticate
        enable_apis
        create_service_account
        create_storage_buckets
        create_lifecycle_config
        upload_training_data
        build_container
        success "Setup completed"
        exit 0
        ;;
    -t|--train-only)
        submit_training_job
        monitor_job
        success "Training completed"
        exit 0
        ;;
    -m|--monitor-only)
        monitor_job
        exit 0
        ;;
    -d|--download-only)
        download_results
        cleanup
        exit 0
        ;;
    -c|--cleanup-only)
        cleanup
        exit 0
        ;;
    "")
        main
        ;;
    *)
        error "Unknown option: $1"
        ;;
esac
