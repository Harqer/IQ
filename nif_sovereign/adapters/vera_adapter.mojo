# VeRA Adapter: Vector-based Random Matrix Adaptation
# Parameter-efficient fine-tuning with trainable scaling vectors only
# Minimizes VRAM usage while maintaining model performance

from tensor import Tensor
from math import sqrt, exp

struct VeRAAdapter:
    var config: NIFConfig
    var hidden_dim: Int
    var vera_rank: Int
    var scaling_vectors: Tensor[DType.float32]
    var random_matrices: Tensor[DType.float32]
    var bias_vectors: Tensor[DType.float32]
    var learning_rate: Float32

    fn __init__(inout self, config: NIFConfig):
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.vera_rank = config.vera_rank
        self.learning_rate = 0.001

        # Initialize VeRA components
        self.scaling_vectors = self.initialize_scaling_vectors()
        self.random_matrices = self.initialize_random_matrices()
        self.bias_vectors = self.initialize_bias_vectors()

        print("🎯 VeRA Adapter Initialized")
        print("   - Rank: {}".format(self.vera_rank))
        print("   - Trainable Parameters: {}".format(self.count_trainable_params()))
        print("   - VRAM Savings: {:.1%}".format(self.compute_vram_savings()))

    @parameter
    fn initialize_scaling_vectors() -> Tensor[DType.float32]:
        # Initialize trainable scaling vectors
        var scaling = Tensor[DType.float32](self.hidden_dim, self.vera_rank)

        for i in range(self.hidden_dim):
            for j in range(self.vera_rank):
                # Small random initialization
                scaling[i, j] = 0.01 * (Float32(i + j) - Float32(self.hidden_dim) / 2.0) / Float32(self.hidden_dim)

        return scaling

    @parameter
    fn initialize_random_matrices() -> Tensor[DType.float32]:
        # Initialize fixed random matrices (not trainable)
        var random_mats = Tensor[DType.float32](self.vera_rank, self.hidden_dim)

        for i in range(self.vera_rank):
            for j in range(self.hidden_dim):
                # Fixed random initialization (Gaussian-like)
                var theta = 2.0 * 3.14159 * Float32(i) / Float32(self.vera_rank)
                random_mats[i, j] = cos(theta + Float32(j))

        return random_mats

    @parameter
    fn initialize_bias_vectors() -> Tensor[DType.float32]:
        # Initialize trainable bias vectors
        var bias = Tensor[DType.float32](self.hidden_dim)

        for i in range(self.hidden_dim):
            bias[i] = 0.0  # Zero initialization for bias

        return bias

    fn apply_scaling(inout self, input_tensor: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # Apply VeRA scaling transformation to input tensor
        var shape = input_tensor.shape()
        var batch_size = shape[0]
        var seq_len = shape[1]
        var output = Tensor[DType.float32](batch_size, seq_len, self.hidden_dim)

        for b in range(batch_size):
            for s in range(seq_len):
                # Apply VeRA transformation: y = x + S * R * x + b
                var input_vector = input_tensor[b, s, :]
                var transformed = self.vera_transform(input_vector)

                for dim in range(self.hidden_dim):
                    output[b, s, dim] = transformed[dim]

        return output

    fn vera_transform(inout self, input_vector: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # Core VeRA transformation: y = x + S * R * x + b
        var transformed = Tensor[DType.float32](self.hidden_dim)

        # Compute R * x (random matrix multiplication)
        var rx = Tensor[DType.float32](self.vera_rank)
        for i in range(self.vera_rank):
            var sum = 0.0
            for j in range(self.hidden_dim):
                sum += self.random_matrices[i, j] * input_vector[j]
            rx[i] = sum

        # Compute S * (R * x) (scaling vector multiplication)
        var srx = Tensor[DType.float32](self.hidden_dim)
        for i in range(self.hidden_dim):
            var sum = 0.0
            for j in range(self.vera_rank):
                sum += self.scaling_vectors[i, j] * rx[j]
            srx[i] = sum

        # Final transformation: y = x + S * R * x + b
        for i in range(self.hidden_dim):
            transformed[i] = input_vector[i] + srx[i] + self.bias_vectors[i]

        return transformed

    fn update_scaling_vectors(inout self, gradients: Tensor[DType.float32]):
        # Update trainable scaling vectors using gradients
        for i in range(self.hidden_dim):
            for j in range(self.vera_rank):
                # Simple gradient descent update
                self.scaling_vectors[i, j] -= self.learning_rate * gradients[i, j]

    fn update_bias_vectors(inout self, gradients: Tensor[DType.float32]):
        # Update trainable bias vectors using gradients
        for i in range(self.hidden_dim):
            self.bias_vectors[i] -= self.learning_rate * gradients[i]

    fn compute_gradients(inout self, input_tensor: Tensor[DType.float32], output_gradients: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # Compute gradients for VeRA parameters
        var shape = input_tensor.shape()
        var batch_size = shape[0]
        var seq_len = shape[1]
        var scaling_gradients = Tensor[DType.float32](self.hidden_dim, self.vera_rank)
        var bias_gradients = Tensor[DType.float32](self.hidden_dim)

        # Accumulate gradients over batch
        for b in range(batch_size):
            for s in range(seq_len):
                var input_vector = input_tensor[b, s, :]
                var output_grad = output_gradients[b, s, :]

                # Compute gradients for bias (direct)
                for i in range(self.hidden_dim):
                    bias_gradients[i] += output_grad[i]

                # Compute gradients for scaling vectors (backprop through R)
                var rx = Tensor[DType.float32](self.vera_rank)
                for i in range(self.vera_rank):
                    var sum = 0.0
                    for j in range(self.hidden_dim):
                        sum += self.random_matrices[i, j] * input_vector[j]
                    rx[i] = sum

                for i in range(self.hidden_dim):
                    for j in range(self.vera_rank):
                        scaling_gradients[i, j] += output_grad[i] * rx[j]

        # Normalize gradients
        var normalization = Float32(batch_size * seq_len)
        for i in range(self.hidden_dim):
            bias_gradients[i] /= normalization
            for j in range(self.vera_rank):
                scaling_gradients[i, j] /= normalization

        return scaling_gradients

    def count_trainable_params(inout self) -> Int:
        # Count number of trainable parameters
        var scaling_params = self.hidden_dim * self.vera_rank
        var bias_params = self.hidden_dim
        return scaling_params + bias_params

    def compute_vram_savings(inout self) -> Float32:
        # Compute VRAM savings compared to full fine-tuning
        var full_params = self.hidden_dim * self.hidden_dim  # Full matrix
        var vera_params = self.count_trainable_params()
        return 1.0 - Float32(vera_params) / Float32(full_params)

    def reset_adapters(inout self):
        # Reset adapter parameters to initial state
        for i in range(self.hidden_dim):
            self.bias_vectors[i] = 0.0
            for j in range(self.vera_rank):
                self.scaling_vectors[i, j] = 0.01 * (Float32(i + j) - Float32(self.hidden_dim) / 2.0) / Float32(self.hidden_dim)

    def get_adapter_statistics(inout self) -> Tensor[Float32]:
        # Get adapter statistics for monitoring
        var stats = Tensor[Float32](4)

        # Compute scaling vector norms
        var scaling_norm = 0.0
        var bias_norm = 0.0

        for i in range(self.hidden_dim):
            bias_norm += self.bias_vectors[i] * self.bias_vectors[i]
            for j in range(self.vera_rank):
                scaling_norm += self.scaling_vectors[i, j] * self.scaling_vectors[i, j]

        stats[0] = sqrt(scaling_norm)  # Scaling vector norm
        stats[1] = sqrt(bias_norm)     # Bias vector norm
        stats[2] = Float32(self.count_trainable_params())  # Parameter count
        stats[3] = self.compute_vram_savings()  # VRAM savings

        return stats

    def apply_layer_normalization(inout self, input_tensor: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # Apply layer normalization before VeRA transformation
        var shape = input_tensor.shape()
        var batch_size = shape[0]
        var seq_len = shape[1]
        var normalized = Tensor[DType.float32](batch_size, seq_len, self.hidden_dim)

        for b in range(batch_size):
            for s in range(seq_len):
                # Compute mean and variance
                var mean = 0.0
                var variance = 0.0

                for i in range(self.hidden_dim):
                    mean += input_tensor[b, s, i]
                mean /= Float32(self.hidden_dim)

                for i in range(self.hidden_dim):
                    variance += (input_tensor[b, s, i] - mean) ** 2
                variance /= Float32(self.hidden_dim)

                var std = sqrt(variance + 1e-6)  # Small epsilon for stability

                # Normalize
                for i in range(self.hidden_dim):
                    normalized[b, s, i] = (input_tensor[b, s, i] - mean) / std

        return normalized
