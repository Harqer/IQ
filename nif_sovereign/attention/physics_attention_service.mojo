# Physics Attention Service
# Atomic component for physics-aware attention
# Single responsibility: attention computation with physics integration

from tensor import Tensor
from math import sqrt, exp, tanh, sin, cos, abs, acos
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.interfaces.attention_interface import AttentionInterface, AttentionConfig
from nif_sovereign.interfaces.embedding_interface import EmbeddingInterface
from nif_sovereign.modules.riemannian_embedding import LorentzianEmbedding

# Physics-aware projection matrix
struct PhysicsProjectionMatrix:
    var matrix: Tensor[DType.float32]
    var manifold_curvature: Float64
    
    fn __init__(hidden_dim: Int, manifold_curvature: Float64):
        self.manifold_curvature = manifold_curvature
        self.matrix = self.initialize_physics_aware_matrix(hidden_dim)
    
    fn initialize_physics_aware_matrix(self, hidden_dim: Int) -> Tensor[DType.float32]:
        """Initialize projection matrix with physics-aware initialization"""
        var projection = Tensor[DType.float32](hidden_dim, hidden_dim)
        
        for i in range(hidden_dim):
            for j in range(hidden_dim):
                # Physics-aware initialization based on manifold curvature
                var curvature_factor = 1.0 / (1.0 + self.manifold_curvature * Float32(i) * Float32(j))
                projection[i, j] = curvature_factor * sin(Float32(i + j))
        
        return projection

# Multi-head projection processor
struct MultiHeadProcessor:
    var num_heads: Int
    var head_dim: Int
    var hidden_dim: Int
    var query_projection: PhysicsProjectionMatrix
    var key_projection: PhysicsProjectionMatrix
    var value_projection: PhysicsProjectionMatrix
    var output_projection: PhysicsProjectionMatrix
    
    fn __init__(num_heads: Int, head_dim: Int, hidden_dim: Int, manifold_curvature: Float64):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.query_projection = PhysicsProjectionMatrix(hidden_dim, manifold_curvature)
        self.key_projection = PhysicsProjectionMatrix(hidden_dim, manifold_curvature)
        self.value_projection = PhysicsProjectionMatrix(hidden_dim, manifold_curvature)
        self.output_projection = PhysicsProjectionMatrix(hidden_dim, manifold_curvature)
    
    fn project_to_queries(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Project input to query space"""
        return self.apply_projection(input, self.query_projection.matrix)
    
    fn project_to_keys(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Project input to key space"""
        return self.apply_projection(input, self.key_projection.matrix)
    
    fn project_to_values(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Project input to value space"""
        return self.apply_projection(input, self.value_projection.matrix)
    
    fn project_output(self, attention_output: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Project attention output"""
        return self.apply_projection(attention_output, self.output_projection.matrix)
    
    fn apply_projection(self, input: Tensor[DType.float32], projection: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply projection matrix to input"""
        var shape = input.shape()
        var output = Tensor[DType.float32](shape)
        
        for b in range(shape[0]):
            for s in range(shape[1]):
                for i in range(shape[2]):
                    var sum = 0.0
                    for j in range(shape[2]):
                        sum += input[b, s, j] * projection[j, i]
                    output[b, s, i] = sum
        
        return output

# Attention score calculator
struct AttentionScoreCalculator:
    var head_dim: Int
    var manifold_curvature: Float64
    
    fn __init__(head_dim: Int, manifold_curvature: Float64):
        self.head_dim = head_dim
        self.manifold_curvature = manifold_curvature
    
    fn compute_scores(self, queries: Tensor[DType.float32], keys: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Compute attention scores with manifold awareness"""
        var shape = queries.shape()
        var batch_size = shape[0]
        var num_heads = shape[1]
        var seq_len = shape[2]
        
        var scores = Tensor[DType.float32](batch_size, num_heads, seq_len, seq_len)
        
        for b in range(batch_size):
            for h in range(num_heads):
                for i in range(seq_len):
                    for j in range(seq_len):
                        # Standard attention score
                        var score = 0.0
                        for d in range(self.head_dim):
                            score += queries[b, h, i, d] * keys[b, h, j, d]
                        
                        # Scale by sqrt(head_dim)
                        score /= sqrt(Float32(self.head_dim))
                        
                        # Apply physics-aware scaling based on manifold distance
                        var physics_factor = self.compute_physics_factor(i, j)
                        score *= physics_factor
                        
                        scores[b, h, i, j] = score
        
        return scores
    
    fn compute_physics_factor(self, i: Int, j: Int) -> Float32:
        """Compute physics-aware scaling factor"""
        var distance = abs(Float32(i - j))
        var manifold_factor = exp(-self.manifold_curvature * distance)
        return manifold_factor

# Attention weight applicator
struct AttentionWeightApplicator:
    fn apply_weights(self, weights: Tensor[DType.float32], values: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply attention weights to values"""
        var shape = weights.shape()
        var batch_size = shape[0]
        var num_heads = shape[1]
        var seq_len = shape[2]
        
        # Apply softmax to weights
        var attention_weights = self.softmax(weights)
        
        # Apply attention to values
        var output = Tensor[DType.float32](batch_size, num_heads, seq_len, self.head_dim)
        
        for b in range(batch_size):
            for h in range(num_heads):
                for i in range(seq_len):
                    for d in range(self.head_dim):
                        var sum = 0.0
                        for j in range(seq_len):
                            sum += attention_weights[b, h, i, j] * values[b, h, j, d]
                        output[b, h, i, d] = sum
        
        return output
    
    fn softmax(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply softmax function"""
        var shape = input.shape()
        var output = Tensor[DType.float32](shape)
        
        for b in range(shape[0]):
            for h in range(shape[1]):
                for i in range(shape[2]):
                    # Find max for numerical stability
                    var max_val = input[b, h, i, 0]
                    for j in range(shape[3]):
                        if input[b, h, i, j] > max_val:
                            max_val = input[b, h, i, j]
                    
                    # Compute exp and sum
                    var sum_exp = 0.0
                    for j in range(shape[3]):
                        var exp_val = exp(input[b, h, i, j] - max_val)
                        output[b, h, i, j] = exp_val
                        sum_exp += exp_val
                    
                    # Normalize
                    if sum_exp > 0.0:
                        for j in range(shape[3]):
                            output[b, h, i, j] /= sum_exp
        
        return output

# Physics Attention Service Implementation
struct PhysicsAttentionService:
    var config: AttentionConfig
    var embedding_service: EmbeddingInterface
    var processor: MultiHeadProcessor
    var score_calculator: AttentionScoreCalculator
    var weight_applicator: AttentionWeightApplicator
    
    fn __init__(config: SystemConfig, embedding_service: EmbeddingInterface):
        self.config = AttentionConfig(32, config.hidden_dim // 32, config.hidden_dim, 0.1)
        self.embedding_service = embedding_service
        self.processor = MultiHeadProcessor(self.config.num_heads, self.config.head_dim, self.config.hidden_dim, config.manifold_curvature)
        self.score_calculator = AttentionScoreCalculator(self.config.head_dim, config.manifold_curvature)
        self.weight_applicator = AttentionWeightApplicator()
        
        print("🧠 Physics Attention Service Initialized")
        print("   - Heads: {}".format(self.config.num_heads))
        print("   - Head Dimension: {}".format(self.config.head_dim))
        print("   - Hidden Dimension: {}".format(self.config.hidden_dim))
    
    fn compute_attention(self, input: Tensor[DType.float32], mask: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Compute physics-aware attention"""
        var shape = input.shape()
        var batch_size = shape[0]
        var seq_len = shape[1]
        
        # Step 1: Apply embedding service for manifold awareness
        var embedded_input = self.embedding_service.transform_manifold(input)
        
        # Step 2: Project to Q, K, V
        var queries = self.processor.project_to_queries(embedded_input)
        var keys = self.processor.project_to_keys(embedded_input)
        var values = self.processor.project_to_values(embedded_input)
        
        # Step 3: Reshape for multi-head attention
        var reshaped_queries = self.reshape_for_multihead(queries, batch_size, seq_len)
        var reshaped_keys = self.reshape_for_multihead(keys, batch_size, seq_len)
        var reshaped_values = self.reshape_for_multihead(values, batch_size, seq_len)
        
        # Step 4: Compute attention scores
        var attention_scores = self.score_calculator.compute_scores(reshaped_queries, reshaped_keys)
        
        # Step 5: Apply mask (if provided)
        if mask.shape()[0] > 0:
            attention_scores = self.apply_mask(attention_scores, mask)
        
        # Step 6: Apply attention weights
        var attention_output = self.weight_applicator.apply_weights(attention_scores, reshaped_values)
        
        # Step 7: Reshape back and project output
        var final_output = self.reshape_from_multihead(attention_output, batch_size, seq_len)
        final_output = self.processor.project_output(final_output)
        
        return final_output
    
    fn compute_scores(self, queries: Tensor[DType.float32], keys: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Compute attention scores"""
        return self.score_calculator.compute_scores(queries, keys)
    
    fn apply_weights(self, weights: Tensor[DType.float32], values: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply attention weights to values"""
        return self.weight_applicator.apply_weights(weights, values)
    
    fn get_config(self) -> AttentionConfig:
        """Get attention configuration"""
        return self.config
    
    fn reshape_for_multihead(self, input: Tensor[DType.float32], batch_size: Int, seq_len: Int) -> Tensor[DType.float32]:
        """Reshape input for multi-head attention"""
        var reshaped = Tensor[DType.float32](batch_size, self.config.num_heads, seq_len, self.config.head_dim)
        
        for b in range(batch_size):
            for h in range(self.config.num_heads):
                for s in range(seq_len):
                    for d in range(self.config.head_dim):
                        var idx = h * self.config.head_dim + d
                        reshaped[b, h, s, d] = input[b, s, idx]
        
        return reshaped
    
    fn reshape_from_multihead(self, input: Tensor[DType.float32], batch_size: Int, seq_len: Int) -> Tensor[DType.float32]:
        """Reshape from multi-head attention"""
        var reshaped = Tensor[DType.float32](batch_size, seq_len, self.config.hidden_dim)
        
        for b in range(batch_size):
            for s in range(seq_len):
                for h in range(self.config.num_heads):
                    for d in range(self.config.head_dim):
                        var idx = h * self.config.head_dim + d
                        reshaped[b, s, idx] = input[b, h, s, d]
        
        return reshaped
    
    fn apply_mask(self, scores: Tensor[DType.float32], mask: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply attention mask"""
        var shape = scores.shape()
        var masked = Tensor[DType.float32](shape)
        
        for b in range(shape[0]):
            for h in range(shape[1]):
                for i in range(shape[2]):
                    for j in range(shape[3]):
                        masked[b, h, i, j] = scores[b, h, i, j] + mask[b, i, j]
        
        return masked
