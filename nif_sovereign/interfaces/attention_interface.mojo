# Attention Interface
# Defines contract for attention mechanisms
# Enables dependency injection and testability

from tensor import Tensor

trait AttentionInterface:
    # Core attention computation
    fn compute_attention(self, input: Tensor[DType.float32], mask: Tensor[DType.float32]) -> Tensor[DType.float32]
    
    # Attention scoring
    fn compute_scores(self, queries: Tensor[DType.float32], keys: Tensor[DType.float32]) -> Tensor[DType.float32]
    
    # Attention weights application
    fn apply_weights(self, weights: Tensor[DType.float32], values: Tensor[DType.float32]) -> Tensor[DType.float32]
    
    # Get attention configuration
    fn get_config(self) -> AttentionConfig

# Attention configuration
struct AttentionConfig:
    var num_heads: Int
    var head_dim: Int
    var hidden_dim: Int
    var dropout_rate: Float32
    
    fn __init__(num_heads: Int, head_dim: Int, hidden_dim: Int, dropout_rate: Float32 = 0.1):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
