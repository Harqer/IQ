# Embedding Interface
# Defines contract for embedding mechanisms
# Enables dependency injection and testability

from tensor import Tensor

trait EmbeddingInterface:
    # Core embedding computation
    fn embed_input(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]
    
    # Position encoding
    fn add_positional_encoding(self, embeddings: Tensor[DType.float32]) -> Tensor[DType.float32]
    
    # Manifold transformation
    fn transform_manifold(self, embeddings: Tensor[DType.float32]) -> Tensor[DType.float32]
    
    # Get embedding configuration
    fn get_config(self) -> EmbeddingConfig

# Embedding configuration
struct EmbeddingConfig:
    var embedding_dim: Int
    var max_sequence_length: Int
    var manifold_curvature: Float64
    var use_positional_encoding: Bool
    
    fn __init__(embedding_dim: Int, max_sequence_length: Int, manifold_curvature: Float64 = 0.1, use_positional_encoding: Bool = true):
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.manifold_curvature = manifold_curvature
        self.use_positional_encoding = use_positional_encoding
