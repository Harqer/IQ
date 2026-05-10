# Gating Interface
# Defines contract for gating network operations in MoE systems
# Enables dependency injection and testability for gating components

from tensor import Tensor

trait GatingInterface:
    # Core gating operations
    fn compute_gating_scores(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]
    
    # Score processing
    fn apply_softmax(self, scores: Tensor[DType.float32]) -> Tensor[DType.float32]
    fn apply_temperature_scaling(self, scores: Tensor[DType.float32], temperature: Float32) -> Tensor[DType.float32]
    
    # Gating configuration
    fn get_gating_config(self) -> GatingConfig
    fn get_num_experts(self) -> Int

# Gating configuration
struct GatingConfig:
    var num_experts: Int
    var hidden_dim: Int
    var temperature: Float32
    var use_top_k: Bool
    var top_k: Int
    
    fn __init__(num_experts: Int, hidden_dim: Int, temperature: Float32 = 1.0, use_top_k: Bool = false, top_k: Int = 1):
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.use_top_k = use_top_k
        self.top_k = top_k
