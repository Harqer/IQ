# Optimization Interface
# Defines contract for optimization mechanisms
# Enables dependency injection and testability

from tensor import Tensor

trait OptimizationInterface:
    # Core optimization step
    fn optimize_step(self, parameters: Tensor[DType.float32], gradients: Tensor[DType.float32]) -> Tensor[DType.float32]
    
    # Learning rate adjustment
    fn adjust_learning_rate(self, loss: Float32) -> Float32
    
    # Parameter regularization
    fn regularize_parameters(self, parameters: Tensor[DType.float32]) -> Tensor[DType.float32]
    
    # Get optimization configuration
    fn get_config(self) -> OptimizationConfig

# Optimization configuration
struct OptimizationConfig:
    var learning_rate: Float32
    var momentum: Float32
    var weight_decay: Float32
    var regularization_type: String
    
    fn __init__(learning_rate: Float32 = 0.001, momentum: Float32 = 0.9, weight_decay: Float32 = 0.01, regularization_type: String = "l2"):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.regularization_type = regularization_type
