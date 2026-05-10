# Expert Interface
# Defines contract for expert processing in MoE systems
# Enables dependency injection and testability for expert components

from tensor import Tensor

trait ExpertInterface:
    # Core expert processing
    fn process(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]
    
    # Expert identification
    fn get_expert_type(self) -> String
    fn get_expert_id(self) -> Int
    fn get_expert_capacity(self) -> Int
    
    # Expert state management
    fn is_available(self) -> Bool
    fn get_current_load(self) -> Float32
    fn update_load(self, load: Float32)

# Expert configuration
struct ExpertConfig:
    var expert_type: String
    var expert_id: Int
    var capacity: Int
    var hidden_dim: Int
    var specialization: String
    
    fn __init__(expert_type: String, expert_id: Int, capacity: Int, hidden_dim: Int, specialization: String = "general"):
        self.expert_type = expert_type
        self.expert_id = expert_id
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.specialization = specialization
