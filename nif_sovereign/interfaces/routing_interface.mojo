# Routing Interface
# Defines contract for expert routing mechanisms
# Enables dependency injection and testability

from tensor import Tensor

trait RoutingInterface:
    # Core routing computation
    fn route_experts(self, input: Tensor[DType.float32]) -> Tensor[Int]
    
    # Load balancing
    fn balance_load(self, expert_loads: Tensor[DType.float32]) -> Tensor[DType.float32]
    
    # Expert selection
    fn select_expert(self, scores: Tensor[DType.float32]) -> Int
    
    # Get routing configuration
    fn get_config(self) -> RoutingConfig

# Routing configuration
struct RoutingConfig:
    var num_experts: Int
    var capacity_factor: Float32
    var load_balance_strategy: String
    var routing_method: String
    
    fn __init__(num_experts: Int = 3, capacity_factor: Float32 = 1.0, load_balance_strategy: String = "softmax", routing_method: String = "geodesic"):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.load_balance_strategy = load_balance_strategy
        self.routing_method = routing_method
