# Load Balancer
# Atomic component for expert load balancing in MoE systems
# Single responsibility: load balancing and capacity management

from tensor import Tensor
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.interfaces.routing_interface import RoutingInterface, RoutingConfig

# Load balancing strategy
struct LoadBalancingStrategy:
    var strategy_type: String
    var capacity_factor: Float32
    var load_decay_rate: Float32
    var min_load_threshold: Float32
    
    fn __init__(strategy_type: String = "adaptive", capacity_factor: Float32 = 1.0, 
                load_decay_rate: Float32 = 0.99, min_load_threshold: Float32 = 0.1):
        self.strategy_type = strategy_type
        self.capacity_factor = capacity_factor
        self.load_decay_rate = load_decay_rate
        self.min_load_threshold = min_load_threshold

# Expert load tracker
struct ExpertLoadTracker:
    var expert_loads: Tensor[Float32]
    var expert_capacities: Tensor[Float32]
    var num_experts: Int
    var update_counter: Int
    
    fn __init__(num_experts: Int, capacity_factor: Float32 = 1.0):
        self.num_experts = num_experts
        self.expert_loads = Tensor[Float32](num_experts)
        self.expert_capacities = Tensor[Float32](num_experts)
        self.update_counter = 0
        
        # Initialize loads and capacities
        for i in range(num_experts):
            self.expert_loads[i] = 0.0
            self.expert_capacities[i] = capacity_factor
    
    fn update_load(mut self, expert_id: Int, additional_load: Float32):
        """Update load for specific expert"""
        if expert_id >= 0 and expert_id < self.num_experts:
            self.expert_loads[expert_id] += additional_load
            self.update_counter += 1
    
    fn decay_loads(mut self, decay_rate: Float32):
        """Apply decay to all loads"""
        for i in range(self.num_experts):
            self.expert_loads[i] *= decay_rate
    
    def get_load(self, expert_id: Int) -> Float32:
        """Get current load for expert"""
        if expert_id >= 0 and expert_id < self.num_experts:
            return self.expert_loads[expert_id]
        return 1.0  # Full load as default
    
    def is_available(self, expert_id: Int) -> Bool:
        """Check if expert is available (not at capacity)"""
        if expert_id >= 0 and expert_id < self.num_experts:
            return self.expert_loads[expert_id] < self.expert_capacities[expert_id]
        return False
    
    def get_available_experts(self) -> Tensor[Int]:
        """Get list of available expert IDs"""
        var available = Tensor[Int](self.num_experts)
        var count = 0
        
        for i in range(self.num_experts):
            if self.is_available(i):
                available[count] = i
                count += 1
        
        return available
    
    def get_load_statistics(self) -> String:
        """Get load statistics"""
        var total_load = 0.0
        var max_load = 0.0
        var min_load = 1.0
        var available_count = 0
        
        for i in range(self.num_experts):
            total_load += self.expert_loads[i]
            if self.expert_loads[i] > max_load:
                max_load = self.expert_loads[i]
            if self.expert_loads[i] < min_load:
                min_load = self.expert_loads[i]
            if self.is_available(i):
                available_count += 1
        
        var avg_load = total_load / Float32(self.num_experts)
        
        var stats = "📊 Load Statistics:\n"
        stats += "  - Average Load: {:.3f}\n".format(avg_load)
        stats += "  - Max Load: {:.3f}\n".format(max_load)
        stats += "  - Min Load: {:.3f}\n".format(min_load)
        stats += "  - Available Experts: {}/{}\n".format(available_count, self.num_experts)
        stats += "  - Update Counter: {}\n".format(self.update_counter)
        
        return stats

# Load balancing calculator
struct LoadBalancingCalculator:
    var strategy: LoadBalancingStrategy
    
    fn __init__(strategy: LoadBalancingStrategy):
        self.strategy = strategy
    
    fn compute_load_balanced_scores(self, original_scores: Tensor[DType.float32], 
                                   expert_loads: Tensor[Float32]) -> Tensor[DType.float32]:
        """Compute load-balanced scores"""
        var shape = original_scores.shape()
        var balanced_scores = Tensor[DType.float32](shape)
        
        for b in range(shape[0]):
            for s in range(shape[1]):
                for expert_idx in range(shape[2]):
                    var original_score = original_scores[b, s, expert_idx]
                    var load_penalty = self.compute_load_penalty(expert_idx, expert_loads)
                    
                    balanced_scores[b, s, expert_idx] = original_score - load_penalty
        
        return balanced_scores
    
    fn compute_load_penalty(self, expert_idx: Int, expert_loads: Tensor[Float32]) -> Float32:
        """Compute load penalty for expert"""
        if expert_idx >= expert_loads.shape()[0]:
            return 0.0
        
        var load = expert_loads[expert_idx]
        
        if self.strategy.strategy_type == "linear":
            return load * 0.1
        elif self.strategy.strategy_type == "exponential":
            return exp(load * 2.0) - 1.0
        elif self.strategy.strategy_type == "adaptive":
            if load < 0.5:
                return load * 0.05
            elif load < 0.8:
                return load * 0.15
            else:
                return load * 0.5
        else:
            return load * 0.1
    
    fn balance_load(self, expert_loads: Tensor[Float32]) -> Tensor[Float32]:
        """Balance loads across experts"""
        var balanced = Tensor[Float32](expert_loads.shape()[0])
        
        if self.strategy.strategy_type == "equalize":
            # Equalize loads
            var total_load = 0.0
            for i in range(expert_loads.shape()[0]):
                total_load += expert_loads[i]
            
            var target_load = total_load / Float32(expert_loads.shape()[0])
            for i in range(expert_loads.shape()[0]):
                balanced[i] = target_load
        else:
            # Apply decay
            for i in range(expert_loads.shape()[0]):
                balanced[i] = expert_loads[i] * self.strategy.load_decay_rate
        
        return balanced

# Load Balancer Service
struct LoadBalancerService:
    var config: RoutingConfig
    var load_tracker: ExpertLoadTracker
    var balancing_calculator: LoadBalancingCalculator
    var strategy: LoadBalancingStrategy
    
    fn __init__(config: SystemConfig):
        self.config = RoutingConfig(config.num_experts, 1.0, "load_balanced", "softmax")
        self.strategy = LoadBalancingStrategy("adaptive", 1.0, 0.99, 0.1)
        self.load_tracker = ExpertLoadTracker(config.num_experts, self.strategy.capacity_factor)
        self.balancing_calculator = LoadBalancingCalculator(self.strategy)
        
        print("⚖️ Load Balancer Service Initialized")
        print("   - Strategy: {}".format(self.strategy.strategy_type))
        print("   - Capacity Factor: {:.2f}".format(self.strategy.capacity_factor))
        print("   - Load Decay Rate: {:.3f}".format(self.strategy.load_decay_rate))
    
    fn balance_load(self, expert_loads: Tensor[Float32]) -> Tensor[Float32]:
        """Balance loads across experts"""
        return self.balancing_calculator.balance_load(expert_loads)
    
    fn update_expert_load(mut self, expert_id: Int, additional_load: Float32):
        """Update load for specific expert"""
        self.load_tracker.update_load(expert_id, additional_load)
    
    fn apply_load_balancing(self, scores: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply load balancing to scores"""
        return self.balancing_calculator.compute_load_balanced_scores(scores, self.load_tracker.expert_loads)
    
    fn get_available_experts(self) -> Tensor[Int]:
        """Get available experts"""
        return self.load_tracker.get_available_experts()
    
    fn is_expert_available(self, expert_id: Int) -> Bool:
        """Check if expert is available"""
        return self.load_tracker.is_available(expert_id)
    
    fn get_expert_load(self, expert_id: Int) -> Float32:
        """Get expert load"""
        return self.load_tracker.get_load(expert_id)
    
    fn decay_all_loads(mut self):
        """Apply decay to all loads"""
        self.load_tracker.decay_loads(self.strategy.load_decay_rate)
    
    fn reset_loads(mut self):
        """Reset all loads to zero"""
        for i in range(self.load_tracker.num_experts):
            self.load_tracker.expert_loads[i] = 0.0
        self.load_tracker.update_counter = 0
    
    def get_balancer_info(self) -> String:
        """Get balancer information"""
        var info = "⚖️ Load Balancer Service Information\n"
        info += "=" * 35 + "\n"
        info += "Strategy: {}\n".format(self.strategy.strategy_type)
        info += "Capacity Factor: {:.2f}\n".format(self.strategy.capacity_factor)
        info += "Load Decay Rate: {:.3f}\n".format(self.strategy.load_decay_rate)
        info += "Min Load Threshold: {:.3f}\n".format(self.strategy.min_load_threshold)
        
        info += "\n"
        info += self.load_tracker.get_load_statistics()
        
        return info

# Factory function
fn create_load_balancer_service(config: SystemConfig) -> LoadBalancerService:
    """Create load balancer service"""
    return LoadBalancerService(config)
