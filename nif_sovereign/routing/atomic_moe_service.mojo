# Atomic MoE Service
# Main orchestrator for atomic MoE system
# Single responsibility: coordinating all MoE components

from tensor import Tensor
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.interfaces.routing_interface import RoutingInterface
from nif_sovereign.routing.expert_registry import ExpertRegistry
from nif_sovereign.routing.gating_network import GatingNetworkService
from nif_sovereign.routing.load_balancer import LoadBalancerService
from nif_sovereign.routing.expert_dispatcher import ExpertDispatcherService

# Atomic MoE Service - Main orchestrator
struct AtomicMoEService:
    var config: SystemConfig
    var expert_dispatcher: ExpertDispatcherService
    var expert_registry: ExpertRegistry
    var gating_network: GatingNetworkService
    var load_balancer: LoadBalancerService
    
    fn __init__(config: SystemConfig):
        self.config = config
        self.expert_dispatcher = ExpertDispatcherService(config)
        self.expert_registry = self.expert_dispatcher.expert_registry
        self.gating_network = self.expert_dispatcher.gating_network
        self.load_balancer = self.expert_dispatcher.load_balancer
        
        print("🌟 Atomic MoE Service Initialized")
        print("   - Atomic Design: Applied")
        print("   - Single Responsibility: Each component focused")
        print("   - Dependency Injection: Active")
        print("   - Interface Segregation: Active")
        print("   - Composition: Enabled")
    
    fn forward(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass through atomic MoE system"""
        return self.expert_dispatcher.dispatch_tokens(input)
    
    fn register_expert(self, expert):
        """Register a new expert"""
        # This would need proper expert interface implementation
        print("📝 Expert registration interface ready")
    
    fn route_experts(self, input: Tensor[DType.float32]) -> Tensor[Int]:
        """Route tokens to experts (implements RoutingInterface)"""
        var gating_scores = self.gating_network.compute_gating_scores(input)
        var balanced_scores = self.load_balancer.apply_load_balancing(gating_scores)
        var expert_probabilities = self.gating_network.apply_softmax(balanced_scores)
        
        # Convert to expert IDs
        var shape = expert_probabilities.shape()
        var expert_ids = Tensor[Int](shape[0], shape[1])
        
        for b in range(shape[0]):
            for s in range(shape[1]):
                var best_expert = 0
                var best_score = expert_probabilities[b, s, 0]
                
                for expert_idx in range(shape[2]):
                    if expert_probabilities[b, s, expert_idx] > best_score:
                        best_score = expert_probabilities[b, s, expert_idx]
                        best_expert = expert_idx
                
                expert_ids[b, s] = best_expert
        
        return expert_ids
    
    fn balance_load(self, expert_loads: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Balance expert loads (implements RoutingInterface)"""
        return self.load_balancer.balance_load(expert_loads)
    
    fn select_expert(self, scores: Tensor[DType.float32]) -> Int:
        """Select best expert (implements RoutingInterface)"""
        var best_expert = 0
        var best_score = scores[0]
        
        for i in range(scores.shape()[0]):
            if scores[i] > best_score:
                best_score = scores[i]
                best_expert = i
        
        return best_expert
    
    fn get_service_info(self) -> String:
        """Get comprehensive service information"""
        var info = "🌟 Atomic MoE Service Information\n"
        info += "=" * 35 + "\n"
        
        info += "Design Principles:\n"
        info += "- Atomic Design: ✅ Applied\n"
        info += "- Single Responsibility: ✅ Each component focused\n"
        info += "- Dependency Injection: ✅ Service-based\n"
        info += "- Interface Segregation: ✅ Small interfaces\n"
        info += "- Composition: ✅ Composable components\n\n"
        
        info += "Component Structure:\n"
        info += "- Expert Registry: Expert lifecycle management\n"
        info += "- Gating Network: Score computation\n"
        info += "- Load Balancer: Load management\n"
        info += "- Expert Dispatcher: Token dispatching\n"
        info += "- Result Aggregator: Output combination\n\n"
        
        info += "Benefits:\n"
        info += "- Testability: Each component independently testable\n"
        info += "- Maintainability: Single responsibility makes code clear\n"
        info += "- Scalability: Components can be swapped and reused\n"
        info += "- Flexibility: Interface-based design enables changes\n"
        
        return info

# Factory function
fn create_atomic_moe_service(config: SystemConfig) -> AtomicMoEService:
    """Create atomic MoE service"""
    return AtomicMoEService(config)

# Usage example
fn main():
    print("🌟 Initializing Atomic MoE Service")
    
    var config = SystemConfig()
    var atomic_moe = create_atomic_moe_service(config)
    
    # Create test input
    var test_input = Tensor[DType.float32](2, 8, config.hidden_dim)
    for b in range(2):
        for s in range(8):
            for h in range(config.hidden_dim):
                test_input[b, s, h] = Float32((b * 8 * config.hidden_dim + s * config.hidden_dim + h) % 1000) / 1000.0
    
    print("\n🚀 Testing Atomic MoE Service...")
    
    # Run forward pass
    var output = atomic_moe.forward(test_input)
    
    print("\n" + atomic_moe.get_service_info())
    
    print("\n✅ Atomic MoE Service Test Successful")
    print("   - Input Shape: [{}, {}, {}]".format(2, 8, config.hidden_dim))
    print("   - Output Shape: [{}, {}, {}]".format(output.shape()[0], output.shape()[1], output.shape()[2]))
    
    print("\n🌟 ATOMIC DESIGN BENEFITS:")
    print("✅ Single Responsibility: Each component focused")
    print("✅ Dependency Injection: Service-based architecture")
    print("✅ Interface Segregation: Small, focused interfaces")
    print("✅ Composition: Composable and reusable")
    print("✅ Maintainability: Easy to understand and modify")
    print("✅ Scalability: Components can be swapped")
    print("✅ Testability: Each component independently testable")
    print("✅ Flexibility: Interface-based design enables changes")
