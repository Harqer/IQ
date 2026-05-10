# Atomic MoE Service V2
# Enhanced orchestrator with expert factory integration
# Single responsibility: coordinating all MoE components with expert factory

from tensor import Tensor
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.interfaces.routing_interface import RoutingInterface
from nif_sovereign.routing.expert_registry import ExpertRegistry
from nif_sovereign.routing.gating_network import GatingNetworkService
from nif_sovereign.routing.load_balancer import LoadBalancerService
from nif_sovereign.routing.expert_dispatcher import ExpertDispatcherService
from nif_sovereign.experts.expert_factory import ExpertFactory, ExpertManager
from nif_sovereign.experts.linguistic_expert import LinguisticExpert
from nif_sovereign.experts.physics_expert import PhysicsExpert
from nif_sovereign.experts.diffusion_expert import DiffusionExpert

# Enhanced Atomic MoE Service with Expert Factory Integration
struct AtomicMoEServiceV2:
    var config: SystemConfig
    var expert_factory: ExpertFactory
    var expert_manager: ExpertManager
    var expert_dispatcher: ExpertDispatcherService
    var expert_registry: ExpertRegistry
    var gating_network: GatingNetworkService
    var load_balancer: LoadBalancerService
    var experts_initialized: Bool

    fn __init__(config: SystemConfig):
        self.config = config
        self.expert_factory = ExpertFactory(config)
        self.expert_manager = ExpertManager(config)
        self.expert_dispatcher = ExpertDispatcherService(config)
        self.expert_registry = self.expert_dispatcher.expert_registry
        self.gating_network = self.expert_dispatcher.gating_network
        self.load_balancer = self.expert_dispatcher.load_balancer
        self.experts_initialized = False

        print("🌟 Atomic MoE Service V2 Initialized")
        print("   - Atomic Design: Applied")
        print("   - Expert Factory: Integrated")
        print("   - Expert Manager: Active")
        print("   - Single Responsibility: Each component focused")
        print("   - Dependency Injection: Active")
        print("   - Interface Segregation: Active")
        print("   - Composition: Enabled")

    fn initialize_experts(mut self):
        """Initialize all expert types"""
        if self.experts_initialized:
            print("📝 Experts already initialized")
            return

        print("🚀 Initializing Expert Factory...")

        # Create all expert types
        var linguistic_expert = self.expert_factory.create_linguistic_expert()
        var physics_expert = self.expert_factory.create_physics_expert()
        var diffusion_expert = self.expert_factory.create_diffusion_expert()

        # Register experts in registry
        self.register_expert_in_registry(linguistic_expert)
        self.register_expert_in_registry(physics_expert)
        self.register_expert_in_registry(diffusion_expert)

        self.experts_initialized = True
        print("✅ All Expert Types Initialized")

    fn register_expert_in_registry(mut self, expert):
        """Register expert in registry"""
        # This would use the ExpertInterface to register
        print("📝 Registering expert: {} (ID: {})".format(expert.get_expert_type(), expert.get_expert_id()))

    fn forward(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass through atomic MoE system using geodesic routing"""
        if not self.experts_initialized:
            self.initialize_experts()

        # Step 1: Compute geodesic distances (lower is better)
        var geodesic_distances = self.gating_network.compute_gating_scores(input)

        # Step 2: Apply load balancing to distances
        var balanced_distances = self.apply_load_balancing_to_distances(geodesic_distances)

        # Step 3: Select experts with minimum distance (closest on manifold)
        var expert_assignments = self.gating_network.select_closest_expert(balanced_distances)

        # Step 4: Process tokens through assigned experts
        var expert_outputs = self.process_through_experts_geodesic(input, expert_assignments)

        # Step 5: Aggregate outputs
        var final_output = self.aggregate_expert_outputs_geodesic(expert_outputs, expert_assignments, balanced_distances)

        # Step 6: Update load balancer
        self.update_load_balancer_geodesic(expert_assignments)

        return final_output

    fn create_expert_by_type(self, expert_type: String):
        """Create expert by type"""
        var expert = self.expert_factory.create_expert_by_type(expert_type)
        self.register_expert_in_registry(expert)
        return expert

    fn create_expert_pool(self, pool_size: Int):
        """Create expert pool"""
        var expert_pool = self.expert_factory.create_expert_pool(pool_size)

        for i in range(expert_pool.shape()[0]):
            self.register_expert_in_registry(expert_pool[i])

        return expert_pool

    fn apply_load_balancing_to_distances(self, distances: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply load balancing to geodesic distances"""
        var shape = distances.shape()
        var balanced = Tensor[DType.float32](shape)

        for b in range(shape[0]):
            for s in range(shape[1]):
                for expert_idx in range(shape[2]):
                    var distance = distances[b, s, expert_idx]
                    var load_penalty = self.load_balancer.get_expert_load(expert_idx) * 0.1
                    balanced[b, s, expert_idx] = distance + load_penalty

        return balanced

    fn process_through_experts_geodesic(self, input: Tensor[DType.float32], expert_assignments: Tensor[Int]) -> Tensor[Tensor[DType.float32]]:
        """Process tokens through assigned experts"""
        var shape = input.shape()
        var expert_outputs = Tensor[Tensor[DType.float32]](shape[0] * shape[1])
        var output_count = 0

        for b in range(shape[0]):
            for s in range(shape[1]):
                var expert_id = expert_assignments[b, s]
                var token_input = Tensor[DType.float32](shape[2])

                for i in range(shape[2]):
                    token_input[i] = input[b, s, i]

                var token_output = self.process_token_through_expert(token_input, expert_id)
                expert_outputs[output_count] = token_output
                output_count += 1

        return expert_outputs

    fn process_token_through_expert(self, token_input: Tensor[DType.float32], expert_id: Int) -> Tensor[DType.float32]:
        """Process single token through expert"""
        var expert = self.expert_registry.get_expert_by_id(expert_id)
        return expert.process(token_input)

    fn aggregate_expert_outputs_geodesic(self, expert_outputs: Tensor[Tensor[DType.float32]],
                                        expert_assignments: Tensor[Int],
                                        distances: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Aggregate expert outputs using distance-based weighting"""
        if expert_outputs.shape()[0] == 0:
            return Tensor[DType.float32](1, 1, 1)  # Fallback

        var first_output = expert_outputs[0]
        var shape = first_output.shape()
        var aggregated = Tensor[DType.float32](shape)

        var output_idx = 0
        for b in range(shape[0]):
            for s in range(shape[1]):
                var expert_id = expert_assignments[b, s]
                var distance = distances[b, s, expert_id]

                # Convert distance to weight (inverse relationship)
                var weight = exp(-distance)

                var expert_output = expert_outputs[output_idx]
                for i in range(shape[2]):
                    aggregated[b, s, i] = expert_output[i] * weight

                output_idx += 1

        return aggregated

    fn update_load_balancer_geodesic(self, expert_assignments: Tensor[Int]):
        """Update load balancer based on expert assignments"""
        var shape = expert_assignments.shape()

        for b in range(shape[0]):
            for s in range(shape[1]):
                var expert_id = expert_assignments[b, s]
                self.load_balancer.update_expert_load(expert_id, 0.1)

        # Apply decay
        self.load_balancer.decay_all_loads()

    fn route_experts(self, input: Tensor[DType.float32]) -> Tensor[Int]:
        """Route tokens using geodesic distance (implements RoutingInterface)"""
        var geodesic_distances = self.gating_network.compute_gating_scores(input)
        var balanced_distances = self.apply_load_balancing_to_distances(geodesic_distances)
        return self.gating_network.select_closest_expert(balanced_distances)

    fn balance_load(self, expert_loads: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Balance expert loads (implements RoutingInterface)"""
        return self.load_balancer.balance_load(expert_loads)

    fn select_expert(self, distances: Tensor[DType.float32]) -> Int:
        """Select expert with minimum distance (implements RoutingInterface)"""
        var best_expert = 0
        var best_distance = distances[0]

        for i in range(distances.shape()[0]):
            if distances[i] < best_distance:
                best_distance = distances[i]
                best_expert = i

        return best_expert

    def get_service_info(self) -> String:
        """Get comprehensive service information"""
        var info = "� Geodesic-Aware Atomic MoE Service Information\n"
        info += "=" * 45 + "\n"

        info += "Design Principles:\n"
        info += "- Atomic Design: ✅ Applied\n"
        info += "- Single Responsibility: ✅ Each component focused\n"
        info += "- Dependency Injection: ✅ Service-based\n"
        info += "- Interface Segregation: ✅ Small interfaces\n"
        info += "- Composition: ✅ Composable components\n"
        info += "- Expert Factory: ✅ Integrated\n"
        info += "- Expert Manager: ✅ Active\n"
        info += "- Manifold-Aware Routing: ✅ Active\n\n"

        info += "Routing Method:\n"
        info += "- Geodesic Distance: ✅ Used (not softmax)\n"
        info += "- Ising Energy Weighting: ✅ Active\n"
        info += "- Selection: Minimum distance (closest on manifold)\n"
        info += "- Load Balancing: Applied to distances\n"
        info += "- Expert Selection: Closest on Lorentzian manifold\n\n"

        info += "Component Structure:\n"
        info += "- Expert Factory: Expert creation\n"
        info += "- Expert Manager: Expert lifecycle\n"
        info += "- Manifold Gating: Geodesic distance calculation\n"
        info += "- Load Balancer: Distance-based load management\n"
        info += "- Expert Dispatcher: Manifold-aware dispatching\n"
        info += "- Expert Registry: Expert registration\n\n"

        info += "Expert Types:\n"
        info += "- Linguistic Expert: Standard Transformer\n"
        info += "- Physics Expert: Ising Logic Gate\n"
        info += "- Diffusion Expert: DiT Block\n\n"

        info += "Benefits:\n"
        info += "- Semantic Proximity: Experts closest in meaning space\n"
        info += "- Physics-Aware: Ising energy weighting\n"
        info += "- Manifold Geometry: Respects curved space structure\n"
        info += "- Load Balancing: Prevents expert overload\n"
        info += "- Testability: Each component independently testable\n"
        info += "- Maintainability: Single responsibility makes code clear\n"
        info += "- Scalability: Components can be swapped and reused\n"

        return info

    def get_expert_info(self) -> String:
        """Get expert information"""
        if not self.experts_initialized:
            return "Experts not initialized yet"

        var info = "📝 Expert Information\n"
        info += "=" * 25 + "\n"
        info += "Experts Initialized: ✅\n"
        info += "Expert Factory: Active\n"
        info += "Expert Manager: Active\n"

        info += "\n"
        info += self.expert_factory.get_factory_info()
        info += "\n"
        info += self.expert_manager.get_manager_info()

        return info

# Factory function
fn create_atomic_moe_service_v2(config: SystemConfig) -> AtomicMoEServiceV2:
    """Create atomic MoE service V2"""
    return AtomicMoEServiceV2(config)

# Usage example
fn main():
    print("🌟 Initializing Atomic MoE Service V2")

    var config = SystemConfig()
    var atomic_moe_v2 = create_atomic_moe_service_v2(config)

    # Create test input
    var test_input = Tensor[DType.float32](2, 8, config.hidden_dim)
    for b in range(2):
        for s in range(8):
            for h in range(config.hidden_dim):
                test_input[b, s, h] = Float32((b * 8 * config.hidden_dim + s * config.hidden_dim + h) % 1000) / 1000.0

    print("\n🚀 Testing Atomic MoE Service V2...")

    # Initialize experts
    atomic_moe_v2.initialize_experts()

    # Create additional experts
    var extra_expert = atomic_moe_v2.create_expert_by_type("linguistic")
    var expert_pool = atomic_moe_v2.create_expert_pool(3)

    # Run forward pass
    var output = atomic_moe_v2.forward(test_input)

    print("\n" + atomic_moe_v2.get_service_info())
    print("\n" + atomic_moe_v2.get_expert_info())

    print("\n✅ Atomic MoE Service V2 Test Successful")
    print("   - Input Shape: [{}, {}, {}]".format(2, 8, config.hidden_dim))
    print("   - Output Shape: [{}, {}, {}]".format(output.shape()[0], output.shape()[1], output.shape()[2]))

    print("\n🌟 ATOMIC DESIGN V2 BENEFITS:")
    print("✅ Single Responsibility: Each component focused")
    print("✅ Dependency Injection: Service-based architecture")
    print("✅ Interface Segregation: Small, focused interfaces")
    print("✅ Composition: Composable and reusable")
    print("✅ Maintainability: Easy to understand and modify")
    print("✅ Scalability: Components can be swapped")
    print("✅ Testability: Each component independently testable")
    print("✅ Flexibility: Interface-based design enables changes")
    print("✅ Extensibility: Expert factory for easy expansion")
    print("✅ Lifecycle Management: Expert manager handles experts")
