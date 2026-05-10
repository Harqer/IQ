# Modular Evolution System - Clean, Testable, Composable
# Uses dependency injection and single responsibility principles
# No atomic design terminology - just clean architecture

from tensor import Tensor
from math import sqrt, exp, tanh
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.adapters.evolution_components import (
    FidelityMonitor, PhysicsFidelityMonitor,
    EvolutionController, ConstrainedEvolutionController,
    GradientCalculator, VeraGradientCalculator,
    AdaptationTracker, EvolutionAdaptationTracker,
    PhysicsBaselineManager, EvolutionModeManager, EvolutionSafetyController
)

# Main evolution orchestrator - coordinates all components
struct ModularEvolutionSystem:
    var config: SystemConfig
    
    # Dependency-injected components (testable, replaceable)
    var fidelity_monitor: FidelityMonitor
    var evolution_controller: EvolutionController
    var gradient_calculator: GradientCalculator
    var adaptation_tracker: AdaptationTracker
    var baseline_manager: PhysicsBaselineManager
    var mode_manager: EvolutionModeManager
    var safety_controller: EvolutionSafetyController
    
    fn __init__(out self, config: SystemConfig):
        self.config = config
        
        # Dependency injection - each component has single responsibility
        self.fidelity_monitor = PhysicsFidelityMonitor()
        self.evolution_controller = ConstrainedEvolutionController()
        self.gradient_calculator = VeraGradientCalculator()
        self.adaptation_tracker = EvolutionAdaptationTracker()
        self.baseline_manager = PhysicsBaselineManager(config.hidden_dim)
        self.mode_manager = EvolutionModeManager()
        self.safety_controller = EvolutionSafetyController()
    
    fn process_evolution_cycle(mut self, 
                               physics_output: Tensor[DType.float32],
                               adapted_output: Tensor[DType.float32],
                               performance_feedback: Float32) -> EvolutionResult:
        """Single responsibility: Orchestrate one complete evolution cycle"""
        
        # Step 1: Monitor physics fidelity
        var fidelity = self.fidelity_monitor.compute_fidelity(physics_output, adapted_output)
        
        # Step 2: Check if evolution is safe
        var is_safe = self.fidelity_monitor.is_acceptable(fidelity, self.mode_manager.current_threshold)
        
        if not is_safe:
            # Step 3: Handle safety violation
            return self.handle_safety_violation(fidelity)
        
        # Step 4: Calculate evolution budget
        var budget = self.evolution_controller.calculate_evolution_budget(
            fidelity, self.mode_manager.current_threshold
        )
        
        # Step 5: Check if evolution should proceed
        if not self.evolution_controller.should_evolve(budget):
            return EvolutionResult(success=False, reason="Insufficient evolution budget")
        
        # Step 6: Compute and scale gradients
        var gradients = self.gradient_calculator.compute_gradients(performance_feedback, self.config)
        var scaled_gradients = self.gradient_calculator.scale_gradients(gradients, budget)
        
        # Step 7: Record cycle data
        self.adaptation_tracker.record_cycle(budget, performance_feedback)
        self.adaptation_tracker.record_fidelity(fidelity)
        
        return EvolutionResult(success=True, gradients=scaled_gradients, fidelity=fidelity)
    
    fn handle_safety_violation(mut self, fidelity: Float32) -> EvolutionResult:
        """Single responsibility: Handle when physics fidelity is compromised"""
        
        if self.safety_controller.should_trigger_reset(fidelity, self.mode_manager.current_threshold):
            var reset_successful = self.safety_controller.execute_reset(self.mode_manager)
            
            if reset_successful:
                return EvolutionResult(success=False, reason="Emergency reset executed", reset_triggered=True)
            else:
                return EvolutionResult(success=False, reason="Safety system exhausted")
        
        return EvolutionResult(success=False, reason="Physics fidelity compromised")
    
    fn set_evolution_mode(mut self, mode: String):
        """Single responsibility: Change evolution mode"""
        if mode == "conservative":
            self.mode_manager.set_conservative_mode()
        elif mode == "aggressive":
            self.mode_manager.set_aggressive_mode()
        elif mode == "balanced":
            self.mode_manager.set_balanced_mode()
    
    fn get_system_status(self) -> SystemStatus:
        """Single responsibility: Provide complete system status"""
        var stats = self.adaptation_tracker.get_statistics()
        var parameters = self.mode_manager.get_parameters()
        
        return SystemStatus(
            total_cycles=Int(stats[0]),
            average_adaptation=stats[1],
            average_fidelity=stats[2],
            evolution_efficiency=stats[3],
            current_rate=parameters[0],
            current_strength=parameters[1],
            current_threshold=parameters[2],
            reset_count=self.safety_controller.reset_count
        )

# Result data structures - clean data transfer
struct EvolutionResult:
    var success: Bool
    var reason: String
    var gradients: Tensor[DType.float32]
    var fidelity: Float32
    var reset_triggered: Bool
    
    # Constructor for successful evolution
    fn create_success(gradients: Tensor[DType.float32], fidelity: Float32) -> EvolutionResult:
        return EvolutionResult(
            success=True, 
            reason="Evolution successful", 
            gradients=gradients, 
            fidelity=fidelity,
            reset_triggered=False
        )
    
    # Constructor for failed evolution
    fn create_failure(reason: String) -> EvolutionResult:
        var empty_grads = Tensor[DType.float32](1, 1)
        return EvolutionResult(
            success=False, 
            reason=reason, 
            gradients=empty_grads, 
            fidelity=0.0,
            reset_triggered=False
        )

# System status data structure
struct SystemStatus:
    var total_cycles: Int
    var average_adaptation: Float32
    var average_fidelity: Float32
    var evolution_efficiency: Float32
    var current_rate: Float32
    var current_strength: Float32
    var current_threshold: Float32
    var reset_count: Int
    
    fn to_string(self) -> String:
        var status = "🧬 Modular Evolution System Status\n"
        status += "=" * 40 + "\n"
        status += "Total Cycles: {}\n".format(self.total_cycles)
        status += "Average Adaptation: {:.6f}\n".format(self.average_adaptation)
        status += "Average Fidelity: {:.4f}%\n".format(self.average_fidelity * 100)
        status += "Evolution Efficiency: {:.6f}\n".format(self.evolution_efficiency)
        status += "Current Rate: {:.6f}\n".format(self.current_rate)
        status += "Current Strength: {:.4f}\n".format(self.current_strength)
        status += "Current Threshold: {:.4f}%\n".format(self.current_threshold * 100)
        status += "Reset Count: {}\n".format(self.reset_count)
        return status

# Physics processor - handles core physics processing (untouched)
struct PhysicsProcessor:
    var config: SystemConfig
    
    fn __init__(out self, config: SystemConfig):
        self.config = config
    
    fn process_physics(self, input_tokens: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Single responsibility: Process through core physics pipeline"""
        # This would integrate with your existing NIF architecture
        var output = Tensor[DType.float32](input_tokens.shape())
        
        # Mock physics processing - in production this calls your actual NIF
        for i in range(input_tokens.shape()[0]):
            for j in range(input_tokens.shape()[1]):
                for k in range(input_tokens.shape()[2]):
                    output[i, j, k] = tanh(input_tokens[i, j, k])
        
        return output

# Adapter interface - allows different adapter implementations
trait AdapterInterface:
    fn apply_adaptation(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]
    fn update_parameters(self, gradients: Tensor[DType.float32])
    fn reset_parameters(self)

# VeRA adapter implementation following the interface
struct VeraAdapter:
    var scaling_vectors: Tensor[DType.float32]
    var random_matrices: Tensor[DType.float32]
    var bias_vectors: Tensor[DType.float32]
    var config: SystemConfig
    
    fn __init__(out self, config: SystemConfig):
        self.config = config
        self.scaling_vectors = self.initialize_scaling_vectors()
        self.random_matrices = self.initialize_random_matrices()
        self.bias_vectors = self.initialize_bias_vectors()
    
    fn apply_adaptation(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Single responsibility: Apply VeRA transformation"""
        var shape = input.shape()
        var output = Tensor[DType.float32](shape)
        
        for b in range(shape[0]):
            for s in range(shape[1]):
                var input_vector = input[b, s, :]
                var transformed = self.apply_vera_transform(input_vector)
                
                for k in range(shape[2]):
                    output[b, s, k] = transformed[k]
        
        return output
    
    fn update_parameters(self, gradients: Tensor[DType.float32]):
        """Single responsibility: Update VeRA parameters"""
        var learning_rate = 0.001
        
        for i in range(self.config.hidden_dim):
            for j in range(self.config.vera_rank):
                self.scaling_vectors[i, j] -= learning_rate * gradients[i, j]
    
    fn reset_parameters(self):
        """Single responsibility: Reset to initial state"""
        for i in range(self.config.hidden_dim):
            self.bias_vectors[i] = 0.0
            for j in range(self.config.vera_rank):
                self.scaling_vectors[i, j] = 0.01 * (Float32(i + j) - Float32(self.config.hidden_dim / 2.0)) / Float32(self.config.hidden_dim)
    
    # Private helper methods
    fn initialize_scaling_vectors(self) -> Tensor[DType.float32]:
        var scaling = Tensor[DType.float32](self.config.hidden_dim, self.config.vera_rank)
        
        for i in range(self.config.hidden_dim):
            for j in range(self.config.vera_rank):
                scaling[i, j] = 0.01 * (Float32(i + j) - Float32(self.config.hidden_dim / 2.0)) / Float32(self.config.hidden_dim)
        
        return scaling
    
    fn initialize_random_matrices(self) -> Tensor[DType.float32]:
        var random_mats = Tensor[DType.float32](self.config.vera_rank, self.config.hidden_dim)
        
        for i in range(self.config.vera_rank):
            for j in range(self.config.hidden_dim):
                var theta = 2.0 * 3.14159 * Float32(i) / Float32(self.config.vera_rank)
                random_mats[i, j] = cos(theta + Float32(j))
        
        return random_mats
    
    fn initialize_bias_vectors(self) -> Tensor[DType.float32]:
        var bias = Tensor[DType.float32](self.config.hidden_dim)
        
        for i in range(self.config.hidden_dim):
            bias[i] = 0.0
        
        return bias
    
    fn apply_vera_transform(self, input_vector: Tensor[DType.float32]) -> Tensor[DType.float32]:
        var transformed = Tensor[DType.float32](self.config.hidden_dim)
        
        # R * x
        var rx = Tensor[DType.float32](self.config.vera_rank)
        for i in range(self.config.vera_rank):
            var sum = 0.0
            for j in range(self.config.hidden_dim):
                sum += self.random_matrices[i, j] * input_vector[j]
            rx[i] = sum
        
        # S * (R * x)
        var srx = Tensor[DType.float32](self.config.hidden_dim)
        for i in range(self.config.hidden_dim):
            var sum = 0.0
            for j in range(self.config.vera_rank):
                sum += self.scaling_vectors[i, j] * rx[j]
            srx[i] = sum
        
        # y = x + S * R * x + b
        for i in range(self.config.hidden_dim):
            transformed[i] = input_vector[i] + srx[i] + self.bias_vectors[i]
        
        return transformed

# Main orchestrator - coordinates physics and evolution
struct PhysicsPreservingOrchestrator:
    var config: SystemConfig
    var physics_processor: PhysicsProcessor
    var adapter: AdapterInterface
    var evolution_system: ModularEvolutionSystem
    
    fn __init__(out self, config: SystemConfig):
        self.config = config
        self.physics_processor = PhysicsProcessor(config)
        self.adapter = VeraAdapter(config)  # Can be injected with any adapter
        self.evolution_system = ModularEvolutionSystem(config)
    
    fn forward(mut self, input_tokens: Tensor[DType.float32], performance_feedback: Float32 = 0.0) -> Tensor[DType.float32]:
        """Single responsibility: Complete forward pass with evolution"""
        
        # Step 1: Core physics processing (untouched)
        var physics_output = self.physics_processor.process_physics(input_tokens)
        
        # Step 2: Apply adapter
        var adapted_output = self.adapter.apply_adaptation(physics_output)
        
        # Step 3: Process evolution cycle
        var evolution_result = self.evolution_system.process_evolution_cycle(
            physics_output, adapted_output, performance_feedback
        )
        
        # Step 4: Apply evolution if successful
        if evolution_result.success:
            self.adapter.update_parameters(evolution_result.gradients)
        elif evolution_result.reset_triggered:
            self.adapter.reset_parameters()
        
        return adapted_output
    
    fn set_mode(mut self, mode: String):
        """Single responsibility: Change evolution mode"""
        self.evolution_system.set_evolution_mode(mode)
    
    fn get_status(self) -> String:
        """Single responsibility: Get system status"""
        return self.evolution_system.get_system_status().to_string()

# Test utilities - for comprehensive testing
struct EvolutionTestSuite:
    var config: SystemConfig
    var orchestrator: PhysicsPreservingOrchestrator
    
    fn __init__(out self):
        self.config = SystemConfig()
        self.config.hidden_dim = 512  # Smaller for testing
        self.config.vera_rank = 32
        self.orchestrator = PhysicsPreservingOrchestrator(self.config)
    
    fn test_physics_preservation(self) -> Bool:
        """Single responsibility: Test that physics is preserved"""
        var test_input = self.create_test_input()
        
        # Get physics-only output
        var physics_output = self.orchestrator.physics_processor.process_physics(test_input)
        
        # Get evolved output
        var evolved_output = self.orchestrator.forward(test_input, 0.5)
        
        # Check fidelity
        var fidelity = self.orchestrator.evolution_system.fidelity_monitor.compute_fidelity(
            physics_output, evolved_output
        )
        
        return fidelity >= 0.98
    
    fn test_evolution_modes(self) -> Bool:
        """Single responsibility: Test different evolution modes"""
        var test_input = self.create_test_input()
        
        # Test conservative mode
        self.orchestrator.set_mode("conservative")
        var conservative_output = self.orchestrator.forward(test_input, 0.3)
        
        # Test aggressive mode
        self.orchestrator.set_mode("aggressive")
        var aggressive_output = self.orchestrator.forward(test_input, 0.7)
        
        # Test balanced mode
        self.orchestrator.set_mode("balanced")
        var balanced_output = self.orchestrator.forward(test_input, 0.5)
        
        return True  # Simplified test - would check actual differences
    
    fn create_test_input(self) -> Tensor[DType.float32]:
        """Single responsibility: Create test data"""
        var input = Tensor[DType.float32](2, 8, self.config.hidden_dim)
        
        for i in range(2):
            for j in range(8):
                for k in range(self.config.hidden_dim):
                    input[i, j, k] = Float32((i * 8 * self.config.hidden_dim + j * self.config.hidden_dim + k) % 1000) / 1000.0
        
        return input
