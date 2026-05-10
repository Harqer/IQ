# Evolution Components - Modular, Testable, Single Responsibility
# Each component has one clear purpose and can be tested independently

from tensor import Tensor
from math import sqrt, exp, tanh
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.core.custom_training_logic import SovereignTensor

# Interface for physics fidelity monitoring
trait FidelityMonitor:
    fn compute_fidelity(self, baseline: Tensor[DType.float32], output: Tensor[DType.float32]) -> Float32
    fn is_acceptable(self, fidelity: Float32, threshold: Float32) -> Bool

# Interface for evolution control
trait EvolutionController:
    fn calculate_evolution_budget(self, fidelity: Float32, threshold: Float32) -> Float32
    fn should_evolve(self, budget: Float32) -> Bool

# Interface for gradient computation
trait GradientCalculator:
    fn compute_gradients(self, feedback: Float32, config: SystemConfig) -> Tensor[DType.float32]
    fn scale_gradients(self, gradients: Tensor[DType.float32], budget: Float32) -> Tensor[DType.float32]

# Interface for adaptation tracking
trait AdaptationTracker:
    fn record_cycle(self, budget: Float32, feedback: Float32)
    fn get_statistics(self) -> Tensor[Float32]
    fn compute_efficiency(self) -> Float32

# Concrete implementation: Physics fidelity monitoring
struct PhysicsFidelityMonitor:
    var tolerance_threshold: Float32
    
    fn __init__(out self, tolerance: Float32 = 1e-6):
        self.tolerance_threshold = tolerance
    
    fn compute_fidelity(self, baseline: Tensor[DType.float32], output: Tensor[DType.float32]) -> Float32:
        """Single responsibility: Calculate physics fidelity between baseline and output"""
        var shape = baseline.shape()
        var total_diff = 0.0
        var total_elements = 0
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    var diff = abs(baseline[i, j, k] - output[i, j, k])
                    total_diff += diff
                    total_elements += 1
        
        var avg_diff = total_diff / Float32(total_elements)
        var fidelity = 1.0 - avg_diff
        
        # Clamp to valid range
        return max(0.0, min(1.0, fidelity))
    
    fn is_acceptable(self, fidelity: Float32, threshold: Float32) -> Bool:
        """Single responsibility: Determine if fidelity meets threshold"""
        return fidelity >= threshold - self.tolerance_threshold

# Concrete implementation: Evolution budget calculation
struct ConstrainedEvolutionController:
    var adaptation_strength: Float32
    var minimum_budget: Float32
    
    fn __init__(out self, strength: Float32 = 0.1, minimum: Float32 = 1e-8):
        self.adaptation_strength = strength
        self.minimum_budget = minimum
    
    fn calculate_evolution_budget(self, fidelity: Float32, threshold: Float32) -> Float32:
        """Single responsibility: Calculate safe evolution budget"""
        var fidelity_margin = fidelity - threshold
        var budget = fidelity_margin * self.adaptation_strength
        return max(self.minimum_budget, budget)
    
    fn should_evolve(self, budget: Float32) -> Bool:
        """Single responsibility: Determine if evolution should proceed"""
        return budget > self.minimum_budget

# Concrete implementation: Gradient computation for VeRA
struct VeraGradientCalculator:
    var evolution_rate: Float32
    var noise_factor: Float32
    
    fn __init__(out self, rate: Float32 = 0.001, noise: Float32 = 0.0001):
        self.evolution_rate = rate
        self.noise_factor = noise
    
    fn compute_gradients(self, feedback: Float32, config: SystemConfig) -> Tensor[DType.float32]:
        """Single responsibility: Compute VeRA evolution gradients"""
        var gradients = Tensor[DType.float32](config.hidden_dim, config.vera_rank)
        
        for i in range(config.hidden_dim):
            for j in range(config.vera_rank):
                # Performance-driven gradient
                var base_gradient = feedback * self.evolution_rate
                
                # Small exploration noise
                var noise = self.noise_factor * (Float32(i + j) - Float32(config.hidden_dim / 2))
                
                gradients[i, j] = base_gradient + noise
        
        return gradients
    
    fn scale_gradients(self, gradients: Tensor[DType.float32], budget: Float32) -> Tensor[DType.float32]:
        """Single responsibility: Scale gradients by available budget"""
        var scaled = Tensor[DType.float32](gradients.shape())
        
        for i in range(gradients.shape()[0]):
            for j in range(gradients.shape()[1]):
                scaled[i, j] = gradients[i, j] * budget
        
        return scaled

# Concrete implementation: Adaptation tracking
struct EvolutionAdaptationTracker:
    var max_cycles: Int
    var cycle_count: Int
    var adaptation_history: Tensor[Float32]
    var fidelity_history: Tensor[Float32]
    
    fn __init__(out self, max_history: Int = 10000):
        self.max_cycles = max_history
        self.cycle_count = 0
        self.adaptation_history = Tensor[Float32](max_history)
        self.fidelity_history = Tensor[Float32](1000)  # Shorter fidelity history
    
    fn record_cycle(self, budget: Float32, feedback: Float32):
        """Single responsibility: Record evolution cycle data"""
        var history_index = self.cycle_count % self.max_cycles
        self.adaptation_history[history_index] = budget * feedback
        self.cycle_count += 1
    
    fn record_fidelity(self, fidelity: Float32):
        """Single responsibility: Record fidelity measurement"""
        var fidelity_index = self.cycle_count % 1000
        self.fidelity_history[fidelity_index] = fidelity
    
    fn get_statistics(self) -> Tensor[Float32]:
        """Single responsibility: Return adaptation statistics"""
        var stats = Tensor[Float32](4)
        stats[0] = Float32(self.cycle_count)  # Total cycles
        stats[1] = self.compute_average_adaptation()  # Average adaptation
        stats[2] = self.compute_average_fidelity()  # Average fidelity
        stats[3] = self.compute_efficiency()  # Evolution efficiency
        return stats
    
    fn compute_average_adaptation(self) -> Float32:
        """Single responsibility: Calculate average adaptation"""
        if self.cycle_count == 0:
            return 0.0
        
        var sum = 0.0
        var count = min(100, self.cycle_count)
        
        for i in range(count):
            sum += self.adaptation_history[i]
        
        return sum / Float32(count)
    
    fn compute_average_fidelity(self) -> Float32:
        """Single responsibility: Calculate average fidelity"""
        if self.cycle_count == 0:
            return 1.0
        
        var sum = 0.0
        var count = min(100, self.cycle_count)
        
        for i in range(count):
            sum += self.fidelity_history[i]
        
        return sum / Float32(count)
    
    fn compute_efficiency(self) -> Float32:
        """Single responsibility: Calculate evolution efficiency"""
        if self.cycle_count < 10:
            return 0.0
        
        var recent_sum = 0.0
        var older_sum = 0.0
        
        # Compare recent vs older adaptations
        for i in range(min(10, self.cycle_count)):
            recent_sum += self.adaptation_history[i]
        
        for i in range(min(10, self.cycle_count - 10), max(0, self.cycle_count - 10)):
            older_sum += self.adaptation_history[i]
        
        var recent_avg = recent_sum / 10.0
        var older_avg = older_sum / 10.0
        
        return recent_avg - older_avg

# Physics baseline manager - handles physics reference
struct PhysicsBaselineManager:
    var baseline: SovereignTensor[DType.float32]
    var is_initialized: Bool
    
    fn __init__(out self, size: Int):
        self.baseline = SovereignTensor[DType.float32](size)
        self.is_initialized = False
    
    fn update_baseline(mut self, physics_output: Tensor[DType.float32]):
        """Single responsibility: Update physics baseline"""
        var shape = physics_output.shape()
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    self.baseline.buffer.ptr.value()[k] = physics_output[i, j, k]
        
        self.is_initialized = True
    
    fn get_baseline(self) -> SovereignTensor[DType.float32]:
        """Single responsibility: Get current baseline"""
        return self.baseline
    
    fn is_ready(self) -> Bool:
        """Single responsibility: Check if baseline is ready"""
        return self.is_initialized

# Evolution mode manager - handles different evolution strategies
struct EvolutionModeManager:
    var current_rate: Float32
    var current_strength: Float32
    var current_threshold: Float32
    
    fn __init__(out self):
        self.current_rate = 0.001
        self.current_strength = 0.1
        self.current_threshold = 0.98
    
    fn set_conservative_mode(mut self):
        """Single responsibility: Set conservative evolution parameters"""
        self.current_rate *= 0.1
        self.current_strength *= 0.5
        self.current_threshold = 0.995
    
    fn set_aggressive_mode(mut self):
        """Single responsibility: Set aggressive evolution parameters"""
        self.current_rate *= 2.0
        self.current_strength *= 1.5
        # Keep threshold at 0.98 for aggressive mode
    
    fn set_balanced_mode(mut self):
        """Single responsibility: Reset to balanced parameters"""
        self.current_rate = 0.001
        self.current_strength = 0.1
        self.current_threshold = 0.98
    
    fn get_parameters(self) -> (Float32, Float32, Float32):
        """Single responsibility: Return current evolution parameters"""
        return (self.current_rate, self.current_strength, self.current_threshold)

# Safety controller - handles emergency resets and safety checks
struct EvolutionSafetyController:
    var reset_count: Int
    var max_resets: Int
    var stability_window: Int
    
    fn __init__(out self, max_resets_allowed: Int = 5):
        self.reset_count = 0
        self.max_resets = max_resets_allowed
        self.stability_window = 100
    
    fn should_trigger_reset(self, fidelity: Float32, threshold: Float32) -> Bool:
        """Single responsibility: Determine if emergency reset is needed"""
        return fidelity < threshold * 0.95  # Reset if 5% below threshold
    
    fn execute_reset(mut self, mode_manager: EvolutionModeManager):
        """Single responsibility: Execute emergency reset"""
        if self.reset_count < self.max_resets:
            mode_manager.set_conservative_mode()
            self.reset_count += 1
            return True
        return False
    
    fn is_stable(self, recent_fidelities: Tensor[Float32]) -> Bool:
        """Single responsibility: Check if system is stable"""
        if recent_fidelities.size() < self.stability_window:
            return False
        
        var min_fidelity = recent_fidelities[0]
        for i in range(recent_fidelities.size()):
            if recent_fidelities[i] < min_fidelity:
                min_fidelity = recent_fidelities[i]
        
        return min_fidelity > 0.99  # Stable if all fidelities > 99%
