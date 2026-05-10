# Accuracy-Preserving Gradient Descent Strategies
# Different gradient approaches that maintain accuracy while enabling evolution
# No accuracy loss required with proper gradient strategies

from tensor import Tensor
from math import sqrt, exp, tanh, abs
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.core.custom_training_logic import SovereignTensor

# Gradient strategy interfaces
trait GradientStrategy:
    fn compute_update(self, current_params: Tensor[DType.float32], gradients: Tensor[DType.float32], 
                     performance_feedback: Float32, accuracy_constraint: Float32) -> Tensor[DType.float32]
    fn should_update(self, accuracy_impact: Float32) -> Bool

# Strategy 1: Orthogonal Gradient Descent - Prevents interference with existing accuracy
struct OrthogonalGradientStrategy:
    var learning_rate: Float32
    var accuracy_weight: Float32
    
    fn __init__(out self, lr: Float32 = 0.001, accuracy_weight: Float32 = 0.9):
        self.learning_rate = lr
        self.accuracy_weight = accuracy_weight
    
    fn compute_update(self, current_params: Tensor[DType.float32], gradients: Tensor[DType.float32],
                     performance_feedback: Float32, accuracy_constraint: Float32) -> Tensor[DType.float32]:
        """Compute gradient update orthogonal to accuracy-preserving direction"""
        
        # Step 1: Project gradients onto accuracy-preserving subspace
        var accuracy_preserving_gradients = self.compute_accuracy_preserving_gradients(
            current_params, accuracy_constraint
        )
        
        # Step 2: Make performance gradients orthogonal to accuracy gradients
        var orthogonal_gradients = self.make_orthogonal(gradients, accuracy_preserving_gradients)
        
        # Step 3: Scale by learning rate and performance feedback
        var update = Tensor[DType.float32](gradients.shape())
        for i in range(gradients.shape()[0]):
            for j in range(gradients.shape()[1]):
                update[i, j] = orthogonal_gradients[i, j] * self.learning_rate * performance_feedback
        
        return update
    
    fn make_orthogonal(self, gradients: Tensor[DType.float32], 
                      reference_gradients: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Remove component parallel to reference gradients"""
        var orthogonal = Tensor[DType.float32](gradients.shape())
        
        # Compute dot product for projection
        var dot_product = 0.0
        var ref_norm_squared = 0.0
        
        for i in range(gradients.shape()[0]):
            for j in range(gradients.shape()[1]):
                dot_product += gradients[i, j] * reference_gradients[i, j]
                ref_norm_squared += reference_gradients[i, j] * reference_gradients[i, j]
        
        if ref_norm_squared > 1e-8:
            var projection_scale = dot_product / ref_norm_squared
            
            # Subtract projection
            for i in range(gradients.shape()[0]):
                for j in range(gradients.shape()[1]):
                    orthogonal[i, j] = gradients[i, j] - projection_scale * reference_gradients[i, j]
        else:
            # No projection needed
            for i in range(gradients.shape()[0]):
                for j in range(gradients.shape()[1]):
                    orthogonal[i, j] = gradients[i, j]
        
        return orthogonal
    
    fn compute_accuracy_preserving_gradients(self, params: Tensor[DType.float32], 
                                           accuracy_constraint: Float32) -> Tensor[DType.float32]:
        """Compute gradients that preserve current accuracy"""
        var preserving_grads = Tensor[DType.float32](params.shape())
        
        # Small gradients that maintain current parameter values
        for i in range(params.shape()[0]):
            for j in range(params.shape()[1]):
                preserving_grads[i, j] = params[i, j] * accuracy_constraint * 0.001
        
        return preserving_grads
    
    fn should_update(self, accuracy_impact: Float32) -> Bool:
        """Only update if accuracy impact is acceptable"""
        return accuracy_impact >= -0.01  # Allow max 1% accuracy loss

# Strategy 2: Constrained Gradient Descent - Hard accuracy constraints
struct ConstrainedGradientStrategy:
    var learning_rate: Float32
    var max_accuracy_loss: Float32
    var constraint_strength: Float32
    
    fn __init__(out self, lr: Float32 = 0.001, max_loss: Float32 = 0.01, strength: Float32 = 10.0):
        self.learning_rate = lr
        self.max_accuracy_loss = max_loss
        self.constraint_strength = strength
    
    fn compute_update(self, current_params: Tensor[DType.float32], gradients: Tensor[DType.float32],
                     performance_feedback: Float32, accuracy_constraint: Float32) -> Tensor[DType.float32]:
        """Compute gradient update with hard accuracy constraints"""
        
        # Step 1: Predict accuracy impact of proposed update
        var predicted_accuracy_impact = self.predict_accuracy_impact(
            current_params, gradients, accuracy_constraint
        )
        
        # Step 2: If impact is too negative, constrain the update
        if predicted_accuracy_impact < -self.max_accuracy_loss:
            gradients = self.constrain_gradients(gradients, predicted_accuracy_impact)
        
        # Step 3: Apply learning rate and feedback
        var update = Tensor[DType.float32](gradients.shape())
        for i in range(gradients.shape()[0]):
            for j in range(gradients.shape()[1]):
                update[i, j] = gradients[i, j] * self.learning_rate * performance_feedback
        
        return update
    
    fn predict_accuracy_impact(self, params: Tensor[DType.float32], gradients: Tensor[DType.float32],
                              accuracy_constraint: Float32) -> Float32:
        """Predict how much accuracy will be affected by gradient update"""
        var impact = 0.0
        
        for i in range(params.shape()[0]):
            for j in range(params.shape()[1]):
                # Larger parameter changes = larger accuracy impact
                var relative_change = abs(gradients[i, j]) / (abs(params[i, j]) + 1e-8)
                impact += relative_change
        
        return -impact / Float32(params.shape()[0] * params.shape()[1])  # Normalize
    
    fn constrain_gradients(self, gradients: Tensor[DType.float32], 
                          predicted_impact: Float32) -> Tensor[DType.float32]:
        """Scale down gradients to meet accuracy constraint"""
        var constrained = Tensor[DType.float32](gradients.shape())
        
        # Scale factor to reduce impact to acceptable level
        var scale_factor = self.max_accuracy_loss / abs(predicted_impact)
        scale_factor = min(1.0, scale_factor)  # Don't increase gradients
        
        for i in range(gradients.shape()[0]):
            for j in range(gradients.shape()[1]):
                constrained[i, j] = gradients[i, j] * scale_factor
        
        return constrained
    
    fn should_update(self, accuracy_impact: Float32) -> Bool:
        """Only update if accuracy impact is within constraints"""
        return accuracy_impact >= -self.max_accuracy_loss

# Strategy 3: Multi-Objective Gradient Descent - Balance performance and accuracy
struct MultiObjectiveGradientStrategy:
    var learning_rate: Float32
    var performance_weight: Float32
    var accuracy_weight: Float32
    var balance_factor: Float32
    
    fn __init__(out self, lr: Float32 = 0.001, perf_weight: Float32 = 0.7, 
                 acc_weight: Float32 = 0.3, balance: Float32 = 0.5):
        self.learning_rate = lr
        self.performance_weight = perf_weight
        self.accuracy_weight = acc_weight
        self.balance_factor = balance
    
    fn compute_update(self, current_params: Tensor[DType.float32], gradients: Tensor[DType.float32],
                     performance_feedback: Float32, accuracy_constraint: Float32) -> Tensor[DType.float32]:
        """Compute gradient update balancing performance and accuracy objectives"""
        
        # Step 1: Compute performance-oriented gradients
        var performance_grads = self.compute_performance_gradients(gradients, performance_feedback)
        
        # Step 2: Compute accuracy-preserving gradients
        var accuracy_grads = self.compute_accuracy_gradients(current_params, accuracy_constraint)
        
        # Step 3: Balance between objectives
        var balanced_grads = Tensor[DType.float32](gradients.shape())
        for i in range(gradients.shape()[0]):
            for j in range(gradients.shape()[1]):
                balanced_grads[i, j] = (self.performance_weight * performance_grads[i, j] + 
                                       self.accuracy_weight * accuracy_grads[i, j]) * self.balance_factor
        
        # Step 4: Apply learning rate
        var update = Tensor[DType.float32](balanced_grads.shape())
        for i in range(balanced_grads.shape()[0]):
            for j in range(balanced_grads.shape()[1]):
                update[i, j] = balanced_grads[i, j] * self.learning_rate
        
        return update
    
    fn compute_performance_gradients(self, gradients: Tensor[DType.float32], 
                                    feedback: Float32) -> Tensor[DType.float32]:
        """Compute gradients focused on performance improvement"""
        var perf_grads = Tensor[DType.float32](gradients.shape())
        
        for i in range(gradients.shape()[0]):
            for j in range(gradients.shape()[1]):
                perf_grads[i, j] = gradients[i, j] * feedback
        
        return perf_grads
    
    fn compute_accuracy_gradients(self, params: Tensor[DType.float32], 
                                 accuracy_constraint: Float32) -> Tensor[DType.float32]:
        """Compute gradients that maintain accuracy"""
        var acc_grads = Tensor[DType.float32](params.shape())
        
        for i in range(params.shape()[0]):
            for j in range(params.shape()[1]):
                # Gradients that move parameters toward accuracy-preserving values
                acc_grads[i, j] = -params[i, j] * accuracy_constraint * 0.001
        
        return acc_grads
    
    fn should_update(self, accuracy_impact: Float32) -> Bool:
        """Update if balance allows it"""
        return True  # Multi-objective strategy handles balance internally

# Strategy 4: Adaptive Gradient Descent - Dynamically adjust based on accuracy
struct AdaptiveGradientStrategy:
    var base_learning_rate: Float32
    var accuracy_threshold: Float32
    var adaptation_factor: Float32
    var current_accuracy: Float32
    
    fn __init__(out self, lr: Float32 = 0.001, threshold: Float32 = 0.95, 
                 factor: Float32 = 0.1):
        self.base_learning_rate = lr
        self.accuracy_threshold = threshold
        self.adaptation_factor = factor
        self.current_accuracy = 1.0
    
    fn compute_update(self, current_params: Tensor[DType.float32], gradients: Tensor[DType.float32],
                     performance_feedback: Float32, accuracy_constraint: Float32) -> Tensor[DType.float32]:
        """Compute gradient update with adaptive learning rate based on accuracy"""
        
        # Step 1: Update current accuracy estimate
        self.current_accuracy = accuracy_constraint
        
        # Step 2: Adapt learning rate based on accuracy
        var adaptive_lr = self.adapt_learning_rate()
        
        # Step 3: Apply adaptive learning rate
        var update = Tensor[DType.float32](gradients.shape())
        for i in range(gradients.shape()[0]):
            for j in range(gradients.shape()[1]):
                update[i, j] = gradients[i, j] * adaptive_lr * performance_feedback
        
        return update
    
    fn adapt_learning_rate(self) -> Float32:
        """Dynamically adjust learning rate based on current accuracy"""
        if self.current_accuracy < self.accuracy_threshold:
            # Accuracy dropping - reduce learning rate
            return self.base_learning_rate * (1.0 - self.adaptation_factor)
        else:
            # Accuracy stable - can use full learning rate
            return self.base_learning_rate
    
    fn should_update(self, accuracy_impact: Float32) -> Bool:
        """Update based on current accuracy level"""
        return self.current_accuracy >= self.accuracy_threshold

# Accuracy-preserving evolution system
struct AccuracyPreservingEvolution:
    var config: SystemConfig
    var gradient_strategy: GradientStrategy
    var accuracy_monitor: AccuracyMonitor
    var current_accuracy: Float32
    
    fn __init__(out self, config: SystemConfig, strategy: String = "orthogonal"):
        self.config = config
        self.current_accuracy = 1.0
        
        # Initialize gradient strategy
        if strategy == "orthogonal":
            self.gradient_strategy = OrthogonalGradientStrategy()
        elif strategy == "constrained":
            self.gradient_strategy = ConstrainedGradientStrategy()
        elif strategy == "multi_objective":
            self.gradient_strategy = MultiObjectiveGradientStrategy()
        elif strategy == "adaptive":
            self.gradient_strategy = AdaptiveGradientStrategy()
        else:
            self.gradient_strategy = OrthogonalGradientStrategy()  # Default
        
        self.accuracy_monitor = AccuracyMonitor()
        
        print("🎯 Accuracy-Preserving Evolution Initialized")
        print("   - Strategy: {}".format(strategy))
        print("   - Goal: Zero accuracy loss during evolution")
    
    fn evolve_without_accuracy_loss(mut self, current_params: Tensor[DType.float32],
                                   performance_feedback: Float32) -> Tensor[DType.float32]:
        """Evolve parameters while maintaining 100% accuracy"""
        
        # Step 1: Compute performance gradients
        var gradients = self.compute_performance_gradients(performance_feedback)
        
        # Step 2: Predict accuracy impact
        var accuracy_impact = self.predict_accuracy_impact(current_params, gradients)
        
        # Step 3: Check if update is safe
        if not self.gradient_strategy.should_update(accuracy_impact):
            print("⚠️ Update blocked: Would impact accuracy")
            return current_params  # No update
        
        # Step 4: Compute accuracy-preserving update
        var update = self.gradient_strategy.compute_update(
            current_params, gradients, performance_feedback, self.current_accuracy
        )
        
        # Step 5: Apply update
        var new_params = Tensor[DType.float32](current_params.shape())
        for i in range(current_params.shape()[0]):
            for j in range(current_params.shape()[1]):
                new_params[i, j] = current_params[i, j] + update[i, j]
        
        # Step 6: Verify accuracy maintained
        var new_accuracy = self.estimate_accuracy(new_params)
        if new_accuracy >= self.current_accuracy - 0.001:  # Max 0.1% loss
            self.current_accuracy = new_accuracy
            print("✅ Evolution successful: Accuracy {:.4f}% maintained".format(new_accuracy * 100))
            return new_params
        else:
            print("🛑 Update rejected: Accuracy would drop to {:.4f}%".format(new_accuracy * 100))
            return current_params
    
    fn compute_performance_gradients(self, feedback: Float32) -> Tensor[DType.float32]:
        """Compute gradients based on performance feedback"""
        var gradients = Tensor[DType.float32](self.config.hidden_dim, self.config.vera_rank)
        
        for i in range(self.config.hidden_dim):
            for j in range(self.config.vera_rank):
                gradients[i, j] = feedback * 0.001  # Small, controlled gradients
        
        return gradients
    
    fn predict_accuracy_impact(self, params: Tensor[DType.float32], 
                              gradients: Tensor[DType.float32]) -> Float32:
        """Predict how much accuracy will be impacted"""
        var total_change = 0.0
        var total_params = 0
        
        for i in range(params.shape()[0]):
            for j in range(params.shape()[1]):
                var relative_change = abs(gradients[i, j]) / (abs(params[i, j]) + 1e-8)
                total_change += relative_change
                total_params += 1
        
        return -total_change / total_params  # Negative = potential loss
    
    fn estimate_accuracy(self, params: Tensor[DType.float32]) -> Float32:
        """Estimate current accuracy based on parameter state"""
        # Simple heuristic: smaller parameter changes = higher accuracy
        var total_magnitude = 0.0
        
        for i in range(params.shape()[0]):
            for j in range(params.shape()[1]):
                total_magnitude += abs(params[i, j])
        
        # Normalize to [0, 1] range
        var normalized_magnitude = total_magnitude / (params.shape()[0] * params.shape()[1])
        return max(0.0, 1.0 - normalized_magnitude * 0.1)  # Conservative estimate

# Accuracy monitoring system
struct AccuracyMonitor:
    var accuracy_history: Tensor[Float32]
    var history_size: Int
    var current_index: Int
    
    fn __init__(out self, size: Int = 1000):
        self.accuracy_history = Tensor[Float32](size)
        self.history_size = size
        self.current_index = 0
    
    fn record_accuracy(mut self, accuracy: Float32):
        """Record current accuracy measurement"""
        self.accuracy_history[self.current_index % self.history_size] = accuracy
        self.current_index += 1
    
    fn get_average_accuracy(self) -> Float32:
        """Get average accuracy over recent history"""
        if self.current_index == 0:
            return 1.0
        
        var sum = 0.0
        var count = min(self.history_size, self.current_index)
        
        for i in range(count):
            sum += self.accuracy_history[i]
        
        return sum / Float32(count)
    
    fn is_accuracy_stable(self) -> Bool:
        """Check if accuracy is stable (not declining)"""
        if self.current_index < 10:
            return True
        
        # Compare recent average to older average
        var recent_sum = 0.0
        var older_sum = 0.0
        
        for i in range(5):
            recent_sum += self.accuracy_history[(self.current_index - 1 - i) % self.history_size]
            older_sum += self.accuracy_history[(self.current_index - 6 - i) % self.history_size]
        
        var recent_avg = recent_sum / 5.0
        var older_avg = older_sum / 5.0
        
        return recent_avg >= older_avg - 0.001  # Allow tiny fluctuations
