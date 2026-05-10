# Muon Optimizer Implementation
# Orthonormal updates for stable training
# Physics-aware optimization for NIF architecture

from tensor import Tensor
from math import sqrt, exp, tanh, sin, cos, abs
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.core.custom_training_logic import SovereignTensor

# Muon Optimizer Core
struct MuonOptimizer:
    var config: SystemConfig
    var learning_rate: Float32
    var momentum: Float32
    var orthonormal_frequency: Int
    var update_counter: Int
    var parameter_history: Tensor[DType.float32]
    var gradient_history: Tensor[DType.float32]
    
    fn __init__(out self, config: SystemConfig):
        self.config = config
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.orthonormal_frequency = 10  # Orthonormalize every 10 steps
        self.update_counter = 0
        
        # Initialize history buffers
        var param_size = config.hidden_dim * config.adapter_rank
        self.parameter_history = Tensor[DType.float32](param_size)
        self.gradient_history = Tensor[DType.float32](param_size)
        
        print("🔥 Muon Optimizer Initialized")
        print("   - Learning Rate: {:.6f}".format(self.learning_rate))
        print("   - Momentum: {:.4f}".format(self.momentum))
        print("   - Orthonormal Frequency: Every {} steps".format(self.orthonormal_frequency))
        print("   - Stable Training: Active")
    
    fn muon_update(mut self, parameters: Tensor[DType.float32], 
                   gradients: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Perform Muon update with orthonormal constraints"""
        
        # Step 1: Apply momentum to gradients
        var momentum_gradients = self.apply_momentum(gradients)
        
        # Step 2: Update parameters with orthonormal constraint
        var updated_parameters = self.orthonormal_update(parameters, momentum_gradients)
        
        # Step 3: Periodic orthonormalization
        self.update_counter += 1
        if self.update_counter % self.orthonormal_frequency == 0:
            updated_parameters = self.orthonormalize_parameters(updated_parameters)
            print("🔄 Muon Orthonormalization Applied (Step {})".format(self.update_counter))
        
        return updated_parameters
    
    fn apply_momentum(mut self, gradients: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply momentum to gradients"""
        var momentum_gradients = Tensor[DType.float32](gradients.shape())
        
        for i in range(gradients.shape()[0]):
            for j in range(gradients.shape()[1]):
                # Update gradient history
                var grad_idx = i * gradients.shape()[1] + j
                if grad_idx < self.gradient_history.shape()[0]:
                    self.gradient_history[grad_idx] = self.momentum * self.gradient_history[grad_idx] + gradients[i, j]
                    momentum_gradients[i, j] = self.gradient_history[grad_idx]
                else:
                    momentum_gradients[i, j] = gradients[i, j]
        
        return momentum_gradients
    
    fn orthonormal_update(self, parameters: Tensor[DType.float32], 
                          gradients: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Update parameters with orthonormal constraint"""
        var updated = Tensor[DType.float32](parameters.shape())
        
        for i in range(parameters.shape()[0]):
            for j in range(parameters.shape()[1]):
                # Standard gradient update
                updated[i, j] = parameters[i, j] - self.learning_rate * gradients[i, j]
                
                # Apply orthonormal constraint (simplified)
                var norm = 0.0
                for k in range(parameters.shape()[1]):
                    norm += updated[i, k] * updated[i, k]
                norm = sqrt(norm)
                
                if norm > 0.0:
                    for k in range(parameters.shape()[1]):
                        updated[i, k] /= norm
        
        return updated
    
    fn orthonormalize_parameters(self, parameters: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Perform full orthonormalization of parameters"""
        var orthonormalized = Tensor[DType.float32](parameters.shape())
        
        # Gram-Schmidt process (simplified)
        for i in range(parameters.shape()[0]):
            for j in range(parameters.shape()[1]):
                orthonormalized[i, j] = parameters[i, j]
                
                # Remove components in previous directions
                for k in range(j):
                    var dot_product = 0.0
                    var norm_squared = 0.0
                    
                    for m in range(parameters.shape()[0]):
                        dot_product += orthonormalized[m, k] * parameters[m, j]
                        norm_squared += orthonormalized[m, k] * orthonormalized[m, k]
                    
                    if norm_squared > 0.0:
                        var projection = dot_product / norm_squared
                        for m in range(parameters.shape()[0]):
                            orthonormalized[m, j] -= projection * orthonormalized[m, k]
                
                # Normalize
                var norm = 0.0
                for m in range(parameters.shape()[0]):
                    norm += orthonormalized[m, j] * orthonormalized[m, j]
                norm = sqrt(norm)
                
                if norm > 0.0:
                    for m in range(parameters.shape()[0]):
                        orthonormalized[m, j] /= norm
        
        return orthonormalized
    
    fn get_optimizer_stats(self) -> String:
        """Get optimizer statistics"""
        var stats = "🔥 Muon Optimizer Statistics\n"
        stats += "=" * 30 + "\n"
        stats += "Update Counter: {}\n".format(self.update_counter)
        stats += "Learning Rate: {:.6f}\n".format(self.learning_rate)
        stats += "Momentum: {:.4f}\n".format(self.momentum)
        stats += "Orthonormal Frequency: {}\n".format(self.orthonormal_frequency)
        stats += "Stable Training: Active\n"
        
        return stats
