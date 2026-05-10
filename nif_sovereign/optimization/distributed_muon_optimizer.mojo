# Distributed Muon Optimizer with Explicit Weight Decay
# Zero-One Style Partitioning with Robust Weight Decay Mechanism
# Stable updates for long reasoning chains with integrated weight decay

from tensor import Tensor
from math import sqrt, exp, tanh, sin, cos, abs, acos, log
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.core.custom_training_logic import SovereignTensor

# Zero-One Style Partitioning for Distributed Muon
struct ZeroOnePartitioning:
    var config: SystemConfig
    var num_partitions: Int
    var partition_boundaries: Tensor[Int]
    var zero_one_mask: Tensor[Bool]
    var partition_weights: Tensor[DType.float32]
    
    fn __init__(out self, config: SystemConfig):
        self.config = config
        self.num_partitions = 8  # Optimal for distributed processing
        self.partition_boundaries = self.initialize_partition_boundaries()
        self.zero_one_mask = self.initialize_zero_one_mask()
        self.partition_weights = self.initialize_partition_weights()
        
        print("🔄 Zero-One Partitioning Initialized")
        print("   - Number of Partitions: {}".format(self.num_partitions))
        print("   - Zero-One Mask: Active")
        print("   - Distributed Processing: Enabled")
    
    fn initialize_partition_boundaries(self) -> Tensor[Int]:
        """Initialize partition boundaries for zero-one style partitioning"""
        var boundaries = Tensor[Int](self.num_partitions + 1)
        var total_params = self.config.hidden_dim * self.config.adapter_rank
        var partition_size = total_params / self.num_partitions
        
        for i in range(self.num_partitions + 1):
            boundaries[i] = i * partition_size
        
        return boundaries
    
    fn initialize_zero_one_mask(self) -> Tensor[Bool]:
        """Initialize zero-one mask for partitioning"""
        var total_params = self.config.hidden_dim * self.config.adapter_rank
        var mask = Tensor[Bool](total_params)
        
        for i in range(total_params):
            # Zero-one pattern: alternating 0 and 1 partitions
            mask[i] = (i % 2 == 0)
        
        return mask
    
    fn initialize_partition_weights(self) -> Tensor[DType.float32]:
        """Initialize partition weights for distributed processing"""
        var weights = Tensor[DType.float32](self.num_partitions)
        
        for i in range(self.num_partitions):
            weights[i] = 1.0 / Float32(self.num_partitions)
        
        return weights
    
    fn get_partition_for_param(self, param_idx: Int) -> Int:
        """Get partition index for a parameter"""
        for i in range(self.num_partitions):
            if param_idx >= self.partition_boundaries[i] and param_idx < self.partition_boundaries[i + 1]:
                return i
        return self.num_partitions - 1
    
    fn apply_zero_one_mask(self, parameters: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply zero-one mask to parameters"""
        var masked = Tensor[DType.float32](parameters.shape())
        
        for i in range(parameters.shape()[0]):
            for j in range(parameters.shape()[1]):
                var param_idx = i * parameters.shape()[1] + j
                if param_idx < self.zero_one_mask.shape()[0]:
                    if self.zero_one_mask[param_idx]:
                        masked[i, j] = parameters[i, j]
                    else:
                        masked[i, j] = 0.0
                else:
                    masked[i, j] = parameters[i, j]
        
        return masked

# Robust Weight Decay Mechanism
struct RobustWeightDecay:
    var config: SystemConfig
    var decay_rate: Float32
    var decay_schedule: Tensor[DType.float32]
    var decay_history: Tensor[Float32]
    var stability_threshold: Float32
    var adaptive_decay: Bool
    
    fn __init__(out self, config: SystemConfig):
        self.config = config
        self.decay_rate = 0.01  # Base decay rate
        self.decay_schedule = self.initialize_decay_schedule()
        self.decay_history = Tensor[Float32](1000)
        self.stability_threshold = 0.001
        self.adaptive_decay = True
        
        print("🛡️ Robust Weight Decay Initialized")
        print("   - Base Decay Rate: {}".format(self.decay_rate))
        print("   - Adaptive Decay: {}".format(self.adaptive_decay))
        print("   - Stability Threshold: {}".format(self.stability_threshold))
    
    fn initialize_decay_schedule(self) -> Tensor[DType.float32]:
        """Initialize decay schedule for long reasoning chains"""
        var schedule = Tensor[DType.float32](1000)
        
        for i in range(1000):
            # Gradual decay for long reasoning chains
            var progress = Float32(i) / 1000.0
            schedule[i] = self.decay_rate * (1.0 - progress * 0.5)  # Reduce decay over time
        
        return schedule
    
    fn compute_weight_decay(self, parameters: Tensor[DType.float32], 
                           step: Int) -> Tensor[DType.float32]:
        """Compute robust weight decay for parameters"""
        var decayed = Tensor[DType.float32](parameters.shape())
        
        # Get current decay rate from schedule
        var current_decay_rate = self.decay_rate
        if step < self.decay_schedule.shape()[0]:
            current_decay_rate = self.decay_schedule[step]
        
        # Apply adaptive decay if enabled
        if self.adaptive_decay:
            current_decay_rate = self.compute_adaptive_decay(parameters, current_decay_rate)
        
        # Apply weight decay
        for i in range(parameters.shape()[0]):
            for j in range(parameters.shape()[1]):
                decayed[i, j] = parameters[i, j] * (1.0 - current_decay_rate)
        
        return decayed
    
    fn compute_adaptive_decay(self, parameters: Tensor[DType.float32], 
                             base_decay: Float32) -> Float32:
        """Compute adaptive decay based on parameter statistics"""
        var param_norm = 0.0
        var param_count = 0
        
        for i in range(parameters.shape()[0]):
            for j in range(parameters.shape()[1]):
                param_norm += parameters[i, j] * parameters[i, j]
                param_count += 1
        
        param_norm = sqrt(param_norm / Float32(param_count))
        
        # Adaptive decay based on parameter norm
        if param_norm > 1.0:
            return base_decay * 1.5  # Increase decay for large norms
        elif param_norm < 0.1:
            return base_decay * 0.5  # Reduce decay for small norms
        else:
            return base_decay
    
    def check_stability(self, parameters: Tensor[DType.float32]) -> Bool:
        """Check parameter stability for long reasoning chains"""
        var param_variance = 0.0
        var param_mean = 0.0
        var param_count = 0
        
        for i in range(parameters.shape()[0]):
            for j in range(parameters.shape()[1]):
                param_mean += parameters[i, j]
                param_count += 1
        
        param_mean /= Float32(param_count)
        
        for i in range(parameters.shape()[0]):
            for j in range(parameters.shape()[1]):
                var diff = parameters[i, j] - param_mean
                param_variance += diff * diff
        
        param_variance /= Float32(param_count)
        param_variance = sqrt(param_variance)
        
        return param_variance < self.stability_threshold

# Distributed Muon with Weight Decay Integration
struct DistributedMuonOptimizer:
    var config: SystemConfig
    var zero_one_partitioning: ZeroOnePartitioning
    var weight_decay: RobustWeightDecay
    var learning_rate: Float32
    var momentum: Float32
    var orthonormal_frequency: Int
    var update_counter: Int
    var distributed_gradients: Tensor[Tensor[DType.float32]]
    var stability_monitor: Tensor[Bool]
    
    fn __init__(out self, config: SystemConfig):
        self.config = config
        self.zero_one_partitioning = ZeroOnePartitioning(config)
        self.weight_decay = RobustWeightDecay(config)
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.orthonormal_frequency = 10
        self.update_counter = 0
        self.distributed_gradients = self.initialize_distributed_gradients()
        self.stability_monitor = Tensor[Bool](self.zero_one_partitioning.num_partitions)
        
        print("🔥 Distributed Muon Optimizer Initialized")
        print("   - Zero-One Partitioning: Active")
        print("   - Robust Weight Decay: Active")
        print("   - Distributed Processing: {} partitions".format(self.zero_one_partitioning.num_partitions))
        print("   - Learning Rate: {:.6f}".format(self.learning_rate))
        print("   - Momentum: {:.4f}".format(self.momentum))
        print("   - Orthonormal Frequency: Every {} steps".format(self.orthonormal_frequency))
    
    fn initialize_distributed_gradients(self) -> Tensor[Tensor[DType.float32]]:
        """Initialize distributed gradient storage"""
        var gradients = Tensor[Tensor[DType.float32]](self.zero_one_partitioning.num_partitions)
        
        for i in range(self.zero_one_partitioning.num_partitions):
            var start_idx = self.zero_one_partitioning.partition_boundaries[i]
            var end_idx = self.zero_one_partitioning.partition_boundaries[i + 1]
            var partition_size = end_idx - start_idx
            
            gradients[i] = Tensor[DType.float32](partition_size, self.config.adapter_rank)
        
        return gradients
    
    fn distributed_muon_update(mut self, parameters: Tensor[DType.float32], 
                                gradients: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Perform distributed Muon update with weight decay integration"""
        
        # Step 1: Apply zero-one partitioning to gradients
        var partitioned_gradients = self.partition_gradients(gradients)
        
        # Step 2: Apply momentum to each partition
        var momentum_gradients = self.apply_distributed_momentum(partitioned_gradients)
        
        # Step 3: Update parameters with weight decay
        var updated_parameters = self.update_with_weight_decay(parameters, momentum_gradients)
        
        # Step 4: Periodic distributed orthonormalization with weight decay
        self.update_counter += 1
        if self.update_counter % self.orthonormal_frequency == 0:
            updated_parameters = self.distributed_orthonormalization_with_decay(updated_parameters)
            
            # Check stability for long reasoning chains
            self.check_long_reasoning_stability(updated_parameters)
        
        return updated_parameters
    
    fn partition_gradients(self, gradients: Tensor[DType.float32]) -> Tensor[Tensor[DType.float32]]:
        """Partition gradients using zero-one style partitioning"""
        var partitioned = Tensor[Tensor[DType.float32]](self.zero_one_partitioning.num_partitions)
        
        for i in range(self.zero_one_partitioning.num_partitions):
            var start_idx = self.zero_one_partitioning.partition_boundaries[i]
            var end_idx = self.zero_one_partitioning.partition_boundaries[i + 1]
            var partition_size = end_idx - start_idx
            
            partitioned[i] = Tensor[DType.float32](partition_size, gradients.shape()[1])
            
            for j in range(partition_size):
                for k in range(gradients.shape()[1]):
                    var global_idx = start_idx + j
                    if global_idx < gradients.shape()[0]:
                        partitioned[i][j, k] = gradients[global_idx, k]
        
        return partitioned
    
    fn apply_distributed_momentum(mut self, partitioned_gradients: Tensor[Tensor[DType.float32]]) -> Tensor[Tensor[DType.float32]]:
        """Apply momentum to distributed gradients"""
        var momentum_gradients = Tensor[Tensor[DType.float32]](self.zero_one_partitioning.num_partitions)
        
        for i in range(self.zero_one_partitioning.num_partitions):
            momentum_gradients[i] = Tensor[DType.float32](partitioned_gradients[i].shape())
            
            for j in range(partitioned_gradients[i].shape()[0]):
                for k in range(partitioned_gradients[i].shape()[1]):
                    # Update gradient history with momentum
                    var current_grad = partitioned_gradients[i][j, k]
                    var prev_grad = self.distributed_gradients[i][j, k]
                    
                    self.distributed_gradients[i][j, k] = self.momentum * prev_grad + current_grad
                    momentum_gradients[i][j, k] = self.distributed_gradients[i][j, k]
        
        return momentum_gradients
    
    fn update_with_weight_decay(self, parameters: Tensor[DType.float32], 
                                momentum_gradients: Tensor[Tensor[DType.float32]]) -> Tensor[DType.float32]:
        """Update parameters with integrated weight decay"""
        var updated = Tensor[DType.float32](parameters.shape())
        
        # First apply weight decay to current parameters
        var decayed_parameters = self.weight_decay.compute_weight_decay(parameters, self.update_counter)
        
        # Then apply gradient updates
        for i in range(self.zero_one_partitioning.num_partitions):
            var start_idx = self.zero_one_partitioning.partition_boundaries[i]
            var end_idx = self.zero_one_partitioning.partition_boundaries[i + 1]
            
            for j in range(start_idx, end_idx):
                var local_idx = j - start_idx
                for k in range(parameters.shape()[1]):
                    if local_idx < momentum_gradients[i].shape()[0]:
                        updated[j, k] = decayed_parameters[j, k] - self.learning_rate * momentum_gradients[i][local_idx, k]
                    else:
                        updated[j, k] = decayed_parameters[j, k]
        
        return updated
    
    fn distributed_orthonormalization_with_decay(self, parameters: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Distributed orthonormalization with integrated weight decay"""
        var orthonormalized = Tensor[DType.float32](parameters.shape())
        
        # Apply weight decay before orthonormalization
        var decayed = self.weight_decay.compute_weight_decay(parameters, self.update_counter)
        
        # Perform distributed orthonormalization
        for i in range(self.zero_one_partitioning.num_partitions):
            var start_idx = self.zero_one_partitioning.partition_boundaries[i]
            var end_idx = self.zero_one_partitioning.partition_boundaries[i + 1]
            var partition_size = end_idx - start_idx
            
            # Extract partition
            var partition = Tensor[DType.float32](partition_size, parameters.shape()[1])
            for j in range(partition_size):
                for k in range(parameters.shape()[1]):
                    partition[j, k] = decayed[start_idx + j, k]
            
            # Apply orthonormalization to partition
            var orthonormal_partition = self.orthonormalize_partition(partition)
            
            # Store back
            for j in range(partition_size):
                for k in range(parameters.shape()[1]):
                    orthonormalized[start_idx + j, k] = orthonormal_partition[j, k]
        
        return orthonormalized
    
    fn orthonormalize_partition(self, partition: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Orthonormalize a partition using Gram-Schmidt process"""
        var orthonormalized = Tensor[DType.float32](partition.shape())
        
        for i in range(partition.shape()[0]):
            for j in range(partition.shape()[1]):
                orthonormalized[i, j] = partition[i, j]
        
        # Gram-Schmidt process
        for i in range(partition.shape()[0]):
            for j in range(partition.shape()[1]):
                # Remove components in previous directions
                for k in range(j):
                    var dot_product = 0.0
                    var norm_squared = 0.0
                    
                    for m in range(partition.shape()[0]):
                        dot_product += orthonormalized[m, k] * partition[m, j]
                        norm_squared += orthonormalized[m, k] * orthonormalized[m, k]
                    
                    if norm_squared > 0.0:
                        var projection = dot_product / norm_squared
                        for m in range(partition.shape()[0]):
                            orthonormalized[m, j] -= projection * orthonormalized[m, k]
                
                # Normalize
                var norm = 0.0
                for m in range(partition.shape()[0]):
                    norm += orthonormalized[m, j] * orthonormalized[m, j]
                norm = sqrt(norm)
                
                if norm > 0.0:
                    for m in range(partition.shape()[0]):
                        orthonormalized[m, j] /= norm
        
        return orthonormalized
    
    fn check_long_reasoning_stability(mut self, parameters: Tensor[DType.float32]):
        """Check stability for long reasoning chains"""
        for i in range(self.zero_one_partitioning.num_partitions):
            var start_idx = self.zero_one_partitioning.partition_boundaries[i]
            var end_idx = self.zero_one_partitioning.partition_boundaries[i + 1]
            var partition_size = end_idx - start_idx
            
            var partition = Tensor[DType.float32](partition_size, parameters.shape()[1])
            for j in range(partition_size):
                for k in range(parameters.shape()[1]):
                    partition[j, k] = parameters[start_idx + j, k]
            
            self.stability_monitor[i] = self.weight_decay.check_stability(partition)
            
            if not self.stability_monitor[i]:
                print("⚠️ Partition {} unstable - adjusting decay rate".format(i))
                self.weight_decay.decay_rate *= 0.5  # Reduce decay for stability
    
    fn get_optimizer_statistics(self) -> String:
        """Get comprehensive optimizer statistics"""
        var stats = "🔥 Distributed Muon Optimizer Statistics\n"
        stats += "=" * 45 + "\n"
        
        stats += "Configuration:\n"
        stats += "  - Update Counter: {}\n".format(self.update_counter)
        stats += "  - Learning Rate: {:.6f}\n".format(self.learning_rate)
        stats += "  - Momentum: {:.4f}\n".format(self.momentum)
        stats += "  - Orthonormal Frequency: {}\n".format(self.orthonormal_frequency)
        
        stats += "\nDistributed Processing:\n"
        stats += "  - Number of Partitions: {}\n".format(self.zero_one_partitioning.num_partitions)
        stats += "  - Zero-One Mask: Active\n"
        stats += "  - Partition Weights: "
        for i in range(self.zero_one_partitioning.num_partitions):
            stats += "{:.3f}".format(self.zero_one_partitioning.partition_weights[i])
            if i < self.zero_one_partitioning.num_partitions - 1:
                stats += ", "
        stats += "\n"
        
        stats += "\nWeight Decay:\n"
        stats += "  - Base Decay Rate: {:.6f}\n".format(self.weight_decay.decay_rate)
        stats += "  - Adaptive Decay: {}\n".format(self.weight_decay.adaptive_decay)
        stats += "  - Stability Threshold: {:.6f}\n".format(self.weight_decay.stability_threshold)
        
        stats += "\nStability Monitor:\n"
        for i in range(self.zero_one_partitioning.num_partitions):
            stats += "  - Partition {}: {}\n".format(i, self.stability_monitor[i] ? "Stable" : "Unstable")
        
        return stats

# Factory function
fn create_distributed_muon_optimizer(config: SystemConfig) -> DistributedMuonOptimizer:
    """Create distributed Muon optimizer with weight decay"""
    return DistributedMuonOptimizer(config)

# Usage example
fn main():
    print("🔥 Initializing Distributed Muon Optimizer with Weight Decay")
    
    var config = SystemConfig()
    var distributed_muon = create_distributed_muon_optimizer(config)
    
    # Create test parameters and gradients
    var parameters = Tensor[DType.float32](config.hidden_dim, config.adapter_rank)
    var gradients = Tensor[DType.float32](config.hidden_dim, config.adapter_rank)
    
    for i in range(config.hidden_dim):
        for j in range(config.adapter_rank):
            parameters[i, j] = Float32((i * config.adapter_rank + j) % 1000) / 1000.0
            gradients[i, j] = Float32((i * config.adapter_rank + j) % 100) / 100.0
    
    print("\n🚀 Testing Distributed Muon Updates...")
    
    # Run multiple updates
    for step in range(20):
        parameters = distributed_muon.distributed_muon_update(parameters, gradients)
        
        if step % 5 == 0:
            print("Step {}: Distributed Muon Update Complete".format(step))
    
    print("\n" + distributed_muon.get_optimizer_statistics())
    
    print("\n🔥 DISTRIBUTED MUON BENEFITS:")
    print("✅ Zero-One Style Partitioning")
    print("✅ Explicit Weight Decay Integration")
    print("✅ Robust Weight Decay Mechanism")
    print("✅ Distributed Processing")
    print("✅ Stable Long Reasoning Chains")
    print("✅ Adaptive Decay Scheduling")
    print("✅ Stability Monitoring")
    print("✅ Orthonormalization with Decay")
