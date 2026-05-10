# Module D: Heterogeneous MoE Router
# Dispatches tokens to three expert types:
# - Linguistic Experts: Standard Transformer blocks
# - Physics Expert: Ising Logic gate
# - Diffusion Expert: DiT block for spatial/video consistency
# Implements load balancing and expert selection logic

from tensor import Tensor
from math import exp, sqrt
from nif_sovereign.config.nif_config import NIFConfig

struct HeterogeneousMoERouter:
    var config: NIFConfig
    var num_experts: Int
    var hidden_dim: Int
    var expert_types: List[String]
    var gating_network: Tensor[DType.float32]
    var load_balancer: Tensor[DType.float32]
    var expert_capacity: Int

    fn __init__(inout self, config: NIFConfig):
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_dim
        self.expert_capacity = config.hidden_dim // self.num_experts

        # Define expert types
        self.expert_types = self.initialize_expert_types()

        # Initialize gating network and load balancer
        self.gating_network = self.initialize_gating_network()
        self.load_balancer = self.initialize_load_balancer()

        print("🔀 Heterogeneous MoE Router Initialized")
        print("   - Expert Types: Linguistic, Physics, Diffusion")
        print("   - Expert Capacity: {}".format(self.expert_capacity))

    fn initialize_expert_types(inout self) -> List[String]:
        # Define the three expert types
        var types = ["linguistic", "physics", "diffusion"]
        return types

    fn initialize_gating_network(inout self) -> Tensor[DType.float32]:
        # Initialize gating network weights for expert selection
        var gating = Tensor[DType.float32](self.hidden_dim, self.num_experts)

        for i in range(self.hidden_dim):
            for j in range(self.num_experts):
                # Initialize with small random weights
                gating[i, j] = 0.01 * (Float32(i + j) - Float32(self.hidden_dim) / 2.0)

        return gating

    fn initialize_load_balancer(inout self) -> Tensor[DType.float32]:
        # Initialize load balancing parameters
        var balancer = Tensor[DType.float32](self.num_experts)

        for i in range(self.num_experts):
            balancer[i] = 1.0 / Float32(self.num_experts)  # Equal initial load

        return balancer

    fn dispatch(inout self, oscillated_embeddings: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # Main dispatch function - route tokens to appropriate experts
        var shape = oscillated_embeddings.shape()
        var batch_size = shape[0]
        var seq_len = shape[1]

        # Compute expert assignments for each token
        var expert_assignments = self.compute_expert_assignments(oscillated_embeddings)

        # Process tokens through assigned experts
        var expert_outputs = self.process_through_experts(oscillated_embeddings, expert_assignments)

        # Combine expert outputs
        var combined_output = self.combine_expert_outputs(expert_outputs, expert_assignments)

        return combined_output

    fn combine_expert_outputs(inout self, expert_outputs: Tensor[DType.float32], assignments: Tensor[Int]) -> Tensor[DType.float32]:
        # Combine outputs from different experts
        var shape = expert_outputs.shape()
        var combined = Tensor[DType.float32](shape[0], shape[1], shape[2])

        # Simple weighted combination based on expert usage
        var expert_weights = self.compute_expert_weights(assignments)

        for b in range(shape[0]):
            for s in range(shape[1]):
                var expert_id = assignments[b, s]
                var weight = expert_weights[expert_id]

                for dim in range(shape[2]):
                    combined[b, s, dim] = weight * expert_outputs[b, s, dim]

        return combined

    fn compute_expert_weights(inout self, assignments: Tensor[Int]) -> Tensor[DType.float32]:
        # Compute weights for expert combination based on usage
        var weights = Tensor[DType.float32](self.num_experts)
        var total_assignments = 0

        # Count expert assignments
        for expert in range(self.num_experts):
            weights[expert] = 0.0

        var shape = assignments.shape()
        for b in range(shape[0]):
            for s in range(shape[1]):
                var expert_id = assignments[b, s]
                weights[expert_id] += 1.0
                total_assignments += 1.0

        # Normalize weights
        if total_assignments > 0.0:
            for expert in range(self.num_experts):
                weights[expert] /= total_assignments
        else:
            for expert in range(self.num_experts):
                weights[expert] = 1.0 / Float32(self.num_experts)

        return weights

    fn update_load_balancer(inout self, expert_id: Int):
        # Update load balancer based on expert usage
        var decay_rate = 0.99
        var learning_rate = 0.01

        # Decay existing loads
        for expert in range(self.num_experts):
            self.load_balancer[expert] *= decay_rate

        # Increment used expert
        self.load_balancer[expert_id] += learning_rate

    fn get_expert_statistics(inout self) -> Tensor[DType.float32]:
        # Get current expert load statistics
        return self.load_balancer

    fn softmax(inout self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply softmax function for expert selection"""
        var shape = input.shape()
        var output = Tensor[DType.float32](shape)

        for i in range(shape[0]):
            # Find max for numerical stability
            var max_val = input[i, 0]
            for j in range(shape[1]):
                if input[i, j] > max_val:
                    max_val = input[i, j]

            # Compute exp and sum
            var sum_exp = 0.0
            for j in range(shape[1]):
                var exp_val = exp(input[i, j] - max_val)
                output[i, j] = exp_val
                sum_exp += exp_val

            # Normalize
            if sum_exp > 0.0:
                for j in range(shape[1]):
                    output[i, j] /= sum_exp

        return output
