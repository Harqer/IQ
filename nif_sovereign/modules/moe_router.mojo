# Module D: Heterogeneous MoE Router
# Dispatches tokens to three expert types:
# - Linguistic Experts: Standard Transformer blocks
# - Physics Expert: Ising Logic gate
# - Diffusion Expert: DiT block for spatial/video consistency
# Implements load balancing and expert selection logic

from tensor import Tensor
from math import exp, softmax, sqrt

struct HeterogeneousMoERouter:
    var config: NIFConfig
    var num_experts: Int
    var hidden_dim: Int
    var expert_types: Tensor[String]
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

    @parameter
    fn initialize_expert_types() -> Tensor[String]:
        # Define the three expert types
        var types = Tensor[String](3)
        types[0] = "linguistic"  # Standard Transformer blocks
        types[1] = "physics"     # Ising Logic gate
        types[2] = "diffusion"   # DiT block for spatial consistency
        return types

    @parameter
    fn initialize_gating_network() -> Tensor[DType.float32]:
        # Initialize gating network weights for expert selection
        var gating = Tensor[DType.float32](self.hidden_dim, self.num_experts)

        for i in range(self.hidden_dim):
            for j in range(self.num_experts):
                # Initialize with small random weights
                gating[i, j] = 0.01 * (Float32(i + j) - Float32(self.hidden_dim) / 2.0)

        return gating

    @parameter
    fn initialize_load_balancer() -> Tensor[DType.float32]:
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

    fn compute_expert_assignments(inout self, embeddings: Tensor[DType.float32]) -> Tensor[Int]:
        # Compute which expert each token should be routed to
        var shape = embeddings.shape()
        var batch_size = shape[0]
        var seq_len = shape[1]
        var assignments = Tensor[Int](batch_size, seq_len)

        for b in range(batch_size):
            for s in range(seq_len):
                # Get token embedding
                var token_embedding = embeddings[b, s, :]

                # Compute gating scores
                var gating_scores = self.compute_gating_scores(token_embedding)

                # Apply load balancing
                var balanced_scores = self.apply_load_balancing(gating_scores)

                # Select expert with highest score
                var expert_id = self.select_expert(balanced_scores)
                assignments[b, s] = expert_id

                # Update load balancer
                self.update_load_balancer(expert_id)

        return assignments

    fn compute_gating_scores(inout self, token_embedding: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # Compute gating scores for each expert
        var scores = Tensor[DType.float32](self.num_experts)

        for expert in range(self.num_experts):
            var score = 0.0

            # Dot product with gating network
            for dim in range(self.hidden_dim):
                score += token_embedding[dim] * self.gating_network[dim, expert]

            scores[expert] = score

        return scores

    fn apply_load_balancing(inout self, scores: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # Apply load balancing to prevent expert overload
        var balanced_scores = Tensor[DType.float32](self.num_experts)

        for expert in range(self.num_experts):
            # Apply load balancing penalty
            var load_penalty = self.load_balancer[expert] * 0.1
            balanced_scores[expert] = scores[expert] - load_penalty

        return balanced_scores

    fn select_expert(inout self, scores: Tensor[DType.float32]) -> Int:
        # Select expert with highest score
        var max_score = scores[0]
        var selected_expert = 0

        for expert in range(1, self.num_experts):
            if scores[expert] > max_score:
                max_score = scores[expert]
                selected_expert = expert

        return selected_expert

    fn process_through_experts(inout self, embeddings: Tensor[DType.float32], assignments: Tensor[Int]) -> Tensor[DType.float32]:
        # Process tokens through their assigned experts
        var shape = embeddings.shape()
        var batch_size = shape[0]
        var seq_len = shape[1]
        var outputs = Tensor[DType.float32](batch_size, seq_len, self.hidden_dim)

        for b in range(batch_size):
            for s in range(seq_len):
                var expert_id = assignments[b, s]
                var token_embedding = embeddings[b, s, :]

                # Process through appropriate expert
                var expert_output = self.call_expert(expert_id, token_embedding)

                # Store output
                for dim in range(self.hidden_dim):
                    outputs[b, s, dim] = expert_output[dim]

        return outputs

    fn call_expert(inout self, expert_id: Int, token_embedding: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # Call specific expert based on ID
        var output = Tensor[DType.float32](self.hidden_dim)

        if expert_id == 0:
            # Linguistic Expert - Standard Transformer block
            output = self.linguistic_expert(token_embedding)
        elif expert_id == 1:
            # Physics Expert - Ising Logic gate
            output = self.physics_expert(token_embedding)
        elif expert_id == 2:
            # Diffusion Expert - DiT block
            output = self.diffusion_expert(token_embedding)

        return output

    fn linguistic_expert(inout self, embedding: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # Linguistic expert - standard transformer processing
        var output = Tensor[DType.float32](self.hidden_dim)

        # Simplified transformer block
        for i in range(self.hidden_dim):
            # Self-attention simulation
            var attention_score = 0.0
            for j in range(self.hidden_dim):
                attention_score += embedding[j] * embedding[j]

            # Feed-forward simulation
            output[i] = tanh(attention_score * 0.1 + embedding[i])

        return output

    fn physics_expert(inout self, embedding: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # Physics expert - Ising logic simulation
        var output = Tensor[DType.float32](self.hidden_dim)

        # Simulate Ising interaction
        for i in range(self.hidden_dim):
            var spin_sum = 0.0
            for j in range(max(0, i-1), min(self.hidden_dim, i+2)):
                if j != i:
                    spin_sum += embedding[j]

            # Apply Ising-like interaction
            output[i] = tanh(embedding[i] + 0.2 * spin_sum)

        return output

    fn diffusion_expert(inout self, embedding: Tensor[DType.float32]) -> Tensor[DType.float32]:
        # Diffusion expert - DiT (Diffusion Transformer) block
        var output = Tensor[DType.float32](self.hidden_dim)

        # Simulate diffusion process
        for i in range(self.hidden_dim):
            var diffusion_sum = 0.0
            var neighbors = 0

            # Spatial neighborhood
            for j in range(max(0, i-2), min(self.hidden_dim, i+3)):
                if j != i:
                    diffusion_sum += embedding[j]
                    neighbors += 1

            # Apply diffusion
            if neighbors > 0:
                output[i] = 0.7 * embedding[i] + 0.3 * diffusion_sum / Float32(neighbors)
            else:
                output[i] = embedding[i]

        return output

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
