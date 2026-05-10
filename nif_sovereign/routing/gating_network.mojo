# Gating Network
# Atomic component for manifold-aware expert selection in MoE systems
# Single responsibility: geodesic distance-based expert selection (replaces softmax)

from tensor import Tensor
from math import sqrt, exp, acos, cos, sin, abs, tanh
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.interfaces.gating_interface import GatingInterface, GatingConfig

# Geodesic distance calculator
struct GeodesicCalculator:
    var manifold_curvature: Float64
    var geodesic_tolerance: Float32

    fn __init__(manifold_curvature: Float64, geodesic_tolerance: Float32 = 1e-6):
        self.manifold_curvature = manifold_curvature
        self.geodesic_tolerance = geodesic_tolerance

    fn compute_lorentzian_dot_product(self, point1: Tensor[DType.float32], point2: Tensor[DType.float32]) -> Float32:
        """Compute Lorentzian dot product"""
        var dot_product = 0.0

        for i in range(point1.shape()[0]):
            if i == 0:
                # Time component (negative in Lorentzian metric)
                dot_product -= point1[i] * point2[i]
            else:
                # Space components (positive in Lorentzian metric)
                dot_product += point1[i] * point2[i]

        return dot_product

    fn compute_norm(self, point: Tensor[DType.float32]) -> Float32:
        """Compute norm on Lorentzian manifold"""
        var lorentzian_norm_sq = self.compute_lorentzian_dot_product(point, point)

        if lorentzian_norm_sq < 0:
            return sqrt(-lorentzian_norm_sq)
        else:
            return sqrt(lorentzian_norm_sq)

    fn compute_geodesic_distance(self, point1: Tensor[DType.float32], point2: Tensor[DType.float32]) -> Float32:
        """Compute geodesic distance on Lorentzian manifold"""
        var dot_product = self.compute_lorentzian_dot_product(point1, point2)
        var norm1 = self.compute_norm(point1)
        var norm2 = self.compute_norm(point2)

        if norm1 < self.geodesic_tolerance or norm2 < self.geodesic_tolerance:
            return 0.0

        var curvature_factor = abs(self.manifold_curvature)
        var argument = dot_product / (curvature_factor * norm1 * norm2)

        # Clamp argument to [-1, 1] for acos
        if argument > 1.0:
            argument = 1.0
        elif argument < -1.0:
            argument = -1.0

        return (1.0 / curvature_factor) * acos(argument)

# Ising energy calculator
struct IsingEnergyCalculator:
    var coupling_strength: Float32
    var ising_temperature: Float32

    fn __init__(coupling_strength: Float32 = 0.5, ising_temperature: Float32 = 0.1):
        self.coupling_strength = coupling_strength
        self.ising_temperature = ising_temperature

    fn compute_ising_energy(self, spin1: Float32, spin2: Float32) -> Float32:
        """Compute Ising interaction energy between spins"""
        return -self.coupling_strength * spin1 * spin2

    fn compute_total_ising_energy(self, spin_config1: Tensor[DType.float32], spin_config2: Tensor[DType.float32]) -> Float32:
        """Compute total Ising interaction energy"""
        var total_energy = 0.0

        for i in range(min(spin_config1.shape()[0], spin_config2.shape()[0])):
            total_energy += self.compute_ising_energy(spin_config1[i], spin_config2[i])

        return total_energy

    fn compute_boltzmann_weight(self, energy: Float32) -> Float32:
        """Compute Boltzmann weighting factor"""
        return exp(-energy / self.ising_temperature)

# Expert prototype manager
struct ExpertPrototypeManager:
    var expert_prototypes: Tensor[DType.float32]
    var expert_spin_configs: Tensor[DType.float32]
    var num_experts: Int
    var hidden_dim: Int

    fn __init__(num_experts: Int, hidden_dim: Int):
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.expert_prototypes = self.initialize_expert_prototypes()
        self.expert_spin_configs = self.initialize_expert_spin_configs()

    fn initialize_expert_prototypes(self) -> Tensor[DType.float32]:
        """Initialize expert prototype embeddings on manifold"""
        var prototypes = Tensor[DType.float32](self.num_experts, self.hidden_dim)

        for expert_idx in range(self.num_experts):
            for dim_idx in range(self.hidden_dim):
                # Create distinct manifold positions for each expert
                if expert_idx == 0:  # Linguistic expert
                    prototypes[expert_idx, dim_idx] = sin(Float32(dim_idx) * 0.1) * 0.3
                elif expert_idx == 1:  # Physics expert
                    prototypes[expert_idx, dim_idx] = cos(Float32(dim_idx) * 0.1) * 0.3
                else:  # Diffusion expert
                    prototypes[expert_idx, dim_idx] = sin(Float32(dim_idx) * 0.15 + 1.57) * 0.3

        return prototypes

    fn initialize_expert_spin_configs(self) -> Tensor[DType.float32]:
        """Initialize expert spin configurations"""
        var spin_configs = Tensor[DType.float32](self.num_experts, self.hidden_dim)

        for expert_idx in range(self.num_experts):
            for dim_idx in range(self.hidden_dim):
                # Create distinct spin configurations
                if expert_idx == 0:  # Linguistic expert
                    spin_configs[expert_idx, dim_idx] = sin(Float32(dim_idx) * 0.2)
                elif expert_idx == 1:  # Physics expert
                    spin_configs[expert_idx, dim_idx] = cos(Float32(dim_idx) * 0.2)
                else:  # Diffusion expert
                    spin_configs[expert_idx, dim_idx] = sin(Float32(dim_idx) * 0.25 + 0.78)

        return spin_configs

    fn get_expert_prototype(self, expert_id: Int) -> Tensor[DType.float32]:
        """Get expert prototype embedding"""
        var prototype = Tensor[DType.float32](self.hidden_dim)

        for i in range(self.hidden_dim):
            prototype[i] = self.expert_prototypes[expert_id, i]

        return prototype

    fn get_expert_spin_config(self, expert_id: Int) -> Tensor[DType.float32]:
        """Get expert spin configuration"""
        var spin_config = Tensor[DType.float32](self.hidden_dim)

        for i in range(self.hidden_dim):
            spin_config[i] = self.expert_spin_configs[expert_id, i]

        return spin_config

    fn apply_top_k_mask(self, scores: Tensor[DType.float32], top_k: Int) -> Tensor[DType.float32]:
        """Apply top-k masking to scores"""
        var shape = scores.shape()
        var masked = Tensor[DType.float32](shape)

        for b in range(shape[0]):
            for s in range(shape[1]):
                # Find top-k indices
                var top_indices = Tensor[Int](top_k)
                var top_values = Tensor[Float32](top_k)

                # Initialize with first k values
                for i in range(top_k):
                    top_indices[i] = i
                    top_values[i] = scores[b, s, i]

                # Find top-k
                for i in range(top_k, shape[2]):
                    for j in range(top_k):
                        if scores[b, s, i] > top_values[j]:
                            top_values[j] = scores[b, s, i]
                            top_indices[j] = i
                            break

                # Apply mask
                for i in range(shape[2]):
                    var is_top_k = False
                    for j in range(top_k):
                        if top_indices[j] == i:
                            is_top_k = true
                            break

                    masked[b, s, i] = is_top_k ? scores[b, s, i] : -1e9  # Large negative value

        return masked

# Manifold-Aware Gating Network Service
struct GatingNetworkService:
    var config: GatingConfig
    var geodesic_calculator: GeodesicCalculator
    var ising_calculator: IsingEnergyCalculator
    var prototype_manager: ExpertPrototypeManager

    fn __init__(config: SystemConfig):
        self.config = GatingConfig(config.num_experts, config.hidden_dim)
        self.geodesic_calculator = GeodesicCalculator(config.manifold_curvature)
        self.ising_calculator = IsingEnergyCalculator(0.5, 0.1)
        self.prototype_manager = ExpertPrototypeManager(config.num_experts, config.hidden_dim)

        print("� Manifold-Aware Gating Network Service Initialized")
        print("   - Number of Experts: {}".format(self.config.num_experts))
        print("   - Hidden Dimension: {}".format(self.config.hidden_dim))
        print("   - Manifold Curvature: {}".format(config.manifold_curvature))
        print("   - Ising Coupling: 0.5")
        print("   - Routing Method: Geodesic Distance + Ising Energy")

    fn compute_gating_scores(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Compute manifold-aware gating scores (actually distances)"""
        var shape = input.shape()
        var batch_size = shape[0]
        var seq_len = shape[1]

        # Return distances (lower is better, opposite of traditional scores)
        var distances = Tensor[DType.float32](batch_size, seq_len, self.config.num_experts)

        for b in range(batch_size):
            for s in range(seq_len):
                # Extract token embedding and spin config
                var token_embedding = Tensor[DType.float32](self.config.hidden_dim)
                var token_spin_config = Tensor[DType.float32](self.config.hidden_dim)

                for i in range(self.config.hidden_dim):
                    token_embedding[i] = input[b, s, i]
                    # Generate spin config from embedding (simplified)
                    token_spin_config[i] = tanh(input[b, s, i])

                # Compute manifold-aware distances to all experts
                for expert_idx in range(self.config.num_experts):
                    var expert_prototype = self.prototype_manager.get_expert_prototype(expert_idx)
                    var expert_spin_config = self.prototype_manager.get_expert_spin_config(expert_idx)

                    distances[b, s, expert_idx] = self.compute_ising_weighted_distance(
                        token_embedding, expert_prototype, token_spin_config, expert_spin_config
                    )

        return distances

    fn compute_ising_weighted_distance(self, token_embedding: Tensor[DType.float32],
                                      expert_prototype: Tensor[DType.float32],
                                      token_spin_config: Tensor[DType.float32],
                                      expert_spin_config: Tensor[DType.float32]) -> Float32:
        """Compute Ising-weighted geodesic distance"""
        # Base geodesic distance
        var geodesic_dist = self.geodesic_calculator.compute_geodesic_distance(token_embedding, expert_prototype)

        # Ising interaction energy
        var ising_energy = self.ising_calculator.compute_total_ising_energy(token_spin_config, expert_spin_config)

        # Boltzmann weighting
        var boltzmann_factor = self.ising_calculator.compute_boltzmann_weight(ising_energy)

        # Combine geodesic distance with Ising weighting
        return geodesic_dist * boltzmann_factor

    fn apply_softmax(self, distances: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Convert distances to probabilities using inverse softmax"""
        var shape = distances.shape()
        var probabilities = Tensor[DType.float32](shape)

        for b in range(shape[0]):
            for s in range(shape[1]):
                # Convert distances to similarities (inverse relationship)
                var similarities = Tensor[DType.float32](shape[2])
                var max_distance = distances[b, s, 0]

                for i in range(shape[2]):
                    if distances[b, s, i] > max_distance:
                        max_distance = distances[b, s, i]

                var sum_exp = 0.0
                for i in range(shape[2]):
                    # Inverse distance to similarity
                    var similarity = exp(-(distances[b, s, i] - max_distance))
                    similarities[i] = similarity
                    sum_exp += similarity

                # Normalize to probabilities
                if sum_exp > 0.0:
                    for i in range(shape[2]):
                        probabilities[b, s, i] = similarities[i] / sum_exp
                else:
                    # Equal probabilities if all distances are equal
                    for i in range(shape[2]):
                        probabilities[b, s, i] = 1.0 / Float32(shape[2])

        return probabilities

    fn apply_temperature_scaling(self, distances: Tensor[DType.float32], temperature: Float32) -> Tensor[DType.float32]:
        """Apply temperature scaling to distances"""
        var shape = distances.shape()
        var scaled = Tensor[DType.float32](shape)

        for b in range(shape[0]):
            for s in range(shape[1]):
                for i in range(shape[2]):
                    scaled[b, s, i] = distances[b, s, i] / temperature

        return scaled

    fn select_closest_expert(self, distances: Tensor[DType.float32]) -> Tensor[Int]:
        """Select expert with minimum distance (closest on manifold)"""
        var shape = distances.shape()
        var expert_ids = Tensor[Int](shape[0], shape[1])

        for b in range(shape[0]):
            for s in range(shape[1]):
                var min_distance = distances[b, s, 0]
                var best_expert = 0

                for expert_idx in range(1, shape[2]):
                    if distances[b, s, expert_idx] < min_distance:
                        min_distance = distances[b, s, expert_idx]
                        best_expert = expert_idx

                expert_ids[b, s] = best_expert

        return expert_ids

    fn get_gating_config(self) -> GatingConfig:
        """Get gating configuration"""
        return self.config

    fn get_num_experts(self) -> Int:
        """Get number of experts"""
        return self.config.num_experts

    def get_service_info(self) -> String:
        """Get service information"""
        var info = "🌌 Manifold-Aware Gating Network Service Information\n"
        info += "=" * 50 + "\n"
        info += "Number of Experts: {}\n".format(self.config.num_experts)
        info += "Hidden Dimension: {}\n".format(self.config.hidden_dim)
        info += "Manifold Curvature: {}\n".format(self.geodesic_calculator.manifold_curvature)
        info += "Ising Coupling: {:.3f}\n".format(self.ising_calculator.coupling_strength)
        info += "Ising Temperature: {:.3f}\n".format(self.ising_calculator.ising_temperature)
        info += "Geodesic Tolerance: {:.6f}\n".format(self.geodesic_calculator.geodesic_tolerance)
        info += "Routing Method: Geodesic Distance + Ising Energy\n"
        info += "Selection: Minimum distance (closest on manifold)\n"

        return info

# Factory function
fn create_gating_network_service(config: SystemConfig) -> GatingNetworkService:
    """Create gating network service"""
    return GatingNetworkService(config)
