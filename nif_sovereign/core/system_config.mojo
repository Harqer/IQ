# System Configuration
# Central configuration management with dependency injection support

struct SystemConfig:
    # Model Architecture
    var model_name: String
    var base_model: String
    var hidden_dim: Int
    var num_layers: Int
    var num_experts: Int

    # Hardware Configuration
    var gpu_target: String
    var memory_gb: Int
    var compute_capability: String

    # Optimization Settings
    var use_muon: Bool
    var use_galore: Bool
    var adapter_rank: Int

    # Data Configuration
    var batch_size: Int
    var sequence_length: Int

    # Physics Parameters
    var manifold_curvature: Float64
    var oscillation_depth: Int
    var ising_iterations: Int

    fn __init__(inout self):
        self.model_name = "NIF-Sovereign"
        self.base_model = "Gemma-4-26B-A4B"
        self.hidden_dim = 4096
        self.num_layers = 32
        self.num_experts = 3

        self.gpu_target = "nvidia-h200-remote"
        self.memory_gb = 192
        self.compute_capability = "8.9"

        self.use_muon = True
        self.use_galore = True
        self.adapter_rank = 64

        self.batch_size = 32
        self.sequence_length = 8192

        self.manifold_curvature = 0.1
        self.oscillation_depth = 5
        self.ising_iterations = 10
