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

    # API Keys
    var tinker_api_key: String
    var thunder_api_key: String

    fn __init__(
        out self,
        model_name: String = "NIF-Sovereign",
        base_model: String = "Gemma-4-26B-A4B",
        hidden_dim: Int = 4096,
        num_layers: Int = 32,
        num_experts: Int = 3,
        gpu_target: String = "nvidia-h200-remote",
        memory_gb: Int = 192,
        compute_capability: String = "8.9",
        use_muon: Bool = True,
        use_galore: Bool = True,
        adapter_rank: Int = 64,
        batch_size: Int = 32,
        sequence_length: Int = 8192,
        manifold_curvature: Float64 = 0.1,
        oscillation_depth: Int = 5,
        ising_iterations: Int = 10,
        tinker_api_key: String = "tml-UfsmNI2GJD8M0Si36Xcpk9iWQtHY8IXgv7eFScev6TT6Zk9fcc5vnlkLQXtnz1CNHAAAA",
        thunder_api_key: String = "THUNDER_API_KEY_PLACEHOLDER"
    ):
        self.model_name = model_name
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.gpu_target = gpu_target
        self.memory_gb = memory_gb
        self.compute_capability = compute_capability
        self.use_muon = use_muon
        self.use_galore = use_galore
        self.adapter_rank = adapter_rank
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.manifold_curvature = manifold_curvature
        self.oscillation_depth = oscillation_depth
        self.ising_iterations = ising_iterations
        self.tinker_api_key = tinker_api_key
        self.thunder_api_key = thunder_api_key

    fn __copyinit__(out self, copy: Self):
        self.model_name = copy.model_name
        self.base_model = copy.base_model
        self.hidden_dim = copy.hidden_dim
        self.num_layers = copy.num_layers
        self.num_experts = copy.num_experts
        self.gpu_target = copy.gpu_target
        self.memory_gb = copy.memory_gb
        self.compute_capability = copy.compute_capability
        self.use_muon = copy.use_muon
        self.use_galore = copy.use_galore
        self.adapter_rank = copy.adapter_rank
        self.batch_size = copy.batch_size
        self.sequence_length = copy.sequence_length
        self.manifold_curvature = copy.manifold_curvature
        self.oscillation_depth = copy.oscillation_depth
        self.ising_iterations = copy.ising_iterations
        self.tinker_api_key = copy.tinker_api_key
        self.thunder_api_key = copy.thunder_api_key

    fn __moveinit__(out self, owned owned_val: Self):
        self.model_name = owned_val.model_name^
        self.base_model = owned_val.base_model^
        self.hidden_dim = owned_val.hidden_dim
        self.num_layers = owned_val.num_layers
        self.num_experts = owned_val.num_experts
        self.gpu_target = owned_val.gpu_target^
        self.memory_gb = owned_val.memory_gb
        self.compute_capability = owned_val.compute_capability^
        self.use_muon = owned_val.use_muon
        self.use_galore = owned_val.use_galore
        self.adapter_rank = owned_val.adapter_rank
        self.batch_size = owned_val.batch_size
        self.sequence_length = owned_val.sequence_length
        self.manifold_curvature = owned_val.manifold_curvature
        self.oscillation_depth = owned_val.oscillation_depth
        self.ising_iterations = owned_val.ising_iterations
        self.tinker_api_key = owned_val.tinker_api_key^
        self.thunder_api_key = owned_val.thunder_api_key^
