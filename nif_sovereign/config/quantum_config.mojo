# Quantum Hardware Configuration for IQ LLM Fine-Tuning
# Supports multiple quantum backends: CUDA-Q, IBM Quantum, IonQ, Rigetti

from std.collections import Dict
from std.os import getenv

@fieldwise_init
struct QuantumBackendConfig(Copyable, Movable):
    var name: String
    var enabled: Bool
    var api_key: String
    var endpoint: String
    var device: String  # Specific device/qubit count
    var max_qubits: Int
    var max_circuit_depth: Int
    var error_mitigation: Bool
    
    def __init__(out self, name: String):
        self.name = name
        self.enabled = False
        self.api_key = ""
        self.endpoint = ""
        self.device = ""
        self.max_qubits = 0
        self.max_circuit_depth = 0
        self.error_mitigation = True

@fieldwise_init
struct QuantumHardwareConfig(Copyable, Movable):
    var backends: Dict[String, QuantumBackendConfig]
    var default_backend: String
    var fallback_to_classical: Bool
    var max_parallel_jobs: Int
    var timeout_seconds: Int
    
    def __init__(out self):
        self.backends = Dict[String, QuantumBackendConfig]()
        self.default_backend = "cuda_q"
        self.fallback_to_classical = True
        self.max_parallel_jobs = 4
        self.timeout_seconds = 300
        
        # Initialize backends
        self._init_cuda_q()
        self._init_ibm_quantum()
        self._init_ionq()
        self._init_rigetti()
    
    def _init_cuda_q(inout self):
        var config = QuantumBackendConfig(name="cuda_q")
        config.enabled = True
        config.api_key = getenv("CUDA_Q_API_KEY", "")
        config.endpoint = getenv("CUDA_Q_ENDPOINT", "http://localhost:8000")
        config.device = getenv("CUDA_Q_DEVICE", "nvidia-h200")
        config.max_qubits = 1000  # Simulated qubits
        config.max_circuit_depth = 10000
        config.error_mitigation = True
        self.backends["cuda_q"] = config
    
    def _init_ibm_quantum(inout self):
        var config = QuantumBackendConfig(name="ibm_quantum")
        config.api_key = getenv("IBM_QUANTUM_API_KEY", "")
        config.endpoint = "https://quantum-computing.ibm.com"
        config.device = getenv("IBM_QUANTUM_DEVICE", "ibm_brisbane")
        config.max_qubits = 127  # IBM Heron
        config.max_circuit_depth = 1000
        config.error_mitigation = True
        
        # Enable if API key is provided
        config.enabled = len(config.api_key) > 0
        self.backends["ibm_quantum"] = config
    
    def _init_ionq(inout self):
        var config = QuantumBackendConfig(name="ionq")
        config.api_key = getenv("IONQ_API_KEY", "")
        config.endpoint = "https://api.ionq.co/v0.2"
        config.device = getenv("IONQ_DEVICE", "ionq_qpu")
        config.max_qubits = 36
        config.max_circuit_depth = 500
        config.error_mitigation = True
        
        # Enable if API key is provided
        config.enabled = len(config.api_key) > 0
        self.backends["ionq"] = config
    
    def _init_rigetti(inout self):
        var config = QuantumBackendConfig(name="rigetti")
        config.api_key = getenv("RIGETTI_API_KEY", "")
        config.endpoint = "https://api.rigetti.com"
        config.device = getenv("RIGETTI_DEVICE", "aspen-m-3")
        config.max_qubits = 80
        config.max_circuit_depth = 300
        config.error_mitigation = True
        
        # Enable if API key is provided
        config.enabled = len(config.api_key) > 0
        self.backends["rigetti"] = config
    
    def get_backend(self, name: String) -> QuantumBackendConfig:
        if name in self.backends:
            return self.backends[name]
        return self.backends[self.default_backend]
    
    def get_enabled_backends(self) -> List[String]:
        var enabled = List[String]()
        for name in self.backends.keys():
            if self.backends[name].enabled:
                enabled.append(name)
        return enabled
    
    def set_api_key(self, backend_name: String, api_key: String):
        if backend_name in self.backends:
            self.backends[backend_name].api_key = api_key
            self.backends[backend_name].enabled = len(api_key) > 0
    
    def validate_config(self) -> Bool:
        var has_enabled = False
        for name in self.backends.keys():
            if self.backends[name].enabled:
                has_enabled = True
                # Validate API key if required
                if name != "cuda_q" and len(self.backends[name].api_key) == 0:
                    print(f"Warning: Backend {name} is enabled but has no API key")
                    return False
        return has_enabled or self.fallback_to_classical

# ============================================================================
# Environment Setup Helper
# ============================================================================

def setup_quantum_environment(config_file: String = ".env.quantum") raises:
    """Setup quantum hardware configuration from environment variables or file"""
    print("Setting up quantum hardware configuration...")
    
    var config = QuantumHardwareConfig()
    
    # Try to load from file
    var python = Python.import_module("os")
    var pathlib = Python.import_module("pathlib")
    
    if pathlib.Path(config_file).exists():
        print(f"Loading quantum config from {config_file}")
        var dotenv = Python.import_module("dotenv")
        dotenv.load_dotenv(config_file)
    
    # Validate configuration
    if config.validate_config():
        print("Quantum configuration validated successfully")
        print("Enabled backends: ", config.get_enabled_backends())
        print("Default backend: ", config.default_backend)
    else:
        print("Warning: No valid quantum backends configured")
        print("Falling back to classical simulation")
    
    return config

# ============================================================================
# Quantum Backend Selection Strategy
# ============================================================================

@fieldwise_init
struct BackendSelector(Copyable, Movable):
    var config: QuantumHardwareConfig
    var task_requirements: Dict[String, Int]  # task -> required qubits
    
    def __init__(out self, config: QuantumHardwareConfig):
        self.config = config
        self.task_requirements = Dict[String, Int]()
        self._init_task_requirements()
    
    def _init_task_requirements(inout self):
        # Define qubit requirements for different tasks
        self.task_requirements["adapter_fusion"] = 10
        self.task_requirements["weight_optimization"] = 50
        self.task_requirements["quantum_routing"] = 20
        self.task_requirements["ising_simulation"] = 30
        self.task_requirements["neutrino_oscillation"] = 15
    
    def select_backend(self, task: String) -> String:
        var required_qubits = 0
        if task in self.task_requirements:
            required_qubits = self.task_requirements[task]
        
        # Find backend that can handle the task
        for backend_name in self.config.get_enabled_backends():
            var backend = self.config.get_backend(backend_name)
            if backend.max_qubits >= required_qubits:
                return backend_name
        
        # Fallback to default
        return self.config.default_backend
    
    def select_backend_for_circuit(self, num_qubits: Int, circuit_depth: Int) -> String:
        for backend_name in self.config.get_enabled_backends():
            var backend = self.config.get_backend(backend_name)
            if backend.max_qubits >= num_qubits and backend.max_circuit_depth >= circuit_depth:
                return backend_name
        
        return self.config.default_backend
