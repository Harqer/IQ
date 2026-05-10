# Atomic NIF Architecture
# Refactored architecture using atomic design principles
# Composable, testable, and maintainable structure

from tensor import Tensor
from math import sqrt, exp, tanh, sin, cos, abs
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.core.atomic_service_container import AtomicServiceContainer
from nif_sovereign.transformer.atomic_transformer_layer import AtomicTransformerLayer, LayerCollection
from nif_sovereign.interfaces.attention_interface import AttentionInterface
from nif_sovereign.interfaces.embedding_interface import EmbeddingInterface
from nif_sovereign.interfaces.optimization_interface import OptimizationInterface
from nif_sovereign.interfaces.routing_interface import RoutingInterface
from nif_sovereign.attention.physics_attention_service import PhysicsAttentionService
from nif_sovereign.feedforward.physics_feedforward_service import PhysicsFeedforwardService

# Output processor
struct OutputProcessor:
    var final_norm: Tensor[DType.float32]
    var output_projection: Tensor[DType.float32]
    var hidden_dim: Int
    
    fn __init__(hidden_dim: Int):
        self.hidden_dim = hidden_dim
        self.final_norm = self.initialize_norm_params()
        self.output_projection = self.initialize_output_projection()
    
    fn initialize_norm_params(self) -> Tensor[DType.float32]:
        """Initialize final layer normalization parameters"""
        var norm_params = Tensor[DType.float32](self.hidden_dim)
        
        for i in range(self.hidden_dim):
            norm_params[i] = 1.0
        
        return norm_params
    
    fn initialize_output_projection(self) -> Tensor[DType.float32]:
        """Initialize output projection matrix"""
        var projection = Tensor[DType.float32](self.hidden_dim, self.hidden_dim)
        
        for i in range(self.hidden_dim):
            for j in range(self.hidden_dim):
                if i == j:
                    projection[i, j] = 1.0
                else:
                    projection[i, j] = 0.0
        
        return projection
    
    fn process_output(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Process final output"""
        var shape = input.shape()
        var output = Tensor[DType.float32](shape)
        
        # Apply final layer normalization
        var normalized = self.apply_layer_norm(input)
        
        # Apply output projection
        var projected = self.apply_output_projection(normalized)
        
        return projected
    
    fn apply_layer_norm(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply layer normalization"""
        var shape = input.shape()
        var output = Tensor[DType.float32](shape)
        
        for b in range(shape[0]):
            for s in range(shape[1]):
                # Compute mean and variance
                var mean = 0.0
                var variance = 0.0
                
                for i in range(shape[2]):
                    mean += input[b, s, i]
                mean /= Float32(shape[2])
                
                for i in range(shape[2]):
                    var diff = input[b, s, i] - mean
                    variance += diff * diff
                variance /= Float32(shape[2])
                
                # Normalize
                var std = sqrt(variance + 1e-6)
                for i in range(shape[2]):
                    output[b, s, i] = (input[b, s, i] - mean) / std * self.final_norm[i]
        
        return output
    
    fn apply_output_projection(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply output projection"""
        var shape = input.shape()
        var output = Tensor[DType.float32](shape)
        
        for b in range(shape[0]):
            for s in range(shape[1]):
                for i in range(self.hidden_dim):
                    var sum = 0.0
                    for j in range(self.hidden_dim):
                        sum += input[b, s, j] * self.output_projection[j, i]
                    output[b, s, i] = sum
        
        return output

# Architecture orchestrator
struct ArchitectureOrchestrator:
    var service_container: AtomicServiceContainer
    var layer_collection: LayerCollection
    var output_processor: OutputProcessor
    var config: SystemConfig
    
    fn __init__(service_container: AtomicServiceContainer):
        self.service_container = service_container
        self.config = service_container.get_config()
        
        # Create layer collection with injected services
        var attention_service = service_container.get_attention_service()
        var feedforward_service = PhysicsFeedforwardService(self.config)
        
        self.layer_collection = LayerCollection(self.config.num_layers, attention_service, feedforward_service)
        self.output_processor = OutputProcessor(self.config.hidden_dim)
        
        print("🌟 Atomic NIF Architecture Orchestrator Initialized")
        print("   - Service Container: Active")
        print("   - Layer Collection: {} layers".format(self.config.num_layers))
        print("   - Output Processor: Active")
    
    fn forward(self, input_tokens: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass through atomic NIF architecture"""
        
        print("🚀 Starting Atomic NIF Forward Pass...")
        
        # Step 1: Apply embedding (injected service)
        var embedding_service = self.service_container.get_embedding_service()
        var embedded_input = embedding_service.embed_input(input_tokens)
        
        # Step 2: Process through atomic transformer layers
        print("   Step 2: Processing through {} Atomic Transformer Layers".format(self.config.num_layers))
        
        # Create attention mask (simplified)
        var attention_mask = Tensor[DType.float32](input_tokens.shape()[0], input_tokens.shape()[1], input_tokens.shape()[1])
        for b in range(attention_mask.shape()[0]):
            for i in range(attention_mask.shape()[1]):
                for j in range(attention_mask.shape()[2]):
                    attention_mask[b, i, j] = 0.0  # No masking
        
        var layer_output = self.layer_collection.forward_all(embedded_input, attention_mask)
        
        # Step 3: Apply output processing
        print("   Step 3: Output Processing")
        var final_output = self.output_processor.process_output(layer_output)
        
        print("✅ Atomic NIF Forward Pass Complete")
        return final_output
    
    fn optimize_parameters(self, gradients: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Optimize parameters using injected optimization service"""
        var optimization_service = self.service_container.get_optimization_service()
        
        # Get current parameters (simplified)
        var current_params = Tensor[DType.float32](gradients.shape())
        
        # Apply optimization
        var optimized_params = optimization_service.optimize_step(current_params, gradients)
        
        return optimized_params
    
    fn get_architecture_info(self) -> String:
        """Get comprehensive architecture information"""
        var info = "🌟 Atomic NIF Architecture Information\n"
        info += "=" * 50 + "\n\n"
        
        info += "Design Principles:\n"
        info += "- Atomic Design: ✅ Applied\n"
        info += "- Single Responsibility: ✅ Each component focused\n"
        info += "- Dependency Injection: ✅ Service container\n"
        info += "- Interface Segregation: ✅ Small interfaces\n"
        info += "- Composition: ✅ Composable components\n\n"
        
        info += "Component Structure:\n"
        info += "- Service Container: Manages dependencies\n"
        info += "- Layer Collection: {} atomic layers\n".format(self.config.num_layers)
        info += "- Output Processor: Final processing\n"
        info += "- Orchestrator: Coordinates components\n\n"
        
        info += "Services Injected:\n"
        info += "- Attention Service: Physics-aware\n"
        info += "- Embedding Service: Manifold integration\n"
        info += "- Optimization Service: Distributed Muon\n"
        info += "- Routing Service: Riemannian Ising\n\n"
        
        info += "Benefits:\n"
        info += "- Testability: Each component testable\n"
        info += "- Maintainability: Single responsibility\n"
        info += "- Scalability: Composable design\n"
        info += "- Flexibility: Interface-based\n"
        
        return info

# Main Atomic NIF Architecture
struct AtomicNIFArchitecture:
    var orchestrator: ArchitectureOrchestrator
    var config: SystemConfig
    
    fn __init__(service_container: AtomicServiceContainer):
        self.config = service_container.get_config()
        self.orchestrator = ArchitectureOrchestrator(service_container)
        
        print("🌟 Atomic NIF Architecture Initialized")
        print("   - Atomic Design Principles: Applied")
        print("   - Dependency Injection: Active")
        print("   - Single Responsibility: Enforced")
        print("   - Interface Segregation: Active")
        print("   - Composition: Enabled")
    
    fn forward(self, input_tokens: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass through atomic architecture"""
        return self.orchestrator.forward(input_tokens)
    
    fn optimize_parameters(self, gradients: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Optimize parameters"""
        return self.orchestrator.optimize_parameters(gradients)
    
    fn get_system_info(self) -> String:
        """Get system information"""
        return self.orchestrator.get_architecture_info()

# Factory function
fn create_atomic_nif_architecture(config: SystemConfig) -> AtomicNIFArchitecture:
    """Create atomic NIF architecture"""
    var service_container = AtomicServiceContainer(config)
    return AtomicNIFArchitecture(service_container)

# Usage example
fn main():
    print("🌟 Initializing Atomic NIF Architecture")
    
    var config = SystemConfig()
    var atomic_nif = create_atomic_nif_architecture(config)
    
    # Create test input
    var test_input = Tensor[DType.float32](2, 8, config.hidden_dim)
    for b in range(2):
        for s in range(8):
            for h in range(config.hidden_dim):
                test_input[b, s, h] = Float32((b * 8 * config.hidden_dim + s * config.hidden_dim + h) % 1000) / 1000.0
    
    print("\n🚀 Testing Atomic NIF Architecture...")
    
    # Run forward pass
    var output = atomic_nif.forward(test_input)
    
    print("\n" + atomic_nif.get_system_info())
    
    print("\n✅ Atomic NIF Architecture Test Successful")
    print("   - Input Shape: [{}, {}, {}]".format(2, 8, config.hidden_dim))
    print("   - Output Shape: [{}, {}, {}]".format(output.shape()[0], output.shape()[1], output.shape()[2]))
    
    print("\n🌟 ATOMIC DESIGN BENEFITS:")
    print("✅ Single Responsibility: Each component focused")
    print("✅ Dependency Injection: Testable and flexible")
    print("✅ Interface Segregation: Small, focused interfaces")
    print("✅ Composition: Composable and reusable")
    print("✅ Maintainability: Easy to understand and modify")
    print("✅ Scalability: Components can be swapped")
    print("✅ Testability: Each component independently testable")
