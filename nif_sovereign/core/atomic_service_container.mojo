# Atomic Service Container
# Dependency injection container for atomic design principles
# Manages component creation and lifecycle

from tensor import Tensor
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.interfaces.attention_interface import AttentionInterface, AttentionConfig
from nif_sovereign.interfaces.embedding_interface import EmbeddingInterface, EmbeddingConfig
from nif_sovereign.interfaces.optimization_interface import OptimizationInterface, OptimizationConfig
from nif_sovereign.interfaces.routing_interface import RoutingInterface, RoutingConfig

# Service factory for creating components
struct ServiceFactory:
    var config: SystemConfig
    
    fn __init__(config: SystemConfig):
        self.config = config
    
    fn create_attention_service(self) -> AttentionInterface:
        """Create attention service based on configuration"""
        # Factory method - could return different implementations
        return PhysicsAttentionService(self.config)
    
    fn create_embedding_service(self) -> EmbeddingInterface:
        """Create embedding service based on configuration"""
        return LorentzianEmbeddingService(self.config)
    
    fn create_optimization_service(self) -> OptimizationInterface:
        """Create optimization service based on configuration"""
        return DistributedMuonService(self.config)
    
    fn create_routing_service(self) -> RoutingInterface:
        """Create routing service based on configuration"""
        return RiemannianIsingRoutingService(self.config)

# Atomic service container
struct AtomicServiceContainer:
    var config: SystemConfig
    var factory: ServiceFactory
    var attention_service: AttentionInterface
    var embedding_service: EmbeddingInterface
    var optimization_service: OptimizationInterface
    var routing_service: RoutingInterface
    
    fn __init__(out self, config: SystemConfig):
        self.config = config
        self.factory = ServiceFactory(config)
        
        # Dependency injection - all services created through factory
        self.attention_service = self.factory.create_attention_service()
        self.embedding_service = self.factory.create_embedding_service()
        self.optimization_service = self.factory.create_optimization_service()
        self.routing_service = self.factory.create_routing_service()
        
        print("🔧 Atomic Service Container Initialized")
        print("   - Attention Service: Injected")
        print("   - Embedding Service: Injected")
        print("   - Optimization Service: Injected")
        print("   - Routing Service: Injected")
    
    fn get_attention_service(self) -> AttentionInterface:
        """Get attention service"""
        return self.attention_service
    
    fn get_embedding_service(self) -> EmbeddingInterface:
        """Get embedding service"""
        return self.embedding_service
    
    fn get_optimization_service(self) -> OptimizationInterface:
        """Get optimization service"""
        return self.optimization_service
    
    fn get_routing_service(self) -> RoutingInterface:
        """Get routing service"""
        return self.routing_service
    
    fn get_config(self) -> SystemConfig:
        """Get system configuration"""
        return self.config

# Service registry for managing multiple instances
struct ServiceRegistry:
    var services: Tensor[ServiceFactory]
    var default_config: SystemConfig
    
    fn __init__(out self, config: SystemConfig):
        self.default_config = config
        self.services = Tensor[ServiceFactory](10)  # Support up to 10 different factories
        
        print("📋 Service Registry Initialized")
        print("   - Default Config: Set")
        print("   - Service Capacity: 10")
    
    fn register_service(mut self, factory: ServiceFactory):
        """Register a new service factory"""
        # Simple registration - in production would use proper registry
        print("📝 Service Factory Registered")
    
    fn create_container(self, config: SystemConfig) -> AtomicServiceContainer:
        """Create service container with specific configuration"""
        return AtomicServiceContainer(config)

# Factory function
fn create_atomic_service_container(config: SystemConfig) -> AtomicServiceContainer:
    """Create atomic service container"""
    return AtomicServiceContainer(config)
