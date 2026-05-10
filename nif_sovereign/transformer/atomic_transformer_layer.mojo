# Atomic Transformer Layer
# Composable transformer layer with single responsibility
# Combines attention and feedforward services

from tensor import Tensor
from math import sqrt, exp, tanh, sin, cos, abs
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.interfaces.attention_interface import AttentionInterface
from nif_sovereign.interfaces.embedding_interface import EmbeddingInterface
from nif_sovereign.attention.physics_attention_service import PhysicsAttentionService
from nif_sovereign.feedforward.physics_feedforward_service import PhysicsFeedforwardService

# Residual connection processor
struct ResidualConnection:
    fn apply_residual(self, input: Tensor[DType.float32], processed: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply residual connection"""
        var shape = input.shape()
        var output = Tensor[DType.float32](shape)
        
        for b in range(shape[0]):
            for s in range(shape[1]):
                for i in range(shape[2]):
                    output[b, s, i] = input[b, s, i] + processed[b, s, i]
        
        return output

# Layer normalization wrapper
struct LayerNormalization:
    var feedforward_service: PhysicsFeedforwardService
    
    fn __init__(feedforward_service: PhysicsFeedforwardService):
        self.feedforward_service = feedforward_service
    
    fn normalize(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply layer normalization"""
        return self.feedforward_service.normalize(input)

# MoE integration processor
struct MoEIntegration:
    var use_moe: Bool
    var layer_index: Int
    
    fn __init__(layer_index: Int, use_moe_frequency: Int = 4):
        self.layer_index = layer_index
        self.use_moe = (layer_index % use_moe_frequency == 0)
    
    fn should_use_moe(self) -> Bool:
        """Check if MoE should be used for this layer"""
        return self.use_moe

# Atomic Transformer Layer
struct AtomicTransformerLayer:
    var layer_index: Int
    var attention_service: AttentionInterface
    var feedforward_service: PhysicsFeedforwardService
    var residual_connection: ResidualConnection
    var layer_norm: LayerNormalization
    var moe_integration: MoEIntegration
    
    fn __init__(layer_index: Int, attention_service: AttentionInterface, feedforward_service: PhysicsFeedforwardService):
        self.layer_index = layer_index
        self.attention_service = attention_service
        self.feedforward_service = feedforward_service
        self.residual_connection = ResidualConnection()
        self.layer_norm = LayerNormalization(feedforward_service)
        self.moe_integration = MoEIntegration(layer_index)
        
        print("🔬 Atomic Transformer Layer {} Initialized".format(layer_index))
        print("   - Attention Service: Active")
        print("   - Feedforward Service: Active")
        print("   - MoE Integration: {}".format(self.moe_integration.should_use_moe() ? "Active" : "Inactive"))
    
    fn forward(self, input: Tensor[DType.float32], attention_mask: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass through atomic transformer layer"""
        
        # Step 1: Physics Multi-Head Attention
        var attention_output = self.attention_service.compute_attention(input, attention_mask)
        
        # Step 2: Add & Norm (attention)
        var attention_residual = self.residual_connection.apply_residual(input, attention_output)
        var attention_normalized = self.layer_norm.normalize(attention_residual)
        
        # Step 3: MoE Routing (if enabled)
        var moe_output = attention_normalized
        if self.moe_integration.should_use_moe():
            # MoE processing would be injected here
            # For now, pass through (MoE service would be injected)
            pass
        
        # Step 4: Physics Feed-Forward Network
        var feedforward_output = self.feedforward_service.process(moe_output)
        
        # Step 5: Add & Norm (feedforward)
        var final_output = self.residual_connection.apply_residual(moe_output, feedforward_output)
        var final_normalized = self.layer_norm.normalize(final_output)
        
        return final_normalized
    
    fn get_layer_info(self) -> String:
        """Get layer information"""
        var info = "🔬 Atomic Transformer Layer {}\n".format(self.layer_index)
        info += "   - Layer Index: {}\n".format(self.layer_index)
        info += "   - Attention Service: Active\n"
        info += "   - Feedforward Service: Active\n"
        info += "   - MoE Integration: {}\n".format(self.moe_integration.should_use_moe() ? "Active" : "Inactive")
        info += "   - Residual Connections: Active\n"
        info += "   - Layer Normalization: Active\n"
        
        return info

# Transformer layer factory
struct TransformerLayerFactory:
    fn create_layer(layer_index: Int, attention_service: AttentionInterface, feedforward_service: PhysicsFeedforwardService) -> AtomicTransformerLayer:
        """Create atomic transformer layer"""
        return AtomicTransformerLayer(layer_index, attention_service, feedforward_service)

# Layer collection manager
struct LayerCollection:
    var layers: Tensor[AtomicTransformerLayer]
    var num_layers: Int
    
    fn __init__(num_layers: Int, attention_service: AttentionInterface, feedforward_service: PhysicsFeedforwardService):
        self.num_layers = num_layers
        self.layers = Tensor[AtomicTransformerLayer](num_layers)
        
        var factory = TransformerLayerFactory()
        
        for i in range(num_layers):
            self.layers[i] = factory.create_layer(i, attention_service, feedforward_service)
        
        print("📚 Layer Collection Initialized")
        print("   - Number of Layers: {}".format(num_layers))
        print("   - All Layers: Atomic Design")
    
    fn forward_all(self, input: Tensor[DType.float32], attention_mask: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass through all layers"""
        var current_output = input
        
        for i in range(self.num_layers):
            current_output = self.layers[i].forward(current_output, attention_mask)
        
        return current_output
    
    fn get_collection_info(self) -> String:
        """Get collection information"""
        var info = "📚 Layer Collection Information\n"
        info += "=" * 30 + "\n"
        info += "Number of Layers: {}\n".format(self.num_layers)
        info += "Design Pattern: Atomic\n"
        info += "All Layers: Single Responsibility\n"
        
        return info
