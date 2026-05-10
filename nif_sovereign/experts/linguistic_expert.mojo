# Linguistic Expert
# Atomic component for linguistic processing in MoE systems
# Single responsibility: linguistic expert processing

from tensor import Tensor
from math import tanh, exp, sqrt
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.interfaces.expert_interface import ExpertInterface, ExpertConfig

# Linguistic attention processor
struct LinguisticAttention:
    var attention_weights: Tensor[DType.float32]
    var hidden_dim: Int
    var num_heads: Int
    
    fn __init__(hidden_dim: Int, num_heads: Int = 8):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention_weights = self.initialize_attention_weights()
    
    fn initialize_attention_weights(self) -> Tensor[DType.float32]:
        """Initialize attention weights"""
        var weights = Tensor[DType.float32](self.hidden_dim, self.hidden_dim)
        
        for i in range(self.hidden_dim):
            for j in range(self.hidden_dim):
                # Simplified attention weight initialization
                var distance = abs(Float32(i - j))
                weights[i, j] = exp(-distance * 0.1)
        
        return weights
    
    fn compute_attention(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Compute linguistic attention"""
        var attention_output = Tensor[DType.float32](self.hidden_dim)
        
        for i in range(self.hidden_dim):
            var attention_score = 0.0
            
            # Self-attention simulation
            for j in range(self.hidden_dim):
                attention_score += input[j] * self.attention_weights[i, j]
            
            attention_output[i] = attention_score
        
        return attention_output

# Linguistic feedforward processor
struct LinguisticFeedforward:
    var feedforward_weights: Tensor[DType.float32]
    var hidden_dim: Int
    var intermediate_dim: Int
    
    fn __init__(hidden_dim: Int, intermediate_dim: Int = 2048):
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.feedforward_weights = self.initialize_feedforward_weights()
    
    fn initialize_feedforward_weights(self) -> Tensor[DType.float32]:
        """Initialize feedforward weights"""
        var weights = Tensor[DType.float32](self.hidden_dim, self.intermediate_dim)
        
        for i in range(self.hidden_dim):
            for j in range(self.intermediate_dim):
                # Xavier initialization
                var scale = sqrt(2.0 / Float32(self.hidden_dim + self.intermediate_dim))
                weights[i, j] = (Float32(i * j % 1000) / 1000.0 - 0.5) * 2.0 * scale
        
        return weights
    
    fn compute_feedforward(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Compute linguistic feedforward"""
        var intermediate = Tensor[DType.float32](self.intermediate_dim)
        var output = Tensor[DType.float32](self.hidden_dim)
        
        # Project to intermediate dimension
        for i in range(self.intermediate_dim):
            var sum = 0.0
            for j in range(self.hidden_dim):
                sum += input[j] * self.feedforward_weights[j, i]
            intermediate[i] = tanh(sum)
        
        # Project back to hidden dimension
        for i in range(self.hidden_dim):
            var sum = 0.0
            for j in range(self.intermediate_dim):
                sum += intermediate[j] * self.feedforward_weights[i, j]
            output[i] = tanh(sum)
        
        return output

# Linguistic Expert Implementation
struct LinguisticExpert:
    var config: ExpertConfig
    var attention: LinguisticAttention
    var feedforward: LinguisticFeedforward
    var expert_id: Int
    var expert_type: String
    var capacity: Int
    
    fn __init__(config: SystemConfig):
        self.config = ExpertConfig("linguistic", 0, config.hidden_dim // config.num_experts, config.hidden_dim, "standard_transformer")
        self.attention = LinguisticAttention(config.hidden_dim, 8)
        self.feedforward = LinguisticFeedforward(config.hidden_dim, config.hidden_dim * 4)
        self.expert_id = 0
        self.expert_type = "linguistic"
        self.capacity = config.hidden_dim // config.num_experts
        
        print("📝 Linguistic Expert Initialized")
        print("   - Expert ID: {}".format(self.expert_id))
        print("   - Expert Type: {}".format(self.expert_type))
        print("   - Capacity: {}".format(self.capacity))
        print("   - Attention Heads: 8")
        print("   - Intermediate Dim: {}".format(config.hidden_dim * 4))
    
    fn process(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Process input through linguistic expert"""
        # Step 1: Apply linguistic attention
        var attention_output = self.attention.compute_attention(input)
        
        # Step 2: Apply feedforward network
        var feedforward_output = self.feedforward.compute_feedforward(attention_output)
        
        # Step 3: Residual connection
        var output = Tensor[DType.float32](self.config.hidden_dim)
        for i in range(self.config.hidden_dim):
            output[i] = input[i] + feedforward_output[i] * 0.1
        
        return output
    
    fn get_expert_type(self) -> String:
        """Get expert type"""
        return self.expert_type
    
    fn get_expert_id(self) -> Int:
        """Get expert ID"""
        return self.expert_id
    
    fn get_expert_capacity(self) -> Int:
        """Get expert capacity"""
        return self.capacity
    
    fn is_available(self) -> Bool:
        """Check if expert is available"""
        return True  # Simplified - would check current load
    
    fn get_current_load(self) -> Float32:
        """Get current load"""
        return 0.0  # Simplified - would return actual load
    
    fn update_load(self, load: Float32):
        """Update expert load"""
        # Simplified - would update actual load tracking
    
    def get_expert_info(self) -> String:
        """Get expert information"""
        var info = "📝 Linguistic Expert Information\n"
        info += "=" * 30 + "\n"
        info += "Expert ID: {}\n".format(self.expert_id)
        info += "Expert Type: {}\n".format(self.expert_type)
        info += "Capacity: {}\n".format(self.capacity)
        info += "Attention Heads: 8\n"
        info += "Specialization: Standard Transformer\n"
        info += "Processing: Self-attention + Feedforward\n"
        
        return info

# Factory function
fn create_linguistic_expert(config: SystemConfig) -> LinguisticExpert:
    """Create linguistic expert"""
    return LinguisticExpert(config)
