# Expert Dispatcher
# Atomic component for dispatching tokens to experts in MoE systems
# Single responsibility: expert dispatching and result aggregation

from tensor import Tensor
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.interfaces.expert_interface import ExpertInterface
from nif_sovereign.interfaces.gating_interface import GatingInterface
from nif_sovereign.routing.expert_registry import ExpertRegistry
from nif_sovereign.routing.gating_network import GatingNetworkService
from nif_sovereign.routing.load_balancer import LoadBalancerService

# Dispatch decision
struct DispatchDecision:
    var expert_id: Int
    var expert_type: String
    var confidence: Float32
    var load_penalty: Float32
    
    fn __init__(expert_id: Int, expert_type: String, confidence: Float32, load_penalty: Float32 = 0.0):
        self.expert_id = expert_id
        self.expert_type = expert_type
        self.confidence = confidence
        self.load_penalty = load_penalty

# Expert processor
struct ExpertProcessor:
    var expert_registry: ExpertRegistry
    var expert_capacity: Int
    
    fn __init__(expert_registry: ExpertRegistry, expert_capacity: Int):
        self.expert_registry = expert_registry
        self.expert_capacity = expert_capacity
    
    fn process_through_expert(self, input: Tensor[DType.float32], expert_id: Int) -> Tensor[DType.float32]:
        """Process input through specific expert"""
        var expert = self.expert_registry.get_expert_by_id(expert_id)
        return expert.process(input)
    
    fn process_batch_through_expert(self, batch_input: Tensor[DType.float32], expert_id: Int) -> Tensor[DType.float32]:
        """Process batch through specific expert"""
        var expert = self.expert_registry.get_expert_by_id(expert_id)
        
        var shape = batch_input.shape()
        var output = Tensor[DType.float32](shape)
        
        for b in range(shape[0]):
            for s in range(shape[1]):
                var token_input = Tensor[DType.float32](shape[2])
                for i in range(shape[2]):
                    token_input[i] = batch_input[b, s, i]
                
                var token_output = expert.process(token_input)
                for i in range(shape[2]):
                    output[b, s, i] = token_output[i]
        
        return output
    
    def get_expert_info(self, expert_id: Int) -> String:
        """Get expert information"""
        var expert = self.expert_registry.get_expert_by_id(expert_id)
        return "Expert {}: {}, Capacity: {}".format(
            expert_id, expert.get_expert_type(), self.expert_capacity
        )

# Result aggregator
struct ResultAggregator:
    var aggregation_method: String
    
    fn __init__(aggregation_method: String = "weighted_average"):
        self.aggregation_method = aggregation_method
    
    fn aggregate_expert_outputs(self, expert_outputs: Tensor[Tensor[DType.float32]], 
                                dispatch_decisions: Tensor[DispatchDecision]) -> Tensor[DType.float32]:
        """Aggregate outputs from multiple experts"""
        if expert_outputs.shape()[0] == 0:
            return Tensor[DType.float32](1, 1, 1)  # Fallback
        
        var first_output = expert_outputs[0]
        var shape = first_output.shape()
        var aggregated = Tensor[DType.float32](shape)
        
        for b in range(shape[0]):
            for s in range(shape[1]):
                for i in range(shape[2]):
                    var weighted_sum = 0.0
                    var total_weight = 0.0
                    
                    for expert_idx in range(expert_outputs.shape()[0]):
                        var output = expert_outputs[expert_idx]
                        var decision = dispatch_decisions[expert_idx]
                        var weight = decision.confidence - decision.load_penalty
                        
                        weighted_sum += output[b, s, i] * weight
                        total_weight += weight
                    
                    if total_weight > 0.0:
                        aggregated[b, s, i] = weighted_sum / total_weight
                    else:
                        aggregated[b, s, i] = first_output[b, s, i]
        
        return aggregated
    
    fn combine_expert_outputs(self, expert_outputs: Tensor[Tensor[DType.float32]], 
                              expert_weights: Tensor[Float32]) -> Tensor[DType.float32]:
        """Combine expert outputs with weights"""
        if expert_outputs.shape()[0] == 0:
            return Tensor[DType.float32](1, 1, 1)  # Fallback
        
        var first_output = expert_outputs[0]
        var shape = first_output.shape()
        var combined = Tensor[DType.float32](shape)
        
        for b in range(shape[0]):
            for s in range(shape[1]):
                for i in range(shape[2]):
                    var sum = 0.0
                    
                    for expert_idx in range(expert_outputs.shape()[0]):
                        var output = expert_outputs[expert_idx]
                        var weight = expert_weights[expert_idx]
                        sum += output[b, s, i] * weight
                    
                    combined[b, s, i] = sum
        
        return combined

# Expert Dispatcher Service
struct ExpertDispatcherService:
    var config: SystemConfig
    var expert_registry: ExpertRegistry
    var gating_network: GatingNetworkService
    var load_balancer: LoadBalancerService
    var expert_processor: ExpertProcessor
    var result_aggregator: ResultAggregator
    
    fn __init__(config: SystemConfig):
        self.config = config
        self.expert_registry = ExpertRegistry(config)
        self.gating_network = GatingNetworkService(config)
        self.load_balancer = LoadBalancerService(config)
        self.expert_processor = ExpertProcessor(self.expert_registry, config.hidden_dim // config.num_experts)
        self.result_aggregator = ResultAggregator("weighted_average")
        
        print("🚀 Expert Dispatcher Service Initialized")
        print("   - Number of Experts: {}".format(config.num_experts))
        print("   - Expert Capacity: {}".format(config.hidden_dim // config.num_experts))
        print("   - Aggregation Method: Weighted Average")
    
    fn dispatch_tokens(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Dispatch tokens to appropriate experts and aggregate results"""
        var shape = input.shape()
        var batch_size = shape[0]
        var seq_len = shape[1]
        
        # Step 1: Compute gating scores
        var gating_scores = self.gating_network.compute_gating_scores(input)
        
        # Step 2: Apply load balancing
        var balanced_scores = self.load_balancer.apply_load_balancing(gating_scores)
        
        # Step 3: Apply softmax to get probabilities
        var expert_probabilities = self.gating_network.apply_softmax(balanced_scores)
        
        # Step 4: Make dispatch decisions
        var dispatch_decisions = self.make_dispatch_decisions(expert_probabilities)
        
        # Step 5: Process through experts
        var expert_outputs = self.process_through_experts(input, dispatch_decisions)
        
        # Step 6: Aggregate results
        var final_output = self.result_aggregator.aggregate_expert_outputs(expert_outputs, dispatch_decisions)
        
        # Step 7: Update load balancer
        self.update_load_balancer(dispatch_decisions)
        
        return final_output
    
    fn make_dispatch_decisions(self, expert_probabilities: Tensor[DType.float32]) -> Tensor[DispatchDecision]:
        """Make dispatch decisions based on expert probabilities"""
        var shape = expert_probabilities.shape()
        var decisions = Tensor[DispatchDecision](shape[0] * shape[1])
        var decision_count = 0
        
        for b in range(shape[0]):
            for s in range(shape[1]):
                # Find best expert for this token
                var best_expert_id = 0
                var best_confidence = expert_probabilities[b, s, 0]
                
                for expert_idx in range(shape[2]):
                    if expert_probabilities[b, s, expert_idx] > best_confidence:
                        best_confidence = expert_probabilities[b, s, expert_idx]
                        best_expert_id = expert_idx
                
                # Check if expert is available
                if not self.load_balancer.is_expert_available(best_expert_id):
                    # Find next best available expert
                    for expert_idx in range(shape[2]):
                        if self.load_balancer.is_expert_available(expert_idx):
                            best_expert_id = expert_idx
                            best_confidence = expert_probabilities[b, s, expert_idx]
                            break
                
                var load_penalty = self.load_balancer.get_expert_load(best_expert_id) * 0.1
                var expert_type = "expert_{}".format(best_expert_id)  # Simplified
                
                decisions[decision_count] = DispatchDecision(best_expert_id, expert_type, best_confidence, load_penalty)
                decision_count += 1
        
        return decisions
    
    fn process_through_experts(self, input: Tensor[DType.float32], dispatch_decisions: Tensor[DispatchDecision]) -> Tensor[Tensor[DType.float32]]:
        """Process input through selected experts"""
        var shape = input.shape()
        var expert_outputs = Tensor[Tensor[DType.float32]](dispatch_decisions.shape()[0])
        
        for decision_idx in range(dispatch_decisions.shape()[0]):
            var decision = dispatch_decisions[decision_idx]
            var token_idx = decision_idx
            
            if token_idx < shape[0] * shape[1]:
                var b = token_idx / shape[1]
                var s = token_idx % shape[1]
                
                var token_input = Tensor[DType.float32](shape[2])
                for i in range(shape[2]):
                    token_input[i] = input[b, s, i]
                
                var token_output = self.expert_processor.process_through_expert(token_input, decision.expert_id)
                expert_outputs[decision_idx] = token_output
        
        return expert_outputs
    
    fn update_load_balancer(mut self, dispatch_decisions: Tensor[DispatchDecision]):
        """Update load balancer with dispatch decisions"""
        for decision in dispatch_decisions:
            self.load_balancer.update_expert_load(decision.expert_id, 0.1)  # Add load for each token
        
        # Apply decay to all loads
        self.load_balancer.decay_all_loads()
    
    fn register_expert(mut self, expert: ExpertInterface, expert_config):
        """Register a new expert"""
        self.expert_registry.register_expert(expert, expert_config)
    
    fn get_dispatcher_info(self) -> String:
        """Get dispatcher information"""
        var info = "🚀 Expert Dispatcher Service Information\n"
        info += "=" * 40 + "\n"
        info += "Number of Experts: {}\n".format(self.config.num_experts)
        info += "Expert Capacity: {}\n".format(self.config.hidden_dim // self.config.num_experts)
        info += "Aggregation Method: {}\n".format(self.result_aggregator.aggregation_method)
        
        info += "\n"
        info += self.expert_registry.get_registry_info()
        info += "\n"
        info += self.gating_network.get_service_info()
        info += "\n"
        info += self.load_balancer.get_balancer_info()
        
        return info

# Factory function
fn create_expert_dispatcher_service(config: SystemConfig) -> ExpertDispatcherService:
    """Create expert dispatcher service"""
    return ExpertDispatcherService(config)
