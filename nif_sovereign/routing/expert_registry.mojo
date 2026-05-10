# Expert Registry
# Atomic component for managing expert registration and lookup
# Single responsibility: expert lifecycle management

from tensor import Tensor
from nif_sovereign.system_config import SystemConfig
from nif_sovereign.interfaces.expert_interface import ExpertInterface, ExpertConfig

# Expert registry entry
struct ExpertEntry:
    var expert: ExpertInterface
    var config: ExpertConfig
    var is_active: Bool
    var current_load: Float32
    var registration_time: Int
    
    fn __init__(expert: ExpertInterface, config: ExpertConfig):
        self.expert = expert
        self.config = config
        self.is_active = True
        self.current_load = 0.0
        self.registration_time = 0  # Simplified timestamp

# Expert registry
struct ExpertRegistry:
    var config: SystemConfig
    var experts: Tensor[ExpertEntry]
    var expert_types: Tensor[String]
    var max_experts: Int
    var registered_count: Int
    
    fn __init__(out self, config: SystemConfig):
        self.config = config
        self.max_experts = config.num_experts
        self.experts = Tensor[ExpertEntry](self.max_experts)
        self.expert_types = Tensor[String](self.max_experts)
        self.registered_count = 0
        
        print("📋 Expert Registry Initialized")
        print("   - Max Experts: {}".format(self.max_experts))
        print("   - Registered: {}".format(self.registered_count))
    
    fn register_expert(mut self, expert: ExpertInterface, config: ExpertConfig) -> Bool:
        """Register a new expert in the registry"""
        if self.registered_count >= self.max_experts:
            print("⚠️ Expert registry full")
            return False
        
        var entry = ExpertEntry(expert, config)
        self.experts[self.registered_count] = entry
        self.expert_types[self.registered_count] = config.expert_type
        self.registered_count += 1
        
        print("✅ Expert Registered: {} (ID: {})".format(config.expert_type, config.expert_id))
        return True
    
    fn get_expert_by_id(self, expert_id: Int) -> ExpertInterface:
        """Get expert by ID"""
        for i in range(self.registered_count):
            if self.experts[i].config.expert_id == expert_id and self.experts[i].is_active:
                return self.experts[i].expert
        
        print("⚠️ Expert not found: {}".format(expert_id))
        return self.experts[0].expert  # Return first expert as fallback
    
    fn get_expert_by_type(self, expert_type: String) -> ExpertInterface:
        """Get expert by type"""
        for i in range(self.registered_count):
            if self.experts[i].config.expert_type == expert_type and self.experts[i].is_active:
                return self.experts[i].expert
        
        print("⚠️ Expert type not found: {}".format(expert_type))
        return self.experts[0].expert  # Return first expert as fallback
    
    fn get_all_experts(self) -> Tensor[ExpertInterface]:
        """Get all active experts"""
        var active_experts = Tensor[ExpertInterface](self.registered_count)
        var active_count = 0
        
        for i in range(self.registered_count):
            if self.experts[i].is_active:
                active_experts[active_count] = self.experts[i].expert
                active_count += 1
        
        return active_experts
    
    fn get_available_experts(self) -> Tensor[ExpertInterface]:
        """Get available experts (not at capacity)"""
        var available_experts = Tensor[ExpertInterface](self.registered_count)
        var available_count = 0
        
        for i in range(self.registered_count):
            if self.experts[i].is_active and self.experts[i].current_load < 1.0:
                available_experts[available_count] = self.experts[i].expert
                available_count += 1
        
        return available_experts
    
    fn update_expert_load(mut self, expert_id: Int, load: Float32):
        """Update expert load"""
        for i in range(self.registered_count):
            if self.experts[i].config.expert_id == expert_id:
                self.experts[i].current_load = load
                break
    
    fn deactivate_expert(mut self, expert_id: Int):
        """Deactivate an expert"""
        for i in range(self.registered_count):
            if self.experts[i].config.expert_id == expert_id:
                self.experts[i].is_active = False
                print("🔴 Expert Deactivated: {}".format(expert_id))
                break
    
    fn activate_expert(mut self, expert_id: Int):
        """Activate an expert"""
        for i in range(self.registered_count):
            if self.experts[i].config.expert_id == expert_id:
                self.experts[i].is_active = True
                print("🟢 Expert Activated: {}".format(expert_id))
                break
    
    fn get_registry_info(self) -> String:
        """Get registry information"""
        var info = "📋 Expert Registry Information\n"
        info += "=" * 30 + "\n"
        info += "Max Experts: {}\n".format(self.max_experts)
        info += "Registered: {}\n".format(self.registered_count)
        
        var active_count = 0
        for i in range(self.registered_count):
            if self.experts[i].is_active:
                active_count += 1
        
        info += "Active: {}\n".format(active_count)
        
        info += "\nExpert Types:\n"
        for i in range(self.registered_count):
            if self.experts[i].is_active:
                info += "  - {} (ID: {}, Load: {:.2f})\n".format(
                    self.experts[i].config.expert_type,
                    self.experts[i].config.expert_id,
                    self.experts[i].current_load
                )
        
        return info

# Factory function
fn create_expert_registry(config: SystemConfig) -> ExpertRegistry:
    """Create expert registry"""
    return ExpertRegistry(config)
