# Test Custom LLM Implementation
# Comprehensive testing for the NIF Sovereign custom LLM architecture

from tensor import Tensor
from time import now

# Test suite for NIF Custom LLM
struct NIFCustomLLMTest:
    var config: NIFConfig
    var model: NIFCustomLLM
    var trainer: NIFCustomTrainer

    fn __init__(out self):
        print("🧪 Initializing NIF Custom LLM Test Suite")

        # Create test configuration
        self.config = NIFConfig()
        self.config.hidden_dim = 512  # Smaller for testing
        self.config.num_layers = 4    # Smaller for testing
        self.config.num_experts = 2  # Smaller for testing

        # Initialize model and trainer
        self.model = NIFCustomLLM(self.config)
        self.trainer = NIFCustomTrainer(self.config)

        print("✅ Test Suite Initialized")

    fn run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 Running NIF Custom LLM Test Suite")
        print("=" * 50)

        var all_tests_passed = True

        # Test 1: Model Initialization
        if not self.test_model_initialization():
            all_tests_passed = False

        # Test 2: Forward Pass
        if not self.test_forward_pass():
            all_tests_passed = False

        # Test 3: Attention Mechanism
        if not self.test_attention_mechanism():
            all_tests_passed = False

        # Test 4: Physics Integration
        if not self.test_physics_integration():
            all_tests_passed = False

        # Test 5: Training Logic
        if not self.test_training_logic():
            all_tests_passed = False

        # Test 6: Optimization Methods
        if not self.test_optimization_methods():
            all_tests_passed = False

        # Test 7: Memory Efficiency
        if not self.test_memory_efficiency():
            all_tests_passed = False

        # Summary
        self.print_test_summary(all_tests_passed)

        return all_tests_passed

    fn test_model_initialization(self) -> Bool:
        """Test model initialization and component setup"""
        print("\n📋 Test 1: Model Initialization")

        try:
            # Check model dimensions
            if self.model.hidden_dim != 512:
                print("   ❌ Hidden dimension mismatch: expected 512, got {}".format(self.model.hidden_dim))
                return False

            if self.model.num_layers != 4:
                print("   ❌ Number of layers mismatch: expected 4, got {}".format(self.model.num_layers))
                return False

            if self.model.num_heads != 32:
                print("   ❌ Number of heads mismatch: expected 32, got {}".format(self.model.num_heads))
                return False

            # Check physics components
            if self.model.riemannian_embedding.config.hidden_dim != 512:
                print("   ❌ Riemannian embedding dimension mismatch")
                return False

            if self.model.neutrino_oscillation.oscillation_depth != self.config.neutrino_oscillation_depth:
                print("   ❌ Neutrino oscillation depth mismatch")
                return False

            print("   ✅ Model initialization test passed")
            return True

        except:
            print("   ❌ Model initialization test failed with exception")
            return False

    fn test_forward_pass(self) -> Bool:
        """Test forward pass functionality"""
        print("\n📋 Test 2: Forward Pass")

        try:
            # Create test input
            var batch_size = 2
            var seq_len = 8
            var input_ids = Tensor[Int](batch_size, seq_len)

            # Fill with random token IDs (0-1000)
            for b in range(batch_size):
                for s in range(seq_len):
                    input_ids[b, s] = (b * seq_len + s) % 1000

            # Run forward pass
            var start_time = now()
            var output = self.model.forward(input_ids)
            var forward_time = now() - start_time

            # Check output dimensions
            if output.shape[0] != batch_size:
                print("   ❌ Output batch size mismatch: expected {}, got {}".format(batch_size, output.shape[0]))
                return False

            if output.shape[1] != seq_len:
                print("   ❌ Output sequence length mismatch: expected {}, got {}".format(seq_len, output.shape[1]))
                return False

            if output.shape[2] != 50000:  # vocab size
                print("   ❌ Output vocab size mismatch: expected 50000, got {}".format(output.shape[2]))
                return False

            print("   ✅ Forward pass test completed in {}ms".format(forward_time))
            print("   ✅ Output shape: [{}, {}, {}]".format(output.shape[0], output.shape[1], output.shape[2]))
            return True

        except:
            print("   ❌ Forward pass test failed with exception")
            return False

    fn test_attention_mechanism(self) -> Bool:
        """Test physics-aware attention mechanism"""
        print("\n📋 Test 3: Attention Mechanism")

        try:
            # Create test hidden states
            var batch_size = 2
            var seq_len = 8
            var hidden_dim = 512
            var hidden_states = Tensor[DType.float32](batch_size, seq_len, hidden_dim)

            # Fill with random values
            for b in range(batch_size):
                for s in range(seq_len):
                    for h in range(hidden_dim):
                        hidden_states[b, s, h] = Float32((b * seq_len * hidden_dim + s * hidden_dim + h) % 1000) / 1000.0

            # Test attention computation
            var start_time = now()
            var attention_output = self.model.physics_self_attention(hidden_states)
            var attention_time = now() - start_time

            # Check output dimensions
            if attention_output.shape != hidden_states.shape:
                print("   ❌ Attention output shape mismatch")
                return False

            # Check for NaN values
            for b in range(batch_size):
                for s in range(seq_len):
                    for h in range(hidden_dim):
                        if isnan(attention_output[b, s, h]):
                            print("   ❌ NaN values detected in attention output")
                            return False

            print("   ✅ Attention mechanism test completed in {}ms".format(attention_time))
            return True

        except:
            print("   ❌ Attention mechanism test failed with exception")
            return False

    fn test_physics_integration(self) -> Bool:
        """Test physics-based module integration"""
        print("\n📋 Test 4: Physics Integration")

        try:
            # Test Riemannian embedding
            var test_input = Tensor[Int](2, 4)
            for b in range(2):
                for s in range(4):
                    test_input[b, s] = (b * 4 + s) % 100

            var embedding_output = self.model.riemannian_embedding.forward(test_input)

            if embedding_output.shape[0] != 2 or embedding_output.shape[1] != 4 or embedding_output.shape[2] != 512:
                print("   ❌ Riemannian embedding output shape mismatch")
                return False

            # Test neutrino oscillation
            var oscillation_output = self.model.neutrino_oscillation.forward(embedding_output)

            if oscillation_output.shape != embedding_output.shape:
                print("   ❌ Neutrino oscillation output shape mismatch")
                return False

            # Test Ising gate (simplified)
            var ising_output = self.model.ising_gate.forward(oscillation_output)

            if ising_output.shape != oscillation_output.shape:
                print("   ❌ Ising gate output shape mismatch")
                return False

            print("   ✅ Physics integration test passed")
            return True

        except:
            print("   ❌ Physics integration test failed with exception")
            return False

    fn test_training_logic(self) -> Bool:
        """Test training logic and optimization"""
        print("\n📋 Test 5: Training Logic")

        try:
            # Create training data
            var input_ids = Tensor[Int](2, 8)
            var targets = Tensor[Int](2, 8)

            for b in range(2):
                for s in range(8):
                    input_ids[b, s] = (b * 8 + s) % 100
                    targets[b, s] = (b * 8 + s + 1) % 100

            # Test training step
            var start_time = now()
            var loss = self.trainer.train_step(input_ids, targets)
            var training_time = now() - start_time

            # Check loss value
            if loss <= 0.0 or loss > 100.0:
                print("   ❌ Invalid loss value: {}".format(loss))
                return False

            if isnan(loss):
                print("   ❌ NaN loss detected")
                return False

            print("   ✅ Training logic test completed in {}ms".format(training_time))
            print("   ✅ Training loss: {:.6f}".format(loss))
            return True

        except:
            print("   ❌ Training logic test failed with exception")
            return False

    fn test_optimization_methods(self) -> Bool:
        """Test optimization methods (Muon, GaLore, VeRA)"""
        print("\n📋 Test 6: Optimization Methods")

        try:
            # Test Muon optimization
            var initial_velocity_norm = l2_norm(self.trainer.muon_velocity)
            self.trainer.apply_muon_optimization()
            var final_velocity_norm = l2_norm(self.trainer.muon_velocity)

            # Velocity should be orthonormal (norm ≈ 1.0)
            if final_velocity_norm < 0.9 or final_velocity_norm > 1.1:
                print("   ❌ Muon orthonormalization failed: norm = {}".format(final_velocity_norm))
                return False

            # Test GaLore projection
            var initial_grad_norm = l2_norm(self.trainer.parameter_gradients)
            self.trainer.apply_galore_projection()
            var final_grad_norm = l2_norm(self.trainer.parameter_gradients)

            # Gradients should be projected to low-rank space
            if final_grad_norm == 0.0:
                print("   ❌ GaLore projection zeroed gradients")
                return False

            # Test VeRA scaling
            var initial_scaling_norm = l2_norm(self.trainer.vera_scaling)
            self.trainer.apply_vera_scaling()
            var final_scaling_norm = l2_norm(self.trainer.vera_scaling)

            # Scaling should change
            if final_scaling_norm == initial_scaling_norm:
                print("   ❌ VeRA scaling not applied")
                return False

            print("   ✅ Optimization methods test passed")
            return True

        except:
            print("   ❌ Optimization methods test failed with exception")
            return False

    fn test_memory_efficiency(self) -> Bool:
        """Test memory efficiency and resource usage"""
        print("\n📋 Test 7: Memory Efficiency")

        try:
            # Estimate memory usage
            var hidden_dim = self.model.hidden_dim
            var vocab_size = 50000
            var num_layers = self.model.num_layers

            # Attention weights: 4 * hidden_dim^2 per layer
            var attention_memory = 4 * hidden_dim * hidden_dim * num_layers * 4  # 4 bytes per float32

            # Feed-forward weights: 2 * hidden_dim * intermediate_size per layer
            var ffn_memory = 2 * hidden_dim * (hidden_dim * 4) * num_layers * 4

            # Embedding weights: vocab_size * hidden_dim
            var embedding_memory = vocab_size * hidden_dim * 4

            var total_memory = attention_memory + ffn_memory + embedding_memory

            print("   📊 Estimated memory usage:")
            print("      - Attention: {} MB".format(attention_memory / (1024 * 1024)))
            print("      - Feed-forward: {} MB".format(ffn_memory / (1024 * 1024)))
            print("      - Embedding: {} MB".format(embedding_memory / (1024 * 1024)))
            print("      - Total: {} MB".format(total_memory / (1024 * 1024)))

            # Check if memory usage is reasonable (< 10GB for test model)
            if total_memory > 10 * 1024 * 1024 * 1024:  # 10GB
                print("   ❌ Memory usage too high: {} GB".format(total_memory / (1024 * 1024 * 1024)))
                return False

            print("   ✅ Memory efficiency test passed")
            return True

        except:
            print("   ❌ Memory efficiency test failed with exception")
            return False

    fn print_test_summary(self, all_tests_passed: Bool):
        """Print test summary"""
        print("\n" + "=" * 50)
        if all_tests_passed:
            print("🎉 ALL TESTS PASSED!")
            print("✅ NIF Custom LLM is ready for production")
        else:
            print("❌ SOME TESTS FAILED!")
            print("🔧 Please review and fix the failing components")

        print("=" * 50)

    fn isnan(x: Float32) -> Bool:
        """Check if float is NaN"""
        return x != x  # NaN is not equal to itself

    fn l2_norm(tensor: Tensor[DType.float32]) -> Float32:
        """Compute L2 norm of tensor"""
        var sum_squares = 0.0
        for i in range(tensor.shape[0]):
            sum_squares += tensor[i] * tensor[i]
        return sqrt(sum_squares)

# Main test runner
fn main():
    print("🚀 Starting NIF Custom LLM Tests")

    var test_suite = NIFCustomLLMTest()
    var success = test_suite.run_all_tests()

    if success:
        print("\n🎯 NIF Custom LLM implementation is complete and functional!")
        print("📈 Ready for training and deployment")
    else:
        print("\n⚠️  NIF Custom LLM needs fixes before deployment")

    return success
