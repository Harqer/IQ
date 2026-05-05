# Model Interface
# Common interfaces for model components

trait ProcessorInterface:
    fn process(inout self, input: String) -> String

trait ConfigurableInterface:
    fn configure(inout self, config: SystemConfig)

trait VerifiableInterface:
    fn verify(inout self) -> Bool

trait OptimizableInterface:
    fn optimize(inout self) -> Bool
