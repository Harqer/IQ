# Test different tensor imports
from std.tensor import Tensor
from std.math import sqrt, exp

fn main():
    var x = Tensor[DType.float32](10, 10)
    print("Tensor import successful!")
