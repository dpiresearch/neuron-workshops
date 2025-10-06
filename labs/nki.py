import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_tensor_add_kernel_(a_input, b_input):

  # Create output tensor
  c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

  # Load input data from device memory (HBM) to on-chip memory (SBUF)
  a_tile = nl.load(a_input)
  b_tile = nl.load(b_input)

  # compute a + b
  c_tile = a_tile + b_tile

  # return the final tensor
  nl.store(c_output, value=c_tile)

  # Transfer the ownership of `c_output` to the caller
  return c_output

a = np.random.rand(128, 512).astype(np.float16)
b = np.random.rand(128, 512).astype(np.float16)

output_nki = nki_tensor_add_kernel_(a, b)

output_np = a + b

allclose = np.allclose(output_np, output_nki, atol=1e-4, rtol=1e-2)
if allclose:
    print("NKI and NumPy match")
else:
    print("NKI and NumPy differ")

import torch
from torch_xla.core import xla_model as xm

device = xm.xla_device()

lhs_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
rhs_small = torch.rand((128, 512), dtype=torch.bfloat16, device=device)

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
  """NKI kernel to compute a 64x128x512 matrix multiplication operation

  Args:
      lhsT: an input tensor of shape [128,64], a left hand side argument of the
        matrix multiplication, delivered transposed for optimal performance
      rhs: an input tensor of shape [128,512], a right hand side argument of the
        matrix multiplication
  Returns:
      result: the resulting output tensor of shape [64,512]
  """
  result = nl.ndarray((64, 512), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Defining indexes for input LHS.T
  # - Note: here we take LayoutConstraint #1 into account:
  # "For MatMult, contraction axis must be mapped to P-dim"
  i_lhsT_p, i_lhsT_f = nl.mgrid[0:128, 0:64]

  # Defining indexes for input RHS
  # - Note: here we take LayoutConstraint #1 into account:
  # "For MatMult, contraction axis must be mapped to P-dim"
  i_rhs_p, i_rhs_f = nl.mgrid[0:128, 0:512]

  # Defining indexes for the output ([64,128]@[128,512] -> [64,512])
  i_out_p, i_out_f = nl.mgrid[0:64, 0:512]

  # Loading the inputs (HBM->SBUF)
  # Note: here we take Tile dtype definition into account,
  # which forces P-dim as the left most index
  lhs_tile = nl.load(lhsT[i_lhsT_p, i_lhsT_f])
  rhs_tile = nl.load(rhs[i_rhs_p, i_rhs_f])

  # Perform the matrix-multiplication
  # Note1: We set transpose_x to True, to indicate that the LHS input is transposed
  # Note2: A NKI matmul instruction always writes to PSUM in float32 data-type
  result_psum = nl.matmul(lhs_tile, rhs_tile, transpose_x=True)

  # Copy the result from PSUM back to SBUF, and cast to expected output data-type
  result_sbuf = nl.copy(result_psum, dtype=result.dtype)

  # The result of a [64,128] x [128,512] matrix multiplication has a shape of [64, 512].
  # This dictates which indices to use to address the result tile.
  nl.store(result[i_out_p, i_out_f], value=result_sbuf)

  return result

# Run NKI kernel
output_small = nki_matmul_basic_(lhs_small.T, rhs_small)

# Run torch reference
output_small_torch = torch.matmul(lhs_small, rhs_small)

# Compare results
print("Checking correctness of nki_matmul_basic")
if torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2):
  print("NKI and Torch match")
else:
  print("NKI and Torch differ")

