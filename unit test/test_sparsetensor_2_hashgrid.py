import numpy as np
import torch
from torch import nn
from gridencoder import GridEncoder

B = 3
D = 4

enc = GridEncoder(input_dim=D, num_levels=1, level_dim=4, base_resolution=B, log2_hashmap_size=19, align_corners=True).cuda()
# enc = GridEncoder(D=D, L=16, C=2, base_resolution=16).cuda()
enc.embeddings.data.zero_()

print(f"=== enc ===")
print(enc.embeddings.shape)
print(enc.embeddings.T)

li = np.linspace(-1, 1, B)

x = torch.tensor(np.vstack([d.ravel() for d in np.meshgrid(*([li] * D))]).T, dtype=torch.float32).cuda()

# x.requires_grad_(True)

print(f"=== x ===")
print(x.T)
print(x.shape)

# y = enc(x, calc_grad_inputs=False)
y = enc(x)

print(f"=== y ===")
print(y.shape)
print(y.T)

(y * x).sum().backward()

print(f"=== grad enc ===")
print(enc.embeddings.grad.shape)
print(enc.embeddings.grad.T)

enc.embeddings.data += enc.embeddings.grad
enc.embeddings.grad.zero_()
print(f"=== new enc ===")
print(enc.embeddings.shape)
print(enc.embeddings.T)

# print(x.grad.shape)
# print(x.grad)
print(f"=== ?x=embed ===")
print((x - enc.embeddings[:x.shape[0]] < 1e-3).all())

print('done')
