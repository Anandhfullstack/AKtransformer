import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
    sqrt_dim_head = query.shape[-1] ** 0.5

    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / sqrt_dim_head

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    weight = F.softmax(scores, dim=-1)
    return torch.matmul(weight, value)


import torch

# create sample input tensors
query = torch.randn((2, 4, 6))  # shape: (batch_size, seq_len, embedding_dim)
key = torch.randn((2, 4, 6))  # shape: (batch_size, seq_len, embedding_dim)
value = torch.randn((2, 4, 6))  # shape: (batch_size, seq_len, embedding_dim)
mask = None

sqrt_dim_head = query.shape[2]

print(sqrt_dim_head)
# apply attention function
output = attention(query, key, value, mask)

# print the output tensor shape
print(output.shape)  # should be (2, 4, 6)
