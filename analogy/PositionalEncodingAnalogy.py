import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding

# Define the dummy input tensor with batch_size = 128 and seq_len = 30
batch_size = 1
seq_len = 30
input_tensor = torch.randn(batch_size, seq_len)

# Instantiate the PositionalEncoding module with d_model = 512 and max_len = 512
d_model = 64
max_len = 64
device = torch.device('cpu')  # Use CPU as device for this example
pos_encoding_module = PositionalEncoding(d_model, max_len, device)

# Pass the input tensor through the PositionalEncoding module
output_tensor = pos_encoding_module(input_tensor)

# Print the output tensor for verification
print("Output tensor:")
print(output_tensor.shape)


import math
import torch


max_positions = 2
dim_embed = 4

pe = torch.zeros(max_positions, dim_embed)

for pos in range(max_positions):
    for i in range(0, dim_embed, 2):
        theta = pos / (10 ** (i / dim_embed))
        print(pos,"-----",i)
        print(pe[pos, i    ], "pe[pos, i    ]")
        val1 = math.sin(theta)
        val2 = math.cos(theta)
        pe[pos, i    ] = math.sin(theta)
        pe[pos, i + 1] = math.cos(theta)

print(pe)