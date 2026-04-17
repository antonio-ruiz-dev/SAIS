
import torch.nn as nn
import inspect

# Check if transformer now returns attention
transformer = nn.TransformerEncoderLayer(d_model=384, nhead=8)

print('transformer:', transformer)
source = inspect.getsource(transformer.forward)
print(f'Source code:\n{source}')
if 'attn' in source:
    print('✓ Transformer modification appears successful')
else:
    print('⚠ Double-check transformer.py modifications')
