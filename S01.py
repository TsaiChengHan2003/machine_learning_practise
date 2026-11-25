import torch

b1 = torch.tensor([1, 2])
b2 = torch.tensor([2, -1])
alpha, beta = 3, 2

# 1. 直接計算 v
v = alpha * b1 + beta * b2

# 2. 將 b1、b2 作為 Column 組成矩陣 B
B = torch.stack([b1, b2], dim=1)

# 3. 建立係數向量 c
c = torch.tensor([alpha, beta])

# 4. 透過矩陣乘法計算 v2
v2 =B @ c.reshape(2, 1)

print(v, v.shape)
print(v2, v2.shape)