import torch
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage

x_train = torch.rand((100, 10))
y_train = torch.rand(100).round().long()
model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

# Train for a few epochs
model.train()
for _ in range(10):
    logits = model(x_train)
    loss = F.cross_entropy(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema.update(model.parameters())

# Compare losses in test:

# Original
model.eval()
logits = model(x_train)
loss = F.cross_entropy(logits, y_train)
print(loss.item())

# With EMA
ema.copy_to(model.parameters())
logits = model(x_train)
loss = F.cross_entropy(logits, y_train)
print(loss.item())
