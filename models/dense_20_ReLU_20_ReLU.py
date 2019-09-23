import torch

Net = lambda n: torch.nn.Sequential(
    torch.nn.Linear(n, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 40),
    torch.nn.ReLU(),
    torch.nn.Linear(40, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1),
)
