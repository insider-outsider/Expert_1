from torch import nn

input_shape = (3, 32, 32)  

model = nn.Sequential(
    nn.Conv2d(3, 12, kernel_size=(5, 5), stride=2),
    nn.ReLU(),
    nn.Conv2d(12, 24, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(24, 24, kernel_size=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(24 * 12 * 12, 32),
    nn.Linear(32, 5)
)
