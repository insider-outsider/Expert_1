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
import numpy as np

# Input array (paste the full values from your question)
x = np.array([
 [-3.7476,  0.1523, -2.71,   6.472,  7.1142,  4.6742,  5.2134, -7.3956,  5.2793, -5.3849],
 [ 0.0132,  2.2883, -6.6087, -8.0126,  2.3187,  6.591,  -2.3418,  0.2374,  4.4914,  9.3202],
 [ -0.6048, 10.0772, -2.7239,  9.8585, -4.9513, -5.6764, -7.5507, -1.3586, -3.3513, -7.6325],
 [ 9.3446, -10.4391, -4.3017,  8.4482, -10.0345,  3.5202,  5.2652,  1.1649,  5.4627,  7.4446],
 [ 4.8904,  4.4207, -9.6771,  7.0472, -0.2595,  4.4236, -2.8757, -7.9476,  2.0327, -10.8131],
 [ 8.191,  1.4621,  1.1729, -5.0761, -7.4513, 10.3875,  4.4675,  0.523,  7.3403, 10.7181],
 [-6.5062, -7.8614, -9.108, -2.9355, -1.7569,  1.6111, 10.2238,  9.261, -3.156, -2.8548],
 [ 8.6142,  7.2752,  7.3001, -7.8067,  2.9039,  6.0091, -10.6173,  5.0877,  8.1579,  2.7171],
 [ 9.1148,  8.5546,  7.0697,  2.5485, -7.3367, -6.1486,  6.3515, -4.9502, -9.7528, -4.7476],
 [-7.2139, -6.6695,  7.4203, -3.5866, 10.7954, 10.2434,  9.0086,  7.6543, -7.5134, -3.686]
])

# Filter (kernel)
f = np.array([
 [ 2,  3,  0, -1],
 [ 0, -2,  2,  1],
 [-2,  0,  3,  2],
 [ 1,  2,  3, -3]
])

bias = 1

# --- Perform convolution manually with 'same' padding ---
pad = f.shape[0] // 2
x_padded = np.pad(x, pad_width=pad, mode='constant', constant_values=0)

out = np.zeros_like(x)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        region = x_padded[i:i+f.shape[0], j:j+f.shape[1]]
        out[i, j] = np.sum(region * f) + bias

# Apply ReLU activation
out = np.maximum(out, 0)

# Output at [6, 4]
print(out[6, 4])
