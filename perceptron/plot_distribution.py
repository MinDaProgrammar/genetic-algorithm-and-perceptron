# Measures how close the learned perceptron weights are to the
# true line y = 2x + 1 as more training points are added.
# Used to generate data visualisation for Q3.

import random
import matplotlib.pyplot as plt

def compute_net(weights, inputs):
    net = weights[0] + weights[1]*inputs[0] + weights[2]*inputs[1]
    return round(net, 10)

def activate(net):
    return 1 if net >= 0 else 0

def train(inputs, targets, initial_weights, lr=0.1):
    weights = initial_weights[:]
    passes = 0
    while True:
        passes += 1
        error_found = False
        for x, t in zip(inputs, targets):
            net    = compute_net(weights, x)
            output = activate(net)
            if output != t:
                error = t - output
                weights[0] += lr * error * 1
                weights[1] += lr * error * x[0]
                weights[2] += lr * error * x[1]
                error_found = True
                break
        if not error_found:
            break
    return weights

# ── Training sets ──────────────────────────────────────────────────
# True line: y = 2x + 1  ->  slope = 2, intercept = 1

inputs_4  = [[0,0], [0,2], [2,0], [2,6]]
targets_4 = [0, 1, 0, 1]

inputs_8  = [[0,0], [0,2], [2,0], [2,6],
             [1,0], [1,4], [3,2], [3,8]]
targets_8 = [0, 1, 0, 1, 0, 1, 0, 1]

inputs_12 = [[0,0], [0,2], [2,0], [2,6],
             [1,0], [1,4], [3,2], [3,8],
             [0.5,0], [0.5,2.5], [1.5,1], [1.5,5]]
targets_12 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

datasets = [
    ("4 points",  inputs_4,  targets_4),
    ("8 points",  inputs_8,  targets_8),
    ("12 points", inputs_12, targets_12),
]

# ── Run trials ─────────────────────────────────────────────────────
TRIALS = 100
TRUE_SLOPE     = 2.0
TRUE_INTERCEPT = 1.0

results = {}
for name, inputs, targets in datasets:
    slopes     = []
    intercepts = []
    for _ in range(TRIALS):
        w = [random.uniform(-0.5, 0.5) for _ in range(3)]
        w = train(inputs, targets, w)
        if w[2] != 0:  # avoid division by zero
            slopes.append(-w[1] / w[2])
            intercepts.append(-w[0] / w[2])
    results[name] = (slopes, intercepts)

# ── Plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

colors = ['steelblue', 'darkorange', 'seagreen']

for (name, (slopes, intercepts)), color in zip(results.items(), colors):
    axes[0].hist(slopes, bins=20, alpha=0.6, label=name,
                 color=color, edgecolor='white')
    axes[1].hist(intercepts, bins=20, alpha=0.6, label=name,
                 color=color, edgecolor='white')

axes[0].axvline(TRUE_SLOPE, color='red', linestyle='--', label='true slope (2)')
axes[0].set_title('Learned slope  (-W1/W2)')
axes[0].set_xlabel('Slope')
axes[0].set_ylabel('Frequency')
axes[0].legend()

axes[1].axvline(TRUE_INTERCEPT, color='red', linestyle='--', label='true intercept (1)')
axes[1].set_title('Learned intercept  (-W0/W2)')
axes[1].set_xlabel('Intercept')
axes[1].legend()

plt.suptitle(f'Learned boundary vs true line y = 2x+1  ({TRIALS} trials)')
plt.tight_layout()
plt.savefig('passes_distribution.png', dpi=150)
plt.show()

# ── Print summary ──────────────────────────────────────────────────
print(f"{'':12}  {'slope mean':>12}  {'slope std':>10}  "
      f"{'intercept mean':>15}  {'intercept std':>13}")
for name, (slopes, intercepts) in results.items():
    print(f"{name:12}  "
          f"{sum(slopes)/len(slopes):>12.3f}  "
          f"{(sum((s - sum(slopes)/len(slopes))**2 for s in slopes)/len(slopes))**0.5:>10.3f}  "
          f"{sum(intercepts)/len(intercepts):>15.3f}  "
          f"{(sum((i - sum(intercepts)/len(intercepts))**2 for i in intercepts)/len(intercepts))**0.5:>13.3f}")
