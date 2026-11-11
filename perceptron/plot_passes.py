# Compares how learned boundary quality changes with:
#   1. More balanced points (4, 8, 16, 32)
#   2. Unbalanced points — mostly above or mostly below the line
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
    while True:
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

# ── Helper: generate n balanced points around y = 2x + 1 ──────────
# Alternates: below point then above point
def make_balanced(n):
    # n must be even
    xs = [i * 3.0 / (n // 2) for i in range(n // 2)]
    inputs, targets = [], []
    for x in xs:
        true_y = 2*x + 1
        inputs.append([x, true_y - 1.5])  # clearly below
        inputs.append([x, true_y + 1.5])  # clearly above
        targets += [0, 1]
    return inputs, targets

# ── Helper: generate n mostly-above points (3 above : 1 below) ────
def make_mostly_above(n):
    inputs, targets = [], []
    xs = [i * 3.0 / n for i in range(n)]
    for i, x in enumerate(xs):
        true_y = 2*x + 1
        if i % 4 == 0:  # every 4th point is below
            inputs.append([x, true_y - 1.5])
            targets.append(0)
        else:
            inputs.append([x, true_y + 1.5])
            targets.append(1)
    return inputs, targets

# ── Helper: generate n mostly-below points (1 above : 3 below) ────
def make_mostly_below(n):
    inputs, targets = [], []
    xs = [i * 3.0 / n for i in range(n)]
    for i, x in enumerate(xs):
        true_y = 2*x + 1
        if i % 4 == 0:  # every 4th point is above
            inputs.append([x, true_y + 1.5])
            targets.append(1)
        else:
            inputs.append([x, true_y - 1.5])
            targets.append(0)
    return inputs, targets

# ── Build datasets ─────────────────────────────────────────────────
sizes = [4, 8, 16, 32]

balanced_datasets     = {f"{n} pts": make_balanced(n)     for n in sizes}
mostly_above_datasets = {f"{n} pts": make_mostly_above(n) for n in sizes}
mostly_below_datasets = {f"{n} pts": make_mostly_below(n) for n in sizes}

# ── Run trials ─────────────────────────────────────────────────────
TRIALS         = 100
TRUE_SLOPE     = 2.0
TRUE_INTERCEPT = 1.0

def run_trials(datasets):
    results = {}
    for name, (inputs, targets) in datasets.items():
        slopes, intercepts = [], []
        for _ in range(TRIALS):
            w = [random.uniform(-0.5, 0.5) for _ in range(3)]
            w = train(inputs, targets, w)
            if w[2] != 0:
                slopes.append(-w[1] / w[2])
                intercepts.append(-w[0] / w[2])
        results[name] = (slopes, intercepts)
    return results

balanced_results     = run_trials(balanced_datasets)
mostly_above_results = run_trials(mostly_above_datasets)
mostly_below_results = run_trials(mostly_below_datasets)

# ── Plotting helper ────────────────────────────────────────────────
colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']

def plot_results(results, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (name, (slopes, intercepts)), color in zip(results.items(), colors):
        # Clip outliers so all distributions are visible
        slopes_clipped     = [s for s in slopes     if -5 <= s <= 10]
        intercepts_clipped = [i for i in intercepts if -5 <= i <= 10]
        axes[0].hist(slopes_clipped,     bins=20, alpha=0.6, label=name,
                     color=color, edgecolor='white')
        axes[1].hist(intercepts_clipped, bins=20, alpha=0.6, label=name,
                     color=color, edgecolor='white')
    axes[0].axvline(TRUE_SLOPE,     color='red', linestyle='--', label='true (2)')
    axes[1].axvline(TRUE_INTERCEPT, color='red', linestyle='--', label='true (1)')
    axes[0].set_title('Learned slope  (-W1/W2)')
    axes[1].set_title('Learned intercept  (-W0/W2)')
    axes[0].set_xlabel('Slope');     axes[0].set_ylabel('Frequency'); axes[0].legend()
    axes[1].set_xlabel('Intercept'); axes[1].legend()
    axes[0].set_xlim(-5, 10)
    axes[1].set_xlim(-5, 10)
    plt.suptitle(f'{title}  ({TRIALS} trials, outliers clipped to [-5, 10])')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")

plot_results(balanced_results,     'Balanced points',      'plot_balanced.png')
plot_results(mostly_above_results, 'Mostly above the line','plot_mostly_above.png')
plot_results(mostly_below_results, 'Mostly below the line','plot_mostly_below.png')

plt.show()

# ── Print summary ──────────────────────────────────────────────────
def print_summary(label, results):
    print(f"\n--- {label} ---")
    print(f"{'':10}  {'slope mean':>10}  {'slope std':>9}  "
          f"{'intercept mean':>14}  {'intercept std':>13}")
    for name, (slopes, intercepts) in results.items():
        sm  = sum(slopes)/len(slopes)
        ss  = (sum((s-sm)**2 for s in slopes)/len(slopes))**0.5
        im  = sum(intercepts)/len(intercepts)
        ist = (sum((i-im)**2 for i in intercepts)/len(intercepts))**0.5
        print(f"  {name:8}  {sm:>10.3f}  {ss:>9.3f}  {im:>14.3f}  {ist:>13.3f}")

print_summary("Balanced",     balanced_results)
print_summary("Mostly above", mostly_above_results)
print_summary("Mostly below", mostly_below_results)
