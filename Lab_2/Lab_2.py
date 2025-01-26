import numpy as np
import matplotlib.pyplot as plt

def plot(title, x_label, y_label, data):
    plt.figure()
    for d in data:
        plt.plot(d['x'], d['y'], label=d['label'])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    plt.show()

def split_intervals(values, n):
    step = (max(values) - min(values)) / n
    intervals = [(min(values) + i * step, min(values) + (i + 1) * step) for i in range(n)]
    return intervals

def gaussmf(x, center, sigma):
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def gbellmf(x, a, b, c):
    return 1 / (1 + (np.abs((x - c) / a) ** (2 * b)))

def create_gauss_mf(universe, intervals, names):
    mfs = {}
    for interval, name in zip(intervals, names):
        center = (interval[0] + interval[1]) / 2
        mfs[name] = gaussmf(universe, center, 0.1)
    return mfs

def create_gbell_mf(universe, intervals, names):
    mfs = {}
    for interval, name in zip(intervals, names):
        center = (interval[0] + interval[1]) / 2
        mfs[name] = gbellmf(universe, 0.1, 5, center)
    return mfs

x_values = np.arange(1, 7, 0.1)
y_values = np.sin(np.abs(x_values)) * np.cos(3 * x_values / 2)
z_values = x_values * np.sin(x_values + y_values)

plot('Function y', 'x', 'y', [{'x': x_values, 'y': y_values, 'label': 'y = sin(|x|) * cos(3x/2)'}])
plot('Function z', 'x', 'z', [{'x': x_values, 'y': z_values, 'label': 'z = x * sin(x + y)'}])

x_intervals = split_intervals(x_values, 6)
y_intervals = split_intervals(y_values, 6)
z_intervals = split_intervals(z_values, 9)

x_universe = x_values
y_universe = np.linspace(min(y_values), max(y_values), 100)
z_universe = np.linspace(min(z_values), max(z_values), 100)

x_gauss_mfs = create_gauss_mf(x_universe, x_intervals, [f'mx{i}' for i in range(1, 7)])
y_gauss_mfs = create_gauss_mf(y_universe, y_intervals, [f'my{i}' for i in range(1, 7)])
z_gauss_mfs = create_gauss_mf(z_universe, z_intervals, [f'mz{i}' for i in range(1, 10)])

x_bell_mfs = create_gbell_mf(x_universe, x_intervals, [f'bx{i}' for i in range(1, 7)])
y_bell_mfs = create_gbell_mf(y_universe, y_intervals, [f'by{i}' for i in range(1, 7)])
z_bell_mfs = create_gbell_mf(z_universe, z_intervals, [f'bz{i}' for i in range(1, 10)])

def plot_mfs(title, universe, mfs):
    data = [{'x': universe, 'y': mf, 'label': name} for name, mf in mfs.items()]
    plot(title, 'x', 'Membership Degree', data)

plot_mfs('Gaussian MFs for x', x_universe, x_gauss_mfs)
plot_mfs('Generalized Bell MFs for x', x_universe, x_bell_mfs)

def create_rules(x_mfs, y_mfs, z_mfs):
    rules = []
    z_mfs_keys = list(z_mfs.keys())
    for i, x_mf in enumerate(x_mfs.keys()):
        for j, y_mf in enumerate(y_mfs.keys()):
            z_mf = z_mfs_keys[(i + j) % len(z_mfs_keys)]
            rules.append((x_mf, y_mf, z_mf))
    return rules

rules = create_rules(x_gauss_mfs, y_gauss_mfs, z_gauss_mfs)

results = []
for xv, yv in zip(x_values, y_values):
    z_val = 0
    for rule in rules:
        x_mf, y_mf, z_mf = rule
        x_degree = x_gauss_mfs[x_mf][np.abs(x_universe - xv).argmin()]
        y_degree = y_gauss_mfs[y_mf][np.abs(y_universe - yv).argmin()]
        z_val += min(x_degree, y_degree) * z_gauss_mfs[z_mf].mean()
    results.append(z_val)

plot('Simulation Results', 'x', 'z', [
    {'x': x_values, 'y': z_values, 'label': 'Original Function z'},
    {'x': x_values, 'y': results, 'label': 'Fuzzy Model z'}
])
