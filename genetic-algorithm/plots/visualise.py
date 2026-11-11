import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

pop_sizes = [10, 25, 50, 100, 200, 500]
q1_gens = [346.7, 86.4, 31.9, 14.5, 9.5, 5.7]
q3_no_crossover = [10000, 134.4, 63.0, 34.0, 19.5, 12.7]
mutation_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
q2_gens = [588.8, 135.5, 71.9, 20.5, 12.9, 20.0]
q4_no_mutation = [10000, 10000, 10000, 10000, 10000, 10000]

# Q1
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(pop_sizes, q1_gens, marker='o', color='steelblue', linewidth=2)
ax.set_title('Q1: Population Size vs Generations to Converge', fontsize=13)
ax.set_xlabel('Population Size', fontsize=11)
ax.set_ylabel('Average Generations', fontsize=11)
plt.tight_layout()
plt.savefig('q1.png', dpi=150)
plt.close()

# Q2
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(mutation_rates, q2_gens, marker='o', color='steelblue', linewidth=2)
ax.axvline(x=0.1, color='red', linestyle='--', linewidth=1.2, label='Optimum (0.1)')
ax.set_xscale('log')
ax.set_title('Q2: Mutation Rate vs Generations to Converge', fontsize=13)
ax.set_xlabel('Mutation Rate (log scale)', fontsize=11)
ax.set_ylabel('Average Generations', fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig('q2.png', dpi=150)
plt.close()

# Q3
q3_plot = [min(v, 500) for v in q3_no_crossover]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(pop_sizes, q1_gens, marker='o', color='steelblue', linewidth=2, label='With crossover')
ax.plot(pop_sizes, q3_plot, marker='o', color='orange', linewidth=2, label='No crossover')
ax.annotate('DNF (>10000)', xy=(10, 500), xytext=(20, 460), fontsize=9, color='orange',
            arrowprops=dict(arrowstyle='->', color='orange'))
ax.set_ylim(0, 500)
ax.set_title('Q3: Effect of Crossover on Convergence', fontsize=13)
ax.set_xlabel('Population Size', fontsize=11)
ax.set_ylabel('Average Generations', fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig('q3.png', dpi=150)
plt.close()

# Q4
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(pop_sizes, q1_gens, marker='o', color='steelblue', linewidth=2, label='With mutation')
ax.plot(pop_sizes, q4_no_mutation, marker='o', color='orange', linewidth=2, label='No mutation (DNF, >10k gen)')
ax.set_yscale('log')
ax.set_title('Q4: Effect of Mutation on Convergence', fontsize=13)
ax.set_xlabel('Population Size', fontsize=11)
ax.set_ylabel('Average Generations (log scale)', fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig('q4.png', dpi=150)
plt.close()
