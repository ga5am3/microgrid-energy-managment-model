
# ============================================================================
# IMPORTS
# ============================================================================

import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from tcl_env_dqn_1 import MicroGridEnv

# ============================================================================
# Discrete Sine-Cosine Algorithm (Shubham Gupta)
# ============================================================================


class DSCA:
    """Discrete Sine-Cosine: X(t+1,inew) = Xti ⊕ (Cti ⊗ (Xtd ⊖ Xti))
       Cti = r1 * sin(r2) or r1 * cos(r2) (r3 > 0.5 or r3 <= 0.5)

    """

    def n_features(self, xi, xd):
        # Compare with the best solution and quantify differences
        F = 0
        for i in range(len(xi)):
            if xi[i] == xd[i]:
                b = 0
            else:
                b = 1
            F += b
        return F

    def update_sol(self, xi, xd, nf):
        # Update xi with the best solution
        for j in range(round(nf)):
            p = random.randint(0, len(xi)-1)
            xi[p] = xd[p]
        return xi

    def algorithm(self, SearchAgents_n, Max_iter, prob, env, n_day):
        # Destination position
        D_pos = []
        D_score = float("-inf")
        Convergence_curve = np.zeros(Max_iter)

        # Initialize the positions of search agents randomly (solutions)
        # Each solution: list of 24 actions, for 24 hours
        Pos = []
        for z in range(SearchAgents_n):
            q = []
            for t in range(24):
                p = random.choice(ACTIONS)
                q.append(p)
            Pos.append(q)
        print("Initial Position", Pos)

        # Main loop
        env.reset(day=n_day)
        for m in range(Max_iter):
            # Evaluate the fitness of each search agent
            for i in range(SearchAgents_n):
                r = 0
                for t in range(24):
                    state, reward, terminal, _ = env.step(Pos[i][t])
                    r += reward
                    # Environment reset
                    if terminal:
                        env.reset(day=n_day)

                # fitness = r / 24  # Average reward
                fitness = r  # Total reward
                # Update Dest_Score (Best solution)
                if fitness > D_score:
                    D_score = fitness
                    D_pos = Pos[i].copy()
                    print("Best solution", D_score, "Position", D_pos)

            # Update the Position of search agents
            for i in range(SearchAgents_n):
                r4 = random.random()
                if r4 > prob:
                    # Random solution
                    for x in range(24):
                        Pos[i][x] = random.choice(ACTIONS)
                    # print("Random solution", Pos[i])
                else:
                    r1 = m / Max_iter
                    r2 = (2 * np.pi) * random.random()
                    r3 = random.random()
                    for x in range(24):
                        # Calculate number of features
                        F = self.n_features(Pos[i][x], D_pos[x])
                        if r3 > 0.5:
                            nf = abs(r1 * np.sin(r2)) * F
                        else:
                            nf = abs(r1 * np.cos(r2)) * F
                        # Update solutions
                        if round(nf) > 0:
                            Pos[i][x] = self.update_sol(
                                Pos[i][x], D_pos[x], nf)
                            # print("Update solution", Pos[i][x])
                    # print("Position", Pos[i])

            Convergence_curve[m] = D_score

            if m % 1 == 0:
                print(["At iteration " + str(m) + " the best fitness is "
                      + str(D_score)])

        return D_pos, Convergence_curve


# ============================================================================
# MAIN
# ============================================================================

# Days range
i_day = 50
f_day = 60

# Save results
results_pos = []
results_conv = []
total_conv = []

ACTIONS = [[i, j, k, l] for i in range(4) for j in range(5) for k in range(2)
           for l in range(2)]


if __name__ == '__main__':
    env = MicroGridEnv(day0=i_day)
    # env.seedy(1)
    sol = DSCA()

    for d in range(i_day, f_day):
        D_pos, Convergence_curve = sol.algorithm(6, 500, 0.9, env, d)
        results_pos.append(D_pos)
        results_conv.append(Convergence_curve)
        total_conv.extend(Convergence_curve)


with open('prueba1.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    for m in range(10):
        writer.writerow(results_pos[m])
        writer.writerow(results_conv[m])


# ============================================================================
# PLOTS
# ============================================================================

plt.plot(total_conv, 'red')
plt.title('DSCA Convergence Curve for 10 days')
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.show()

# plt.plot(Convergence_curve, 'green')
# plt.title('DSCA Convergence Curve for 1 day')
# plt.xlabel('Iterations')
# plt.ylabel('Reward')
# plt.show()
