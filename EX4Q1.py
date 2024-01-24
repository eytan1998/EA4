import cvxpy
import numpy as np
import timeit
import matplotlib.pyplot as plt


def egalitarian_allocation_boolean(valuations: list[list[float]]):
    #      r1  r2  r3  r4  r5
    #  p1 [11, 11, 22, 33, 44]
    #  p2 [11, 22, 44, 55, 66]
    #  p3 [11, 33, 22, 11, 66]
    players = len(valuations)
    res = len(valuations[0])
    # matrix of cvxpy vars
    mVars = cvxpy.Variable((players, res), boolean=True)

    # utility for each player
    utility = [sum(mVars[j][i] * valuations[j][i] for i in range(res)) for j in range(players)]

    # (egalitarian) max the min
    min_utility = cvxpy.Variable()

    # all vars are 0 or 1
    bin_constraints = [0 <= mVars[i, j] for i in range(players) for j in range(res)]
    bin_constraints += [1 >= mVars[i, j] for i in range(players) for j in range(res)]

    # all utility bigger than min_utility
    utility_constraints = [utility[i] >= min_utility for i in range(players)]

    # the sum of each coll (how much each player take from r[i]) i LE to 1 (cant get more than the resource)
    sum_constraints = [cvxpy.sum(mVars[:, j]) <= 1 for j in range(res)]

    prob = cvxpy.Problem(cvxpy.Maximize(min_utility)
                         , bin_constraints + utility_constraints + sum_constraints)
    prob.solve()

    # print
    for i in range(players):
        player_val = 0
        first = True
        print(f"Player {i} gets items ", end="")
        for j in range(res):
            if round(mVars[i][j].value, 1) == 1:
                player_val += valuations[i][j]
                if first:
                    print(f'{j}', end="")
                    first = False
                else:
                    print(f', {j}', end="")
        print(f' with value of {player_val}')


def egalitarian_allocation(valuations: list[list[float]]):
    #      r1  r2  r3  r4  r5
    #  p1 [11, 11, 22, 33, 44]
    #  p2 [11, 22, 44, 55, 66]
    #  p3 [11, 33, 22, 11, 66]
    players = len(valuations)
    res = len(valuations[0])
    # matrix of cvxpy vars
    mVars = cvxpy.Variable((players, res))

    # utility for each player
    utility = [sum(mVars[j][i] * valuations[j][i] for i in range(res)) for j in range(players)]

    # (egalitarian) max the min
    min_utility = cvxpy.Variable()

    # all vars are 0 or 1
    bin_constraints = [0 <= mVars[i, j] for i in range(players) for j in range(res)]
    bin_constraints += [1 >= mVars[i, j] for i in range(players) for j in range(res)]

    # all utility bigger than min_utility
    utility_constraints = [utility[i] >= min_utility for i in range(players)]

    # the sum of each coll (how much each player take from r[i]) i LE to 1 (cant get more than the resource)
    sum_constraints = [cvxpy.sum(mVars[:, j]) <= 1 for j in range(res)]

    prob = cvxpy.Problem(cvxpy.Maximize(min_utility)
                         , bin_constraints + utility_constraints + sum_constraints)
    prob.solve(solver=cvxpy.ECOS)

    # print the result
    for i in range(players):
        print(f"player {i} receives ", end=" ")
        for j in range(res):
            if j == 0:
                print(f"{abs(round(mVars[i][j].value * 100, 2))}% of resource {j}", end="")
            else:
                print(f" and {abs(round(mVars[i][j].value * 100, 2))}% of resource {j}", end="")
        # for new line
        print()


if __name__ == '__main__':
    resources_amount = [10,20,30,40,50,60,70,80]
    execution_times_boolean = []
    execution_times = []
    times_to_run = 1

    for amount in resources_amount:
        # rnd res valuations between 10 and 100 with 5 players
        valuations = np.random.randint(10, 100, (5, amount)).tolist()

        execution_time_boolean = timeit.timeit(lambda: egalitarian_allocation_boolean(valuations), number=times_to_run)
        execution_time = timeit.timeit(lambda: egalitarian_allocation(valuations), number=times_to_run)

        execution_times_boolean.append(execution_time_boolean)
        execution_times.append(execution_time)

    # Create a graph
    plt.plot(resources_amount, execution_times_boolean, label="egalitarian_allocation_boolean", color="red", marker="o")
    plt.plot(resources_amount, execution_times, label="egalitarian_allocation", color="blue", marker="o")
    plt.title('Execution Time vs Resource amount')
    plt.xlabel('Number of resource')
    plt.ylabel('Execution Time (seconds) - less is better')
    plt.legend()
    plt.grid(True)
    plt.show()
