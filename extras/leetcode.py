

def canCompleteCircuit(gas: list[int], cost: list[int]) -> int:

    def is_valid(i: int) -> bool:
        trunk = 0
        for j in range(len(gas)):
            next_i = (i + j) % len(gas)
            trunk += gas[next_i]

            if trunk < cost[next_i]: return False

            trunk -= cost[next_i]

        return True

    for i in range(len(gas)):
        if is_valid(i): return i
    else:
        return -1


if __name__ == "__main__":

    gas = [1, 2, 3, 4, 5]
    cost= [3, 4, 5, 1, 2]

    print(canCompleteCircuit(gas, cost))
