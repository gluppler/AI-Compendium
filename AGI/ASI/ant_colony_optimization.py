import math
import random

def euclidean_distance(city1, city2):
    dx = city1[0] - city2[0]
    dy = city1[1] - city2[1]
    return math.sqrt(dx * dx + dy * dy)

def ant_colony_optimization(cities, num_ants=10, num_iterations=20,
                             evaporation_rate=0.7, alpha=1.0, beta=5.0, q=10.0):
    if not cities:
        return None
    if num_ants == 0:
        return None

    n = len(cities)
    pheromones = [[1.0] * n for _ in range(n)]
    best_route = []
    best_distance = float('inf')

    for _ in range(num_iterations):
        routes = []
        for _ in range(num_ants):
            route = [0]
            unvisited = set(range(1, n))
            current = 0

            while unvisited:
                probabilities = []
                total = 0.0
                for city in unvisited:
                    phero = pheromones[current][city]
                    dist = euclidean_distance(cities[current], cities[city])
                    heuristic = 1.0 / dist if dist > 0 else 0.0
                    prob = (phero ** alpha) * (heuristic ** beta)
                    probabilities.append((city, prob))
                    total += prob

                r = random.random() * total
                for city, prob in probabilities:
                    r -= prob
                    if r <= 0.0:
                        next_city = city
                        break
                else:
                    next_city = probabilities[-1][0]

                route.append(next_city)
                unvisited.remove(next_city)
                current = next_city

            route.append(0)
            routes.append(route)

        for route in routes:
            distance = sum(euclidean_distance(cities[route[i]], cities[route[i+1]])
                          for i in range(len(route) - 1))
            if distance < best_distance:
                best_distance = distance
                best_route = route[:]

        # Update pheromones
        for i in range(n):
            for j in range(n):
                pheromones[i][j] *= evaporation_rate

        for route in routes:
            distance = sum(euclidean_distance(cities[route[i]], cities[route[i+1]])
                          for i in range(len(route) - 1))
            deposit = q / distance if distance > 0 else 0
            for i in range(len(route) - 1):
                a, b = route[i], route[i+1]
                pheromones[a][b] += deposit
                pheromones[b][a] += deposit

    if not best_route:
        return None
    return (best_route, best_distance)


if __name__ == "__main__":
    cities = [(0.0, 0.0), (2.0, 2.0)]
    result = ant_colony_optimization(cities, 5, 5, 0.7, 1.0, 5.0, 10.0)
    assert result is not None
    route, distance = result
    assert route == [0, 1, 0]
    expected = 2.0 * math.sqrt(8.0)
    assert abs(distance - expected) < 0.001

    assert ant_colony_optimization([]) is None
    print("ant_colony_optimization: OK")
