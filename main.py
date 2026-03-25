import heapq
import math
import time
import tracemalloc

coords = {
    'Hauptbahnhof': (50.9727, 11.0384),
    'Anger':        (50.9780, 11.0310),
    'Rieth':        (50.9900, 11.0197),
    'HELIOS':       (50.9886, 11.0276),
    'Daberstedt':   (50.9605, 11.0551),
    'Hueffner':     (50.9635, 11.0422),
    'Suedost':      (50.9580, 11.0600),
    'Stotternheim': (51.0200, 11.0500),
    'Nordklinik':   (51.0250, 11.0600),
}

hospitals = {'HELIOS', 'Hueffner', 'Nordklinik'}


def distance(a, b):
    lat1, lon1 = coords[a]
    lat2, lon2 = coords[b]
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111


graph = {
    'Hauptbahnhof': [('Anger', distance('Hauptbahnhof', 'Anger')),
                     ('Daberstedt', distance('Hauptbahnhof', 'Daberstedt'))],

    'Anger': [('Hauptbahnhof', distance('Anger', 'Hauptbahnhof')),
              ('HELIOS', distance('Anger', 'HELIOS')),
              ('Rieth', distance('Anger', 'Rieth'))],

    'Rieth': [('Anger', distance('Rieth', 'Anger')),
              ('HELIOS', distance('Rieth', 'HELIOS')),
              ('Stotternheim', distance('Rieth', 'Stotternheim'))],

    'HELIOS': [('Anger', distance('HELIOS', 'Anger')),
               ('Rieth', distance('HELIOS', 'Rieth'))],

    'Daberstedt': [('Hauptbahnhof', distance('Daberstedt', 'Hauptbahnhof')),
                   ('Hueffner', distance('Daberstedt', 'Hueffner')),
                   ('Suedost', distance('Daberstedt', 'Suedost'))],

    'Hueffner': [('Daberstedt', distance('Hueffner', 'Daberstedt')),
                 ('Suedost', distance('Hueffner', 'Suedost'))],

    'Suedost': [('Daberstedt', distance('Suedost', 'Daberstedt')),
                ('Hueffner', distance('Suedost', 'Hueffner'))],

    'Stotternheim': [('Rieth', distance('Stotternheim', 'Rieth')),
                     ('Nordklinik', distance('Stotternheim', 'Nordklinik'))],

    'Nordklinik': [('Stotternheim', distance('Nordklinik', 'Stotternheim'))]
}


def build_path(previous_nodes, target):
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()
    return path


def dijkstra(graph, start, hospitals):
    dist_map = {node: float('inf') for node in graph}
    previous_nodes = {node: None for node in graph}
    dist_map[start] = 0

    pq = [(0, start)]
    visited = set()
    visited_nodes = 0

    while pq:
        cost, node = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        visited_nodes += 1

        if node in hospitals:
            return cost, node, visited_nodes, build_path(previous_nodes, node)

        for neighbor, weight in graph[node]:
            new_cost = cost + weight

            if new_cost < dist_map[neighbor]:
                dist_map[neighbor] = new_cost
                previous_nodes[neighbor] = node
                heapq.heappush(pq, (new_cost, neighbor))

    return float('inf'), None, visited_nodes, []


def heuristic(node):
    return min(distance(node, h) for h in hospitals)


def astar(graph, start, hospitals):
    dist_map = {node: float('inf') for node in graph}
    previous_nodes = {node: None for node in graph}
    dist_map[start] = 0

    pq = [(heuristic(start), 0, start)]
    visited = set()
    visited_nodes = 0

    while pq:
        _, g, node = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        visited_nodes += 1

        if node in hospitals:
            return g, node, visited_nodes, build_path(previous_nodes, node)

        for neighbor, weight in graph[node]:
            new_g = g + weight

            if new_g < dist_map[neighbor]:
                dist_map[neighbor] = new_g
                previous_nodes[neighbor] = node
                f = new_g + heuristic(neighbor)
                heapq.heappush(pq, (f, new_g, neighbor))

    return float('inf'), None, visited_nodes, []


if __name__ == "__main__":
    test_nodes = ['Hauptbahnhof', 'Daberstedt', 'Anger', 'Suedost', 'Stotternheim']

    for start in test_nodes:
        print("===== Start:", start, "=====")

        tracemalloc.start()
        t1 = time.time()
        d = dijkstra(graph, start, hospitals)
        t2 = time.time()
        _, d_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        a1 = time.time()
        a = astar(graph, start, hospitals)
        a2 = time.time()
        _, a_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print("\n--- DIJKSTRA ---")
        print("Hospital:", d[1])
        print("Distance:", round(d[0], 2))
        print("Path:", " -> ".join(d[3]))
        print("Visited:", d[2])
        print("Time:", round(t2 - t1, 6))
        print("Memory:", round(d_mem / 1024, 2), "KB")

        print("\n--- A* ---")
        print("Hospital:", a[1])
        print("Distance:", round(a[0], 2))
        print("Path:", " -> ".join(a[3]))
        print("Visited:", a[2])
        print("Time:", round(a2 - a1, 6))
        print("Memory:", round(a_mem / 1024, 2), "KB")

        print("\n--- CHECK ---")
        same = (d[1] == a[1]) and (round(d[0], 5) == round(a[0], 5))
        print("Same result:", same)
        print()