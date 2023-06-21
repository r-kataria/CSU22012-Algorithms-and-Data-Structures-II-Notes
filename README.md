# CSU22012 - Algorithms and Data Structures II Notes

## 📚 Overview

These are the notes I made for CSU22012 – Algorithms and Data Structures II at Trinity College Dublin (2022-2023). They cover the majority of the course syllabus, including a wide range of algorithms, data structures, and design paradigms discussed throughout the term. These notes are made to help in exam preparation and provide a simple way to understand key concepts in the module.

## Topic 1: Sorting algorithms

Here are sorting algorithms we covered. 🚀

### 1.1 Insertion sort

- 🧐 Logic: Iterate through the list, insert each element into its correct position in the sorted section.
- ⏱️ Best case: O(n)
- ⏱️ Worst case: O(n^2)
- ✅ Stable: Yes
- ✅ In-place: Yes
- 💪 Algorithm type: Incremental
- 💻 Pseudocode:

```
      for i in range(1, len(arr)):
    	key = arr[i]
    	j = i - 1

    	while j >= 0 and key < arr[j]:
    		arr[j+1] = arr[j]
    		j -= 1
    		arr[j+1] = key

```

### 1.2 Bubble sort

- 🧐 Logic: Repeatedly swap adjacent elements if they are in the wrong order.
- ⏱️ Best case: O(n)
- ⏱️ Worst case: O(n^2)
- ✅ Stable: Yes
- ✅ In-place: Yes
- 💪 Algorithm type: Incremental
- 💻 Pseudocode:

```
    for i in range(len(arr)-1):
	    for j in range(len(arr)-i-1):
    	    if arr[j] > arr[j+1]:
    		    arr[j], arr[j+1] = arr[j+1], arr[j]

```

### 1.3 Selection sort

- 🧐 Logic: Select the smallest element in the unsorted section and swap it with the first unsorted element.
- ⏱️ Best case: O(n^2)
- ⏱️ Worst case: O(n^2)
- ❌ Stable: No
- ✅ In-place: Yes
- 💪 Algorithm type: Incremental
- 💻 Pseudocode:

```
for i in range(len(arr)):
	min_idx = i for j in range(i+1, len(arr)):
		if arr[j] < arr[min_idx]:
			min_idx = j arr[i], arr[min_idx] = arr[min_idx], arr[i]

```

### 1.4 Merge sort

- 🧐 Logic: Divide the list into halves, recursively sort the halves, and merge them.
- ⏱️ Best case: O(n * log(n))
- ⏱️ Worst case: O(n * log(n))
- ✅ Stable: Yes
- ❌ In-place: No (requires additional memory)
- 💪 Algorithm type: Divide and conquer
- 💻 Pseudocode:

```
def merge_sort(arr):
	if len(arr) <= 1: return arr

	mid = len(arr) // 2

	left = merge_sort(arr[:mid])
	right = merge_sort(arr[mid:])

	return merge(left, right)

def merge(left, right):
	result = []
	i, j = 0

	while i < len(left) and j < len(right):
		if left[i] < right[j]:
			result.append(left[i])
			i += 1
		else:
			result.append(right[j])
			j += 1

	result += left[i:]
	result += right[j:]

	return result

```

### 1.5 Quick sort

- 🧐 Logic: Select a pivot, partition the list around the pivot, and recursively sort the partitions.
- ⏱️ Best case: O(n * log(n))
- ⏱️ Worst case: O(n^2) (rarely occurs with bad pivot selection)
- ✅ Stable: No
- ✅ In-place: Yes
- 💪 Algorithm type: Divide and conquer
- 💻 Pseudocode:

```
def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return i+1

```

### 1.6 Shell sort

- 🧐 Logic: Sort elements at specific intervals, reducing the interval until it's 1 (Insertion sort).
- ⏱️ Best case: O(n * log(n))
- ⏱️ Worst case: O(n^(3/2)) (depends on the gap sequence)
- ❌ Stable: No
- ✅ In-place: Yes
- 💪 Algorithm type: Incremental
- 💻 Pseudocode:

```
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2

```

## Topic 2: Graphs

- 🤔 A graph is a collection of nodes (vertices) connected by edges.
- 👨‍🎓 **Pro Tip:** Refer to the graph notes from Discrete Maths, specially for MSTs.

### 2.1 Using Graphs

#### 2.1.1 Graph ADT

- 🧑‍💻 Abstract Data Type representing a graph, typically including methods for adding vertices, edges, and querying neighbours.

#### 2.1.2 Array-based Representations

- 📊 Two common ways to represent a graph using arrays.

#### 2.1.2.1 Adjacency Matrix

- 2D array with rows/columns representing vertices and elements indicating edge presence/weight.
- 💻 Pseudocode for checking if there is an edge between two vertices:

```
if adjacency_matrix[v1][v2] != 0:
    ## there is an edge between v1 and v2

```

#### 2.1.2.2 Adjacency List

- Array of lists, each list representing a vertex's neighbors.
- 💻 Pseudocode for checking if there is an edge between two vertices:

```
if v2 in adjacency_list[v1]:
    ## there is an edge between v1 and v2

```

### 2.2 Traversal Algorithms

#### 2.2.1 DFS (Depth-First Search)

- Algorithm for traversing or searching graphs, visiting children of a node before visiting its siblings.
- ⏱️ Time complexity: O(V+E)
- 💾 Space complexity: O(V)
- 💻 Pseudocode:

```
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

```

- Applications: Pathfinding, Connected components, Topological sorting

#### 2.2.2 BFS (Breadth-First Search)

- Algorithm for traversing or searching graphs, visiting all neighbors of a node before visiting their children.
- ⏱️ Time complexity: O(V+E)
- 💾 Space complexity: O(V)
- 💻 Pseudocode:

```
def bfs(graph, start):
    visited = set()
    queue = [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
    return visited

```

- Applications: Shortest path, Connected components

### 2.3 Shortest Path Algorithms

#### 2.3.1 Dijkstra's Algorithm

- Finds the shortest path from a single source to all other vertices in a weighted graph with non-negative weights.
- ⏱️ Time complexity: O(V^2) or O(V+E*log(V)) with priority queue
- 💰 Greedy algorithm
- ❌ Does not work with negative edge weights
- 💻 Pseudocode:

```
def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        (cost, node) = heapq.heappop(pq)
        if cost > dist[node]:
            continue
        for neighbor, weight in graph[node].items():
            alt = cost + weight
            if alt < dist[neighbor]:
                dist[neighbor] = alt
                heapq.heappush(pq, (alt, neighbor))
    return dist

```

#### 2.3.2 Bellman-Ford Algorithm

- Finds the shortest path from a single source to all other vertices in a weighted graph, even with negative weights.
- ⏱️ Time complexity: O(V*E)
- ❗ Detects negative-weight cycles
- 💻 Pseudocode:

```
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    for i in range(len(graph) - 1):
        for u, edges in graph.items():
            for v, weight in edges.items():
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
    for u, edges in graph.items():
        for v, weight in edges.items():
            if dist[u] + weight < dist[v]:
                raise ValueError("Negative weight cycle detected")
    return dist

```

### 2.4 Topological Sort

- Linear ordering of vertices in a directed acyclic graph (DAG) such that for every directed edge (u, v), vertex u comes before vertex v.
- ⏱️ Time complexity: O(V+E)
- 🦄 Can produce different outputs for the same graph
- 💻 Pseudocode:

```
def topological_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = [node for node in in_degree if in_degree[node] == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(graph):
        raise ValueError("Graph has a cycle")
    return result

```

### 2.5 Minimum Spanning Tree Algorithms

#### 2.5.1 Prim's Algorithm

- Finds the minimum spanning tree of an undirected, connected, weighted graph.
- ⏱️ Time complexity: O(V^2) or O(E*log(V)) with priority queue
- 💰 Greedy algorithm
- 💻 Pseudocode:

```
    mst = set()
    visited = set()
    start_node = next(iter(graph))
    visited.add(start_node)
    edges = [
        (cost, start_node, to)
        for to, cost in graph[start_node].items()
    ]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.add((frm, to, cost))
            for to_next, cost in graph[to].items():
                if to_next not in visited:
                    heapq.heappush(edges, (cost, to, to_next))

    return mst

```

#### 2.5.2 Kruskal's Algorithm

- Finds the minimum spanning tree of an undirected, connected, weighted graph.
- ⏱️ Time complexity: O(E*log(V))
- 💰 Greedy algorithm
- 💻 Pseudocode:

```
def kruskal(graph):
    mst = set()
    edges = [
        (cost, frm, to)
        for frm in graph
        for to, cost in graph[frm].items()
    ]
    edges.sort()

    parent = {node: node for node in graph}

    def find(node):
        if parent

```

### 2.6 All Pairs Shortest Path

#### 2.6.1 Floyd-Warshall Algorithm

- Finds the shortest paths between all pairs of vertices in a weighted graph with or without negative edge weights (but no negative cycles).
- ⏱️ Time complexity: O(V^3)
- 💻 Pseudocode:

```
def floyd_warshall(graph):
    dist = {}
    for i in graph:
        dist[i] = {}
        for j in graph:
            dist[i][j] = float("inf")

    for node in graph:
        dist[node][node] = 0
        for neighbor, weight in graph[node].items():
            dist[node][neighbor] = weight

    for k in graph:
        for i in graph:
            for j in graph:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist

```

### 2.7 Single-Pair Shortest Path

#### 2.7.1 Uniform Cost Search

- A search algorithm that expands the node with the lowest path cost.
- ⏱️ Time complexity: O(b^d) (b: branching factor, d: depth of the solution)
- 💾 Space complexity: O(b^d)
- Applications: Pathfinding, AI planning
- 💻 Pseudocode:

```
def uniform_cost_search(graph, start, goal):
    queue = [(0, start)]
    visited = set()
    while queue:
        (cost, node) = heapq.heappop(queue)
        if node == goal:
            return cost
        if node not in visited:
            visited.add(node)
            for neighbor, weight in graph[node].items():
                heapq.heappush(queue, (cost + weight, neighbor))
    return None

```

#### 2.7.2 Greedy Best-First Search

- A search algorithm that expands the node with the lowest heuristic value.
- ⏱️ Time complexity: O(b^m) (b: branching factor, m: maximum depth of the search space)
- 💾 Space complexity: O(b^m)
- Applications: Pathfinding, AI planning, constraint satisfaction problems
- 💻 Pseudocode:

```
def greedy_best_first_search(graph, start, goal, heuristic):
    queue = [(heuristic(start, goal), start)]
    visited = set()
    while queue:
        (cost, node) = heapq.heappop(queue)
        if node == goal:
            return cost
        if node not in visited:
            visited.add(node)
            for neighbor, weight in graph[node].items():
                heapq.heappush(queue, (heuristic(neighbor, goal), neighbor))
    return None

```

‼️ **NB** that `heuristic` is a function that estimates the cost from a given node to the goal node. The idea is to use this heuristic to guide the search towards the goal node.

#### 2.7.3 A* Search

- Finds the shortest path between two nodes in a graph, combines the cost function of the Uniform Cost and the heuristic function from the Greedy Best-First to guide the search.
- ⏱️ Time complexity: O(V+E*log(V)) (depends on the heuristic)

## Topic 3: Recursion

### 3.1 What is recursion

- 🤔 A process in which a function calls itself as a subroutine to solve a problem.
- 💫 In a recursive function, the solution to the base case is provided, and the problem is divided into smaller subproblems.
- 😎 The base case is the simplest possible case that can be solved directly.

### 3.2 Pros and cons of recursion

- 👍 Recursion can simplify code and make it easier to understand.
- 👍 Recursion is often more elegant and concise than iterative solutions.
- 👎 Recursion can be less efficient than iterative solutions, especially for large problems.
- 👎 Recursion can be harder to debug and understand for complex problems.
- 👎 Recursion can cause infinite loops if the base case isn't properly defined.

### 3.3 Tail Recursion

- 🚀 A form of recursion where the recursive call is the last operation performed in the function.
- 📊 The recursive call is optimized by the compiler, which turns it into a loop.
- 💡 Why Tail Recursion?
    - Tail recursion is usually more efficient (although more difficult to write) than non-tail recursion.
    - The recursive calls do not need to be added to the call stack: there is only one, the current call, in the stack.
    - It is possible to turn tail recursions into iterative algorithms.

#### 3.3.1 Example of Tail Recursion

- 🧑‍💻 Here's an example of a tail-recursive function that calculates the factorial of a given number:

```
function factorial(n, acc = 1) {
  if (n === 0) return acc;
  return factorial(n - 1, acc * n);
}

```

- 📝 The `factorial` function takes two arguments: `n` is the number whose factorial is being calculated, and `acc` is the accumulator that keeps track of the product of the numbers seen so far.
- 🛠️ The function uses a tail-recursive call to calculate the factorial of `n - 1`, passing `acc * n` as the new accumulator.
- 🚀 When `n` reaches 0, the function returns the accumulator, which contains the factorial of the original number.
- 📊 The function uses tail recursion and is optimized by the compiler, which turns it into a loop.

## Topic 4: Algorithm design

### 4.1 Brute-force/exhaustive search

- 🤯 Systematically enumerating all possible candidates for the solution and checking whether each candidate satisfies the problem's statement
- Often simplest to implement but not very efficient
- Impractical for all but smallest instances of a problem
- 🧐 Examples:
    - Selection sort
    - Bubble sort
    - In graphs – depth-first search (DFS), breadth-first search (BFS)

### 4.2 Decrease and conquer

- 🎯 Establish relationship between a problem and a smaller instance of that problem
- Exploit that relationship top down or bottom up to solve the bigger problem
- Naturally implemented using recursion
- 🧐 Examples:
    - Insertion sort
    - In graphs – topological sorting

### 4.3 Divide and conquer

- 🔪 Divide a problem into several subproblems of the same type, ideally of the same size
- Solve subproblems, typically recursively
- If needed, combine solutions
- 🧐 Examples:
    - Mergesort
    - Quicksort
    - Binary tree traversal – preorder, inorder, postorder
        - Visit root, its left subtree, and its right subtree

### 4.4 Transform and conquer

- 🌀 Modify a problem to be more amenable to solution, then solve
    - Transform to a simpler/more convenient instance of the same problem – instance simplification
    - Transform to a different representation of the same instance – representation change
    - Transform to an instance of a different problem for which an algorithm is available – problem reduction
- 🧐 Examples:
    - Balanced search trees – AVL trees, 2-3 trees – Reduction to graph problems

### 4.5 Dynamic programming

- 🤖 Similar to divide and conquer, solves problems by combining the solutions to subproblems
    - In divide and conquer subproblems are disjoint
    - In dynamic programming, subproblems overlap, i.e., share subsubproblems
- Solutions to those are stored, indexed, and reused
- 🧐 Examples:
    - Floyd-Warshall shortest path algorithm

### 4.6 Greedy

- 😈 Always make the choice that looks best at the moment
    - Does not always yield the most optimal solution, but often does
- 🧐 Examples
    - Graphs:
        - Dijkstra – find the shortest path from the source to the vertex nearest to it, then second nearest, etc.
        - Prim
        - Kruskal
    - Hill Climbing

### 4.7 Genetic Algorithms

- 🧬 Inspired by the process of natural selection and evolution
- 🌐 Population-based optimization technique using stochastic methods
- 👨‍👩‍👧‍👦 Represents solutions as individuals in a population
- 🔄 Iteratively improves the population through selection, crossover (recombination), and mutation
- 🎯 Typically applied to optimization and search problems
- ⏲️ May not always find the optimal solution, but can often find near-optimal solutions in a reasonable time
- 🧐 Examples:
    - Traveling Salesman Problem
    - Machine learning model hyperparameter optimization
    - Feature selection in classification problems

### 4.8 **Constraint Programming**

- ❗️ Formulates a problem as a set of variables, domains, and constraints
- 🎯 Focuses on finding a solution that satisfies all constraints while optimizing an objective function (if any)
- 📐 Variables have specific domains (ranges of possible values)
- 🔗 Constraints define the relationships between variables and restrict the feasible solution space
- 🧠 Often relies on backtracking, search, and propagation techniques to find feasible solutions
- 🔧 Highly expressive and flexible for solving a wide variety of combinatorial problems
- 🧐 Examples:
    - Scheduling problems
    - Resource allocation
    - Puzzles (e.g., Sudoku, N-queens problem)
    - Graph coloring

## Topic 5: Misc

Stuff that doesn't really fit in anywhere else but might be important.

### 5.1 Strings

- 📚 Sequences of characters: Text, Genome sequences
- 🧩 Characters: In C (char), In Java (char, 16-bit unsigned int)
- 🚀 Java.lang.String: Immutable sequence of characters.
- 🔒 Security: Parameters in many methods which could introduce vulnerability - security threats, eg network connection is passed a string - it could be modified to connect to a different machine, or a modified file name can be passed in etc
- 🧵 Thread-safe: No need for synchronization if shared between threads - no thread can modify it (So no Concurrent Systems Logic Needed 😄)

#### 5.1.1 String Sorting Algorithms

- 🧐 When to use which?
    - Insertion: Small arrays, nearly sorted
    - Quick: General purpose, tight space
    - Merge: General purpose, stable
    - 3-way quick: Large number of equal keys
    - LSD: Short fixed-length strings
    - MSD: Random strings
    - 3-way string quicksort: General purpose, long prefix matches


## Important Notice

💥 **THESE NOTES ARE INCOMPLETE**. They are currently missing **`LSD`, `MSD`, `Key-indexed counting`, and `anything to do with tries`**.

✨ In addition to everything here, you should also go through the project requirements, since it is explicitly stated that questions regarding the algorithms discussed for the projects may be asked.

🔧 If anyone else wants to contribute and finish these up, please make a pull request. I will not be working on these anymore since I've already finished the module.
