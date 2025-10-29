import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from graphviz import Digraph
import networkx as nx

class Graph:
    def __init__(self,
                 nodes: dict,
                 edges: list,
                 dof_per_node = 6,
                 dof_per_edge = 6):
        """
        nodes: dict of part_id --> part_name
        edges: list of (src, dst, constraint_type)
        """
        self.nodes = nodes
        self.edges = edges
        self.dof_per_node = dof_per_node
        self.dof_per_edge = dof_per_edge

    def num_nodes(self):
        return len(self.nodes)

    def num_edges(self):
        return len(self.edges)

    def num_edge_dofs(self):
        return len(self.edges) * self.dof_per_edge

    def num_node_dofs(self):
        return len(self.nodes) * self.dof_per_node

    def compute_reachability(self):
        """
        Determine connected components
        Find isolated nodes
        Control in multibody systems
        Causality in simulation
        Identify redundant constraints
        """

        nnodes = self.num_nodes()
        index = {name: i for i, name in enumerate(self.nodes)}
        A = np.zeros((nnodes, nnodes), dtype=int)

        # make edges locally bidirectional for mechanics specific graph diagnostics
        for u, v, ename, ecolor in self.edges:
            A[index[u]][index[v]] = 1
            A[index[v]][index[u]] = 1

        # Warshall's algorithm
        R = A.copy()
        for k in range(nnodes):
            for i in range(nnodes):
                for j in range(nnodes):
                    R[i][j] = R[i][j] or (R[i][k] and R[k][j])
        return R, A

    def verify(self):
        """
        Verify structure connectivity and detect redundant constraints
        based on rank of reachability matrix and singular vector analysis.
        """

        import numpy as np
        from scipy.linalg import svd

        # Build incidence matrix A
        nnodes = len(self.nodes)
        index = {name: i for i, name in enumerate(self.nodes)}
        A = np.zeros((nnodes, nnodes), dtype=int)

        for u, v, ename, ecolor in self.edges:
            A[index[u], index[v]] = 1
            A[index[v], index[u]] = 1  # Bidirectional for mechanical systems

        # Compute reachability matrix R using Warshall's algorithm
        R = A.copy()
        for k in range(nnodes):
            for i in range(nnodes):
                for j in range(nnodes):
                    R[i][j] = R[i][j] or (R[i][k] and R[k][j])

        # Flattened constraint Jacobian as binary matrix
        C = []
        edge_map = []

        for u, v, ename, ecolor in self.edges:
            row = np.zeros(nnodes)
            row[index[u]] = -1
            row[index[v]] = +1
            C.append(row)
            edge_map.append((u, v, ename, ecolor))
        C = np.array(C)

        # Perform SVD
        U, S, Vh = svd(C)
        tol = 1e-10
        rank = np.sum(S > tol)
        nullity = C.shape[0] - rank

        print(f"Reachability matrix R has shape {R.shape}")
        print(f"Constraint matrix C has shape {C.shape}")
        print(f"Rank = {rank}, Nullity = {nullity}")

        if nullity == 0:
            print("✅ No redundant constraints found.")
        else:
            print("⚠️ Redundant constraints detected.")
            # Look at smallest singular vectors for clues
            null_vectors = Vh[-nullity:]
            for i, vec in enumerate(null_vectors):
                significant = np.where(np.abs(vec) > 1e-3)[0]
                print(f"Redundant group {i+1}:")
                for j in significant:
                    u, v, ename, ecolor = edge_map[j]
                    print(f"    ⮕ Edge ({u} ↔ {v}) [{ename}, {ecolor}] with weight ~ {vec[j]:.3f}")

    def plot_ranked_graph(self, filename='ranked_graph.dot', fileformat='pdf', title='Ranked Rotorcraft Graph'):
        """
        Automatically assign ranks (BFS depth levels) to nodes and plot using rank=same subgraphs.
        """
        dot = Digraph(comment=title, format=fileformat)
        dot.attr(rankdir='LR', fontname="Courier") #, ordering="out")
        # dot.attr('node', shape='box')
        dot.attr('node', shape='box', width='2.5', fixedsize='false')

        # Build a directed graph for traversal
        Gnx = nx.DiGraph()
        for src, dst, *_ in self.edges:
            Gnx.add_edge(f"P{src}", f"P{dst}")

        # Default root is the node with smallest ID
        root = f"P{min(self.nodes.keys())}"
        levels = nx.single_source_shortest_path_length(Gnx, root)

        # Add nodes into subgraphs by BFS depth level
        level_to_nodes = {}
        for node, depth in levels.items():
            level_to_nodes.setdefault(depth, []).append(node)

        for depth, group in sorted(level_to_nodes.items()):
            with dot.subgraph() as s:
                s.attr(rank='same')
                for node in group:
                    idx = int(node[1:])
                    s.node(node, f"{node}: {self.nodes[idx]}")

        # Add edges
        for src, dst, edge_label, edge_color in self.edges:
            dot.edge(f"P{src}", f"P{dst}", label=edge_label, color=edge_color)

        dot.save(filename)
        dot.render(filename=filename.replace('.dot', ''), cleanup=False)

    def plot_graph(self, filename = 'graph.dot', fileformat='pdf', title='Rotorcraft Assembly Graph', overlay_dual=True):
        """
        Create the directed graph with labels and edge colors.

        Parameters:
            filename (str)      : Path to output DOT file (should end in .dot)
            format (str)        : Output format, e.g., 'pdf', 'png'
            title (str)         : Title/comment for the graph
            overlay_dual (bool) : If True, overlay transposed graph (adjoint edges) in blue
        """
        dot = Digraph(comment=title, format=fileformat)
        dot.attr(rankdir='Radial', fontname="Courier") #, ordering="out")
        dot.attr('node', shape='box', width='2.5', fixedsize='false')

        # Add nodes
        for inode, node_name in self.nodes.items():
            dot.node(f"P{inode}", f"P{inode}: {node_name}")

        # Add primal edges
        for src_node, dst_node, edge_label, edge_color in self.edges:
            dot.edge(f"P{src_node}", f"P{dst_node}", label=edge_label, color=edge_color)

        # Add adjoint (dual) edges
        if overlay_dual:
            for src_node, dst_node, edge_label, edge_color in self.edges:
                dot.edge(f"P{dst_node}", f"P{src_node}", label=f"{edge_label}ᵗ", color="blue", style="dashed")

        # Save and render
        dot.save(filename)
        dot.render(filename=filename.replace('.dot', ''), cleanup=False)

# Node labels (Part numbers and names)
nodes = {
    0: "blade0", 1: "blade90", 2: "blade180", 3: "blade270",
    4: "shaft", 5: "lsp", 6: "usp", 7: "lpl30", 8: "lpl120",
    9: "lpl210", 10: "lpl300", 11: "upl30", 12: "upl120",
    13: "upl210", 14: "upl300", 15: "sphere", 16: "prod90",
    17: "prod180", 18: "prod270", 19: "lph", 20: "uph", 21: "bsp",
    22: "bcap0", 23: "bcap90", 24: "bcap180", 25: "bcap270"}

# Constraint edges with types and colors
edges = [
    (0, 22, "rigid link", "black"),
    (1, 23, "rigid link", "black"),
    (2, 24, "rigid link", "black"),
    (3, 25, "rigid link", "black"),
    (22, 11, "rigid link", "black"),
    (23, 12, "rigid link", "black"),
    (24, 13, "rigid link", "black"),
    (25, 14, "rigid link", "black"),

    (4, 5, "revolute driver", "red"),
    (5, 6, "revolute", "blue"),
    (6, 7, "spherical", "green"),
    (6, 8, "spherical", "green"),
    (6, 9, "spherical", "green"),
    (6,10, "spherical", "green"),
    (7,11, "spherical", "green"),
    (8,12, "spherical", "green"),
    (9,13, "spherical", "green"),
    (10,14,"spherical", "green"),
    (4,11, "revolute", "blue"),
    (4,12, "revolute", "blue"),
    (4,13, "revolute", "blue"),
    (4,14, "revolute", "blue"),

    (11,22, "rigid link", "black"),
    (12,23, "rigid link", "black"),
    (13,24, "rigid link", "black"),
    (14,25, "rigid link", "black"),

    (5,15, "spherical", "green"),
    (15,15, "cylindrical", "purple"),

    (16,5, "spherical", "green"),
    (17,5, "spherical", "green"),
    (18,5, "spherical", "green"),

    (16,16, "motion driver", "red"),
    (17,17, "motion driver", "red"),
    (18,18, "motion driver", "red"),

    (21,19, "revolute", "blue"),
    (19,20, "revolute", "blue"),
    (20,5,  "spherical", "green")
]

graph = Graph(nodes, edges)
graph.plot_graph()
graph.plot_ranked_graph()
R,A = graph.compute_reachability()

print(R.T @ R, np.shape(R))
print(A)
print(A.sum(axis=0))  # Outgoing
print(A.sum(axis=1))  # Incoming
print(R.sum(axis=1))  # Should be 22 if all connected
print(G.verify())
