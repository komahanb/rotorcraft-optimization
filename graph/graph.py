from graphviz import Digraph

# Create the directed graph with labels and edge colors
dot = Digraph(comment='Rotorcraft Assembly Graph', format='pdf')

# set style elements
dot.attr(rankdir='LR', fontname="Courier")
dot.attr('node', shape='box')

# Node labels (Part numbers and names)
nodes = {1: "blade0", 2: "blade90", 3: "blade180", 4: "blade270",
         5: "shaft", 6: "lsp", 7: "usp", 8: "lpl30", 9: "lpl120",
         10: "lpl210", 11: "lpl300", 12: "upl30", 13: "upl120", 14: "upl210",
         15: "upl300", 16: "sphere", 17: "prod90", 18: "prod180", 19: "prod270",
         20: "lph", 21: "uph", 22: "bsp"}

for i, name in nodes.items():
    dot.node(f"P{i}", f"P{i}: {name}")

# Constraint edges with types and colors
edges = [(1, 12, "rigid link", "black"),
         (2, 13, "rigid link", "black"),
         (3, 14, "rigid link", "black"),
         (4, 15, "rigid link", "black"),

         (12, 5, "revolute", "blue"),
         (13, 5, "revolute", "blue"),
         (14, 5, "revolute", "blue"),
         (15, 5, "revolute", "blue"),

         (8, 7, "spherical", "green"),
         (9, 7, "spherical", "green"),
         (10, 7, "spherical", "green"),
         (11, 7, "spherical", "green"),
         (8, 12, "spherical", "green"),
         (9, 13, "spherical", "green"),
         (10, 14, "spherical", "green"),
         (11, 15, "spherical", "green"),

         (6, 7, "revolute", "blue"),
         (6, 16, "spherical", "green"),
         (16, 5, "cylindrical", "orange"),

         (17, 6, "spherical", "green"),
         (18, 6, "spherical", "green"),
         (19, 6, "spherical", "green"),

         (20, 22, "revolute", "blue"),
         (20, 21, "revolute", "blue"),
         (21, 6, "spherical", "green")]

for src, dst, label, color in edges:
    dot.edge(f"P{src}", f"P{dst}", label=label, color=color)

# Save to file and render
output_path = "/home/komahan/git/rotorcraft-optimization/graph/graph.dot"
pdf_path = output_path.replace(".dot", ".pdf")
dot.save(output_path)
dot.render(filename=output_path.replace(".dot", ""), cleanup=False)
