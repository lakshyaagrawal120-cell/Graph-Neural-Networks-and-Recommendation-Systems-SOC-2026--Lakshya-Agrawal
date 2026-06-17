class CustomGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id, features=None):
        self.nodes[node_id] = features or []

    def add_edge(self, source, target, attributes=None):
        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append({'target': target, 'attributes': attributes or []})

    def get_adjacency_list(self):
        return self.edges

g = CustomGraph()
g.add_node(0, [1.0, 0.5])
g.add_node(1, [0.2, 0.8])
g.add_edge(0, 1, [0.5])

print(g.nodes)
print(g.edges)