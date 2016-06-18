from networkx.algorithms import is_directed_acyclic_graph

class AdjustmentException(Exception):
    pass

class AdjustForDirectCauses(object):
    def __init__(self): 
        pass

    def find_predecessors(self, g, causes):
        predecessors = set()
        for cause in causes:
            predecessors = predecessors.union(g.predecessors(cause))
        return predecessors - set(causes)

    def assumptions_satisfied(self, g, causes, effects, predecessors):
        if not is_directed_acyclic_graph(g):
            return False
        if not len(set(effects).intersection(set(causes).union(predecessors))) == 0:
            return False 
        return True 

    def admissable_set(self, g, causes, effects):
        predecessors = self.find_predecessors(g, causes)
        if not self.assumptions_satisfied(g, causes, effects, predecessors):
            raise AdjustmentException("Failed to satisfy adjustment assumptions")
        return predecessors
