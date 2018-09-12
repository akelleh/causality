from networkx.algorithms import is_directed_acyclic_graph
import networkx as nx
from pgmpy.models import BayesianModel
from itertools import combinations

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

class backDoorAdjustments(object):
    def __init__(self,):
        pass

    def assumptions_satisfied(self, g, causes, effects):
        if not is_directed_acyclic_graph(g):
            raise AdjustmentException("Suplied Graph is not Directed and Acyclic")
        if (len(causes)==0 or len(effects)==0):
            raise AdjustmentException("Causes/Effects can not be empty")

    def __are_causes_dseparated_from_effects(self, g, s, causes, effects):
        # Internal function to exit double loop
        def is_cause_dseparated_from_effects(g, s, cause, effects):
            for effect in effects:
                if g.is_active_trail(cause, effect, observed=s):
                    return(False)
            return(True)

        causesDSeparatedFromEffectsInGraph = True
        for cause in causes:
            if not is_cause_dseparated_from_effects(g,s,cause,effects):
                causesDSeparatedFromEffectsInGraph = False
                break
        return(causesDSeparatedFromEffectsInGraph)


    def minimal_backdoor_admissable_sets(self, g, causes, effects):

        def is_superset_of_any_set_of_sets(s, setOfSets):
            isSubset = False
            for i in setOfSets:
                if set(s).issuperset(i):
                    isSubset = True
                    break
            return(isSubset)

        # Check arguments
        self.assumptions_satisfied(g, causes, effects)

        # Bayesian Network is a DiGraph wrapper from pgmpy
        # used because of its d-separation function (is_active_trail)
        backDoorGraph = BayesianModel(nx.edges(g))
        descendantsOfCauses = set()

        # Create back door graph and collect descendants from causes
        for cause in causes:
            outEdgesOfCause = backDoorGraph.out_edges(cause)
            descendantsOfCauses = descendantsOfCauses.union(nx.descendants(backDoorGraph,cause))
            backDoorGraph.remove_edges_from(outEdgesOfCause)

        # Possible adjustment nodes are those from the original graph that:
        # i) Are not causes
        # ii) Are not effects
        # iii) Are not descendants of the causes
        possibleAdjustmentNodes = set(backDoorGraph.nodes()).difference(set(causes),
                                                                        set(effects),
                                                                        set(descendantsOfCauses))
        # Initialize outcome variable which will be a set of sets
        minAdmissablesSets = set()

        # If the empty set d-separates causes and effects in the back door graph
        # then return the empty set
        if self.__are_causes_dseparated_from_effects(backDoorGraph, set(), causes, effects):
            return(set())

        # Check all set partitions of possibleAdjustmentNodes
        for r in range(len(possibleAdjustmentNodes)):
            for s in combinations(possibleAdjustmentNodes,r+1):
                # Check s only if s is not a super set of any set already in minAdmissablesSets
                if not is_superset_of_any_set_of_sets(s,minAdmissablesSets):
                    # Only add set to minAdmissablesSets if all causes are d-Separated of causes
                    if self.__are_causes_dseparated_from_effects(backDoorGraph, s, causes, effects):
                        minAdmissablesSets.add(frozenset(s))

        # If after checking all combinations we don't find any admissable set then raise an Exception
        if len(minAdmissablesSets)==0:
            raise AdjustmentException("Failed to satisfy adjustment assumptions")

        return(minAdmissablesSets)
