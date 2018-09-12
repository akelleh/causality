import networkx as nx

from causality.estimation.adjustments import backDoorAdjustments
from tests.unit import TestAPI


class TestBackDoorAdjustments(TestAPI):
    def setUp(self):
        # Graph from Causality (Judea Pearl) first edition pag.80
        self.g = nx.DiGraph()
        self.g.add_nodes_from(['x','y','x1','x2','x3','x4', 'x5','x6'])
        self.g.add_edges_from([('x1','x3'),('x3','x'),('x1','x4'),('x2','x4'),
                               ('x2','x5'),('x5','y'),('x4','x'),('x4','y'),
                               ('x','x6'),('x6','y')])

    def test_backdoor_adjustment_single_cause_single_effect(self):
        causes = ['x']
        effects = ['y']

        adjustment = backDoorAdjustments()
        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = {frozenset({'x2', 'x4'}),
                                  frozenset({'x4', 'x5'}),
                                  frozenset({'x3', 'x4'}),
                                  frozenset({'x1', 'x4'})}
        assert(realMinAdmissablesSets==minAdmissablesSets)

    def test_backdoor_adjustment_multiple_causes_single_effect(self):
        causes = ['x','x5']
        effects = ['y']

        adjustment = backDoorAdjustments()
        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = {frozenset({'x4'})}
        assert(realMinAdmissablesSets==minAdmissablesSets)
