import networkx as nx

from causality.estimation.adjustments import backDoorAdjustments,AdjustmentException
from tests.unit import TestAPI


class TestBackDoorAdjustments(TestAPI):
    def setUp(self):
        # Graph from Causality (Judea Pearl) first edition pag.80
        # All tests have been checked with DAGity
        self.g = nx.DiGraph()
        self.g.add_nodes_from(['x','y','x1','x2','x3','x4', 'x5','x6'])
        self.g.add_edges_from([('x1','x3'),('x3','x'),('x1','x4'),('x2','x4'),
                               ('x2','x5'),('x5','y'),('x4','x'),('x4','y'),
                               ('x','x6'),('x6','y')])

    def test_backdoor_adjustment_single_cause_single_effect(self):
        # Initialize adjustment class
        adjustment = backDoorAdjustments()

        # Test 1
        causes = ['x']
        effects = ['y']

        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = set({frozenset({'x2', 'x4'}),
                                  frozenset({'x4', 'x5'}),
                                  frozenset({'x3', 'x4'}),
                                  frozenset({'x1', 'x4'})})
        assert(realMinAdmissablesSets==minAdmissablesSets)

        # Test2
        causes = ['x6']
        effects = ['y']

        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = set({frozenset({'x'}),
                                      frozenset({'x2', 'x4'}),
                                      frozenset({'x4', 'x5'}),
                                      frozenset({'x3', 'x4'}),
                                      frozenset({'x1', 'x4'})})
        assert(realMinAdmissablesSets==minAdmissablesSets)

        # Test3
        causes = ['x4']
        effects = ['y']

        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = set({frozenset({'x1', 'x2'}),
                                      frozenset({'x1', 'x5'}),
                                      frozenset({'x2', 'x3'}),
                                      frozenset({'x3', 'x5'})})
        assert(realMinAdmissablesSets==minAdmissablesSets)

        # Test4
        causes = ['x1']
        effects = ['y']

        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = set()
        assert(realMinAdmissablesSets==minAdmissablesSets)

        # Test5
        causes = ['x3']
        effects = ['y']

        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = set({frozenset({'x1'}),
                                      frozenset({'x2', 'x4'}),
                                      frozenset({'x4', 'x5'})})
        assert(realMinAdmissablesSets==minAdmissablesSets)

        # Test6
        causes = ['x5']
        effects = ['y']

        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = set({frozenset({'x2'}),
                                      frozenset({'x1', 'x4'}),
                                      frozenset({'x', 'x4'}),
                                      frozenset({'x3', 'x4'}),
                                      frozenset({'x4', 'x6'})})
        assert(realMinAdmissablesSets==minAdmissablesSets)


    def test_backdoor_adjustment_multiple_causes_single_effect(self):
        # Initialize adjustment class
        adjustment = backDoorAdjustments()

        # Test 1
        causes = ['x','x5']
        effects = ['y']

        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = set({frozenset({'x4'})})
        assert(realMinAdmissablesSets==minAdmissablesSets)

        # Test 2
        causes = ['x','x4']
        effects = ['y']

        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = set({frozenset({'x2'}),
                                      frozenset({'x5'})})
        assert(realMinAdmissablesSets==minAdmissablesSets)

        # Test 3
        causes = ['x1','x2']
        effects = ['y']

        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = set()
        assert(realMinAdmissablesSets==minAdmissablesSets)

        # Test 4
        causes = ['x','x2']
        effects = ['y']

        # This test should raise an AdjustmentException since there are not admissable sets
        with self.assertRaises(AdjustmentException):
            set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])


    def test_backdoor_adjustment_multiple_causes_multiple_effects(self):
        # Initialize adjustment class
        adjustment = backDoorAdjustments()

        # Test 1
        causes = ['x','x3']
        effects = ['y','x5']

        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = set({frozenset({'x1', 'x4'}),
                                      frozenset({'x2', 'x4'})})
        assert(realMinAdmissablesSets==minAdmissablesSets)

        # Test 2
        causes = ['x','x1']
        effects = ['y','x5']

        # This test should raise an AdjustmentException since there are not admissable sets
        with self.assertRaises(AdjustmentException):
            set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        # Test 3
        causes = ['x1','x3']
        effects = ['y','x6']

        minAdmissablesSets = set([ s for s in adjustment.minimal_backdoor_admissable_sets(self.g,causes,effects)])

        realMinAdmissablesSets = set()
        assert(realMinAdmissablesSets==minAdmissablesSets)
