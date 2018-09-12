import networkx as nx

# All this file has the functions that are needed to be added to
# Causality from pgmpy in order to check active trail in the sense
# of d-connectness.
# It only has minimum changes in order to make it work as a standalone
# function.
# The function making all heavy work is active_trail_nodes

class utils(object):
    def __init__(self):
        pass

    def _get_ancestors_of(self, g, obs_nodes_list):
        """
        Returns a dictionary of all ancestors of all the observed nodes including the
        node itself.
        Parameters
        ----------
        obs_nodes_list: string, list-type
            name of all the observed nodes
        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'),
        ...                        ('I', 'L')])
        >>> model._get_ancestors_of('G')
        {'D', 'G', 'I'}
        >>> model._get_ancestors_of(['G', 'I'])
        {'D', 'G', 'I'}
        """
        if not isinstance(obs_nodes_list, (list, tuple)):
            obs_nodes_list = [obs_nodes_list]

        for node in obs_nodes_list:
            if node not in g.nodes():
                raise ValueError('Node {s} not in not in graph'.format(s=node))

        ancestors_list = set()
        nodes_list = set(obs_nodes_list)
        while nodes_list:
            node = nodes_list.pop()
            if node not in ancestors_list:
                nodes_list.update(g.predecessors(node))
            ancestors_list.add(node)
        return ancestors_list

    def active_trail_nodes(self, g, variables, observed=None):
        """
        Returns a dictionary with the given variables as keys and all the nodes reachable
        from that respective variable as values.
        Parameters
        ----------
        variables: str or array like
            variables whose active trails are to be found.
        observed : List of nodes (optional)
            If given the active trails would be computed assuming these nodes to be observed.
        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> student = BayesianModel()
        >>> student.add_nodes_from(['diff', 'intel', 'grades'])
        >>> student.add_edges_from([('diff', 'grades'), ('intel', 'grades')])
        >>> student.active_trail_nodes('diff')
        {'diff': {'diff', 'grades'}}
        >>> student.active_trail_nodes(['diff', 'intel'], observed='grades')
        {'diff': {'diff', 'intel'}, 'intel': {'diff', 'intel'}}
        References
        ----------
        Details of the algorithm can be found in 'Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman
        Page 75 Algorithm 3.1
        """
        if observed:
            observed_list = observed if isinstance(observed, (list, tuple)) else [observed]
        else:
            observed_list = []
        ancestors_list = self._get_ancestors_of(g, observed_list)

        # Direction of flow of information
        # up ->  from parent to child
        # down -> from child to parent

        active_trails = {}
        for start in variables if isinstance(variables, (list, tuple)) else [variables]:
            visit_list = set()
            visit_list.add((start, 'up'))
            traversed_list = set()
            active_nodes = set()
            while visit_list:
                node, direction = visit_list.pop()
                if (node, direction) not in traversed_list:
                    if node not in observed_list:
                        active_nodes.add(node)
                    traversed_list.add((node, direction))
                    if direction == 'up' and node not in observed_list:
                        for parent in g.predecessors(node):
                            visit_list.add((parent, 'up'))
                        for child in g.successors(node):
                            visit_list.add((child, 'down'))
                    elif direction == 'down':
                        if node not in observed_list:
                            for child in g.successors(node):
                                visit_list.add((child, 'down'))
                        if node in ancestors_list:
                            for parent in g.predecessors(node):
                                visit_list.add((parent, 'up'))
            active_trails[start] = active_nodes
        return active_trails

    def is_active_trail(self, g, start, end, observed=None):
        """
        Returns True if there is any active trail between start and end node
        Parameters
        ----------
        start : Graph Node
        end : Graph Node
        observed : List of nodes (optional)
            If given the active trail would be computed assuming these nodes to be observed.
        additional_observed : List of nodes (optional)
            If given the active trail would be computed assuming these nodes to be observed along with
            the nodes marked as observed in the model.
        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> student = BayesianModel()
        >>> student.add_nodes_from(['diff', 'intel', 'grades', 'letter', 'sat'])
        >>> student.add_edges_from([('diff', 'grades'), ('intel', 'grades'), ('grades', 'letter'),
        ...                         ('intel', 'sat')])
        >>> student.is_active_trail('diff', 'intel')
        False
        >>> student.is_active_trail('grades', 'sat')
        True
        """
        if end in self.active_trail_nodes(g, start, observed)[start]:
            return True
        else:
            return False
