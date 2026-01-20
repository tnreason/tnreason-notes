from tnreason.engine import contract
from tnreason.engine import create_from_slice_iterator as create
import numpy as np


def create_sudoku_rule_tensor_network(n):
    """
    Creates a tensor network of n^2 \tau^k matrices to each Sudoku constraint
    """
    rulesSpecDict = {
        ## Column Constraints
        **{f"I_:_:_{c0}_{c1}_{i}": [f"X_{r0}_{r1}_{c0}_{c1}_{i}" for r0 in range(n) for r1 in
                                    range(n)] for c0 in range(n) for c1 in range(n)
           for i in range(n ** 2)},
        ## Row Constraints
        **{f"I_{r0}_{r1}_:_:_{i}": [f"X_{r0}_{r1}_{c0}_{c1}_{i}" for c0 in range(n) for c1 in
                                    range(n)] for r0 in range(n) for r1 in range(n)
           for i in range(n ** 2)},
        ## Squares Constraints
        **{f"I_{r0}_:_{c0}_:_{i}": [f"X_{r0}_{r1}_{c0}_{c1}_{i}" for r1 in range(n) for c1 in
                                    range(n)] for r0 in range(n) for c0 in range(n)
           for i in range(n ** 2)},
        ## Position Constraints
        **{f"I_{r0}_{r1}_{c0}_{c1}_:": [f"X_{r0}_{r1}_{c0}_{c1}_{i}" for i in range(n ** 2)]
           for r0 in range(n) for r1 in range(n) for c0 in range(n) for c1 in range(n)}
    }
    cores = {}
    for decomKey in rulesSpecDict:
        cores.update({
            decomKey + "_" + atomVar: create(
                shape=[2, len(rulesSpecDict[decomKey])],
                colors=[atomVar, decomKey],
                sliceIterator=[(1, {atomVar: 0}),
                               (-1, {atomVar: 0, decomKey: i}),
                               (1, {atomVar: 1, decomKey: i})])
            for i, atomVar in enumerate(rulesSpecDict[decomKey])
        })
    return cores


def encode_trivial_extended_evidence(E, n):
    """
    Prepares e_1 basis vectors for known variables and trivial vectors for others
    """
    return {**{f"{r0}_{r1}_{c0}_{c1}_{i}_eC":
                   create(shape=[2], colors=[f"X_{r0}_{r1}_{c0}_{c1}_{i}"],
                          sliceIterator=[(1, {f"X_{r0}_{r1}_{c0}_{c1}_{i}": 1})])
               for r0, r1, c0, c1, i in E},
            **{f"{r0}_{r1}_{c0}_{c1}_{i}_eC":
                   create(shape=[2], colors=[f"X_{r0}_{r1}_{c0}_{c1}_{i}"],
                          sliceIterator=[(1, {})])
               for r0 in range(n) for r1 in range(n) for c0 in range(n)
               for c1 in range(n) for i in range(n ** 2) if (r0, r1, c0, c1, i) not in E}}


def extract_resulting_evidence(propagator, n):
    """
    Returns the evidence given a ContractionPropagation instance
    """
    return [(r0, r1, c0, c1, i) for r0 in range(n) for r1 in range(n)
            for c0 in range(n) for c1 in range(n) for i in range(n ** 2)
            if contract({
            "eC": propagator.cores[f"{r0}_{r1}_{c0}_{c1}_{i}_eC"],
            **propagator.messages[f"{r0}_{r1}_{c0}_{c1}_{i}_eC"]},
            openColors=[f"X_{r0}_{r1}_{c0}_{c1}_{i}"])[{f"X_{r0}_{r1}_{c0}_{c1}_{i}": 0}] == 0]


def tuples_to_array(evidence, n=2):
    """
    Arranges the variables in an array
    """
    array = np.zeros(shape=(n ** 2, n ** 2))
    for (r0, r1, c0, c1, i) in evidence:
        array[r0 * n + r1, c0 * n + c1] = i + 1
    return array


from demonstrations.comp_act_nets.algorithms import propagation as cp

n = 2
evidence = [(0, 0, 0, 0, 0), (0, 0, 1, 0, 2), (0, 0, 1, 1, 1),
            (0, 1, 0, 1, 1), (1, 0, 1, 0, 3), (1, 1, 0, 0, 3),
            (1, 1, 0, 1, 2)]
propagator = cp.ContractionPropagation(
    cores={**create_sudoku_rule_tensor_network(n=n),
           **encode_trivial_extended_evidence(evidence, n=n)})
propagator.constraint_propagation([f"{r0}_{r1}_{c0}_{c1}_{i}_eC" for (r0, r1, c0, c1, i) in evidence])
solutionArray = tuples_to_array(extract_resulting_evidence(propagator, n=2))
assert np.all(solutionArray == np.array([[1, 4, 3, 2], [3, 2, 1, 4], [2, 1, 4, 3], [4, 3, 2, 1]]))