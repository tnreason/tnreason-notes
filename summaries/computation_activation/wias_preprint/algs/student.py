from tnreason.engine import create_random_core, contract

studentTensorNetwork = {
    "t0": create_random_core(name="t0", colors=["G", "D", "I"], shape=[6, 3, 2]),
    "t1": create_random_core(name="t1", colors=["L", "G"], shape=[2, 6]),
    "t2": create_random_core(name="t2", colors=["I", "S"], shape=[2, 10]),
}

## Execute the contraction propagation algorithm in the tree-based implementation

from demonstrations.comp_act_nets.algorithms import propagation as cp

propagator = cp.ContractionPropagation(studentTensorNetwork)
propagator.tree_propagation()

## Test on the marginals of the variables "L","G" (core "t1")

testContraction = contract(studentTensorNetwork, openColors=["L", "G"])
propContraction = contract({"mes_t0_t1": propagator.messages["t1"]["t0"],
                                   "t1": studentTensorNetwork["t1"]}, openColors=["L", "G"])

tolerance = 1e-6
for posDict in [{"L": 0, "G": 1}, {"L": 1, "G": 5}]:
    assert abs(testContraction[posDict] - propContraction[posDict]) < tolerance