from tnreason import engine
import math

from copy import deepcopy


def get_sum_tn(m, d):
    return {"b_0": engine.create_from_slice_iterator(
        shape=[m, 2, m, m],
        colors=[f"Y_{0}", f"Z_{0}", f"X_{0}", f"TX_{0}"],
        sliceIterator=[(1, {f"Y_{0}": (x + tx) % m, f"Z_{0}": math.floor((x + tx) / m),
                            f"X_{0}": x, f"TX_{0}": tx}) for x in range(m) for tx in range(m)]),
        **{f"middleBlock{k}": engine.create_from_slice_iterator(
            shape=[m, 2, m, m, 2],
            colors=[f"Y_{k}", f"Z_{k}", f"X_{k}", f"TX_{k}", f"Z_{k - 1}"],
            sliceIterator=[
                (1, {f"Y_{k}": (x + tx + z0) % m,
                     f"Z_{k}": math.floor((x + tx + z0) / m),
                     f"X_{k}": x, f"TX_{k}": tx, f"fZ_{k - 1}": z0}) for x
                in range(m) for tx in range(m) for z0 in range(2)]
        ) for k in range(1, d - 1)},
        **{f"b_{d - 1}": engine.create_from_slice_iterator(
            shape=[m, 2, m, m, 2],
            colors=[f"Y_{d - 1}", f"Y_{d}", f"X_{d - 1}", f"TX_{d - 1}", f"Z_{d - 2}"],
            sliceIterator=[
                (1, {f"Y_{d - 1}": (x + tx + z0) % m, f"Y_{d}": math.floor((x + tx + z0) / m),
                     f"X_{d - 1}": x, f"TX_{d - 1}": tx, f"Z_{d - 2}": z0}) for x
                in range(m) for tx in range(m) for z0 in range(2)]
        )}}


def encode_digits(num0, num1, m):
    return {**{f"X_{len(num0) - 1 - i}_eC": engine.create_from_slice_iterator(shape=[m], colors=[
        f"X_{len(num0) - 1 - i}"], sliceIterator=[(1, {f"X_{len(num0) - 1 - i}": int(digit)})]) for
               i, digit in enumerate(num0)},
            **{f"TX_{len(num1) - 1 - i}_eC": engine.create_from_slice_iterator(shape=[m], colors=[
                f"TX_{len(num1) - 1 - i}"], sliceIterator=[
                (1, {f"TX_{len(num0) - 1 - i}": int(digit)})]) for i, digit in
               enumerate(num1)}}


assert 1 == encode_digits("0001", "0000", 10)["X_0_eC"][{"X_0": 1}]
assert 0 == encode_digits("0001", "0000", 10)["X_0_eC"][{"X_0": 0}]

## Example: 08+12=020 in basis 10
m = 10
catorder = 2
assert 1 == int(engine.contract(coreDict={**get_sum_tn(m, catorder), **encode_digits("08", "12", m)},
                                openColors=[f"Y_{k}" for k in range(catorder + 1)])[
                    {"Y_2": 0, "Y_1": 2, "Y_0": 0}])
assert 1 == int(engine.contract(coreDict={**get_sum_tn(m, catorder), **encode_digits("00", "00", m)},
                                openColors=[])[:])
## Example: 10+11=101 in basis 2
m = 2
catorder = 2
assert 1 == int(engine.contract(coreDict={**get_sum_tn(m, catorder), **encode_digits("10", "11", m)},
                                openColors=[f"Y_{k}" for k in range(catorder + 1)])[
                    {"Y_2": 1, "Y_1": 0, "Y_0": 1}])
assert 1 == int(engine.contract(coreDict={**get_sum_tn(m, catorder), **encode_digits("10", "11", m)},
                                openColors=[])[:])

from demonstrations.comp_act_nets.algorithms import propagation as cp

edgeDirections = {
    **{f"X_{i}_eC": [[], [f"X_{i}"]] for i in range(catorder)},
    **{f"TX_{i}_eC": [[], [f"TX_{i}"]] for i in range(catorder)},
    "b_0": [["X_0", "TX_0"], ["Y_0", "Z_0"]],
    **{f"b__{i}": [[f"X_{i}", f"TX_{i}", f"Z_{i - 1}"], [f"Y_{i}", f"Z_{i}"]]
       for i in range(1, catorder - 1)},
    f"b_{catorder - 1}": [[f"X_{catorder - 1}", f"TX_{catorder - 1}", f"Z_{catorder - 2}"],
                          [f"Y_{catorder - 1}", f"Y_{catorder}"]],
}

propagator = cp.ContractionPropagation({**get_sum_tn(m, catorder), **encode_digits("01", "01", m)})
propagator.directed_propagation(edgeDirections=deepcopy(edgeDirections))

## Check whether the message arrived at b_1 states that the carry bit is 1
assert propagator.messages["b_1"]["b_0"][{"Z_0": 0}] == 0
assert propagator.messages["b_1"]["b_0"][{"Z_0": 1}] == 1

propagator = cp.ContractionPropagation({**get_sum_tn(m, catorder),
                                        **encode_digits("10", "10", m)})
propagator.directed_propagation(edgeDirections=deepcopy(edgeDirections))

## Check whether the message arrived at b_1 states that the carry bit is 1
assert propagator.messages["b_1"]["b_0"][{"Z_0": 0}] == 1
assert propagator.messages["b_1"]["b_0"][{"Z_0": 1}] == 0