import pandas as pd

samples = pd.DataFrame({
    "A1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "A2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "F": [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
})

formulaExpressions = {
    "(xor_A1_A2)": ["xor", "A1", "A2", 0],
    "(imp_F_A1)": ["imp", "F", "A1", 0],
}

from tnreason.engine import normalize
from tnreason.application import data_to_cores as dtc
from tnreason.application import create_cores_to_expressionsDict as cte
from demonstrations.comp_act_nets.algorithms import moment_matching as mm

satRates = {
    formulaKey + "_cV":
        normalize({**dtc.create_data_cores(samples),
                   **cte({formulaKey: formulaExpressions[formulaKey]})},
                  outColors=[formulaKey + "_cV"], inColors=[])[{formulaKey + "_cV": 1}]
    for formulaKey in formulaExpressions
}

matcher = mm.MomentMatcher(cores=cte(formulaExpressions),
                           satRates=satRates, hCols=["(xor_A1_A2)_cV", "(imp_F_A1)_cV"])
matcher.alternate(iterations=1)
assert abs(matcher.softParams["(imp_F_A1)_cV"] - 1.09861228866811) < 1e-8