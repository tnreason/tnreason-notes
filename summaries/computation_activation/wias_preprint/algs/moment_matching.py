from tnreason.engine import contract
from tnreason.engine import create_from_slice_iterator as create
import math


class MomentMatcher:
    def __init__(self, cores, hCols, satRates):
        self.cores = cores
        self.hCols = hCols
        self.satRates = satRates

        self.hardParams = {hCol: int(satRates[hCol]) for hCol in self.hCols if
                           satRates[hCol] in [0, 1]}
        self.softParams = {hCol: 0 for hCol in self.hCols if hCol not in self.hardParams}

    def update_canonical_parameter(self, uCol):
        con = contract({**self.cores,
                        **{hCol: create(shape=[2], colors=[hCol],
                                        sliceIterator=[(1, {hCol: self.hardParams[hCol]})])
                           for hCol in self.hardParams},
                        **{hCol: create(shape=[2], colors=[hCol],
                                        sliceIterator=[(1, {hCol: 0}),
                                                       (math.exp(self.softParams[hCol]), {hCol: 1})])
                           for hCol in self.softParams if hCol != uCol}
                        }, openColors=[uCol])
        self.softParams[uCol] = math.log(self.satRates[uCol] * con[{uCol: 0}] / (
                (1 - self.satRates[uCol]) * con[{uCol: 1}]))

    def alternate(self, iterations=1):
        for _ in range(iterations):
            for hCol in self.softParams:
                self.update_canonical_parameter(hCol)