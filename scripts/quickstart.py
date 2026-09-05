"""Synthetic demonstration of the representation contract; no trained prediction."""
import numpy as np
from lpcode_v1.representations import build_representation
h=np.zeros((1,28));c=np.ones((1,28))
result=build_representation(h,c,'full')
assert result.shape==(1,112) and np.isfinite(result).all()
print('MSTF representation:',result.shape,'finite=',bool(np.isfinite(result).all()))
print('Synthetic feature example only. No checkpoint or detection score is claimed.')
