# -*-coding:utf-8-*-
# cython:language_level=3
# boussinesq_eq1d.pyx
import numpy as np
cimport numpy as cnp
from scipy.special import erfc
from cython import boundscheck, wraparound


@boundscheck(False)
@wraparound(False)
cpdef cnp.float32_t[:, :] h(
    cnp.ndarray[float, ndim=2] y,
    int t,
    cnp.ndarray[float, ndim=3] rgse_diff,
    cnp.float32_t[:, :, :] a,
    cnp.ndarray[float, ndim=3] diff):
    cdef:
        cnp.ndarray[int, ndim=1] t_m = np.arange(t)
        cnp.ndarray[int, ndim=1] t_i = np.full(t_m.shape[0], t)
        cnp.float32_t[:, :, :] sqrt_term = np.sqrt(a * (t_i - t_m)[np.newaxis, np.newaxis, ...].astype(np.float32))
        cnp.float32_t[:, :, :] denominator = np.float32(2) * sqrt_term
        cnp.float32_t[:, :, :] erfc_arg = y[..., np.newaxis] / denominator
        cnp.float32_t[:, :, :] erfc_term = erfc(erfc_arg).astype(np.float32)
        cnp.ndarray[float, ndim=3] value = (diff[..., t_m] - rgse_diff[..., t_m]) * erfc_term + rgse_diff[..., t_m]
        cnp.float32_t[:, :] result = np.sum(value, axis=-1)
    return result
