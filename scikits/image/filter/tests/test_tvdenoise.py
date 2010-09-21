import numpy as np
from numpy.testing import assert_array_almost_equal
import scipy


from scikits.image.filter.tvdenoise import tvdenoise

def gradient_magnitude(a):
    grad_mag = np.zeros(tuple(np.array(a.shape) - 1), dtype=a.dtype)
    for dim in range(a.ndim):
        a_roll = np.rollaxis(a, dim, start=0)
        grad_axis = np.rollaxis(np.diff(a, axis=0)[[slice(None, None)] + \
            (a.ndim -1) * [slice(0, -1)]], 0, start=dim)
        grad_mag += (grad_axis ** 2)
    return np.sqrt(grad_mag)

def test_tvdenoise():
    lena = scipy.lena().astype(np.float)
    noisy_lena = lena + 0.2 * lena.std()*np.random.randn(*lena.shape)
    denoised_lena_W5 = tvdenoise(lena, niter=10, W=5.0)
    denoised_lena_W50 = tvdenoise(lena, niter=10, W=50.)
    grad_mag_lena = gradient_magnitude(lena).sum()
    grad_mag_noisy = gradient_magnitude(noisy_lena).sum()
    grad_mag_denoised_W5 = gradient_magnitude(denoised_lena_W5).sum()
    grad_mag_denoised_W50 = gradient_magnitude(denoised_lena_W50).sum()
    assert grad_mag_noisy > max(grad_mag_denoised_W5, grad_mag_denoised_W50)
    assert grad_mag_denoised_W5 > grad_mag_denoised_W50
    assert grad_mag_denoised_W5 > 0.5 * grad_mag_lena 
