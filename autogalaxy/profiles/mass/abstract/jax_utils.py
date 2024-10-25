import jax
import jax.numpy as jnp
import numpy as np

from jax import custom_jvp
from functools import partial


def reg1(z, _, i_sqrt_pi):
    return i_sqrt_pi / z


def reg2(z, _, i_sqrt_pi):
    z2 = z**2
    return i_sqrt_pi * z / (z2 - 0.5)


def reg3(z, _, i_sqrt_pi):
    z2 = z**2
    return (i_sqrt_pi / z) * (1 + 0.5 / (z2 - 1.5))


def reg4(z, _, i_sqrt_pi):
    z2 = z**2
    return (i_sqrt_pi * z) * (z2 - 2.5) / (z2 * (z2 - 3.0) + 0.75)


def reg5(z, sqrt_pi, _):
    mz2 = -(z**2)
    f1 = sqrt_pi
    f2 = 1.0
    s1 = [1.320522, 35.7668, 219.031, 1540.787, 3321.99, 36183.31]
    s2 = [1.841439, 61.57037, 364.2191, 2186.181, 9022.228, 24322.84, 32066.6]

    for s in s1:
        f1 = s - f1 * mz2
    for s in s2:
        f2 = s - f2 * mz2

    return jnp.exp(mz2) + 1j * z * f1 / f2


def reg6(z, sqrt_pi, _):
    miz = -1j * z
    f1 = sqrt_pi
    f2 = 1
    s1 = [5.9126262, 30.180142, 93.15558, 181.92853, 214.38239, 122.60793]
    s2 = [10.479857, 53.992907, 170.35400, 348.70392, 457.33448, 352.73063, 122.60793]

    for s in s1:
        f1 = s + f1 * miz
    for s in s2:
        f2 = s + f2 * miz

    return f1 / f2


@custom_jvp
def w_f_approx(z):
    """Compute the Faddeeva function :math:`w_{\\mathrm F}(z)` using the
    approximation given in Zaghloul (2017).

    :param z: complex number
    :type z: ``complex`` or ``numpy.array(dtype=complex)``
    :return: :math:`w_\\mathrm{F}(z)`
    :rtype: ``complex``

    # This function is a JAX conversion of
    # "https://github.com/sibirrer/lenstronomy/tree/master/lenstronomy/LensModel/Profiles"
    # original function written by Anowar J. Shajib (see 1906.08263)
    # JAX conversion written by Coleman M. Krawczyk
    """
    sqrt_pi = 1 / jnp.sqrt(jnp.pi)
    i_sqrt_pi = 1j * sqrt_pi

    z_imag2 = z.imag**2
    abs_z2 = z.real**2 + z_imag2

    r1 = abs_z2 >= 38000.0
    r2 = (abs_z2 >= 256.0) & (abs_z2 < 38000.0)
    r3 = (abs_z2 >= 62.0) & (abs_z2 < 256.0)
    r4 = (abs_z2 >= 30.0) & (abs_z2 < 62.0) & (z_imag2 >= 1e-13)
    # region bounds for 5 taken directly from Zaghloul (2017)
    # https://dl.acm.org/doi/pdf/10.1145/3119904
    r5_1 = (abs_z2 >= 30.0) & (abs_z2 < 62.0) & (z_imag2 < 1e-13)
    r5_2 = (abs_z2 >= 2.5) & (abs_z2 < 30.0) & (z_imag2 < 0.072)
    r5 = r5_1 | r5_2
    r6 = (abs_z2 < 30.0) & jnp.logical_not(r5)

    args = (z, sqrt_pi, i_sqrt_pi)
    wz = jnp.empty_like(z)
    wz = jnp.where(r1, reg1(*args), wz)
    wz = jnp.where(r2, reg2(*args), wz)
    wz = jnp.where(r3, reg3(*args), wz)
    wz = jnp.where(r4, reg4(*args), wz)
    wz = jnp.where(r5, reg5(*args), wz)
    wz = jnp.where(r6, reg6(*args), wz)
    return wz


@w_f_approx.defjvp
def w_f_approx_jvp(primals, tangents):
    # define a custom jvp to avoid the issue using `jnp.where` with `jax.grad`
    # also the derivative is defined analytically for this function so bypass
    # auto diffing over the complex functions above.
    (z,) = primals
    (z_dot,) = tangents
    primal_out = w_f_approx(z)
    i_sqrt_pi = 1j / jnp.sqrt(jnp.pi)
    tangent_out = z_dot * 2 * (i_sqrt_pi - z * primal_out)
    return primal_out, tangent_out


@partial(jax.jit, static_argnums=(0,))
def all_comb(n):
    i = jnp.arange(1, n, 1)
    return jnp.cumprod((n + 1 - i) / i)
