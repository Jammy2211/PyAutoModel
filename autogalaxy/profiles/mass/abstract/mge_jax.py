import jax.numpy as np
import jax
import numpy as base_np

from .jax_utils import w_f_approx, all_comb


class MassProfileMGE:
    """
    This class speeds up deflection angle calculations of certain mass profiles by decompositing them into many
    Gaussians.

    This follows the method of Shajib 2019 - https://academic.oup.com/mnras/article/488/1/1387/5526256
    """

    def __init__(self):
        pass

    @staticmethod
    def zeta_from(grid, amps, sigmas, axis_ratio):
        """
        The key part to compute the deflection angle of each Gaussian.
        """
        q2 = axis_ratio**2.0

        scale_factor = axis_ratio / np.sqrt(2.0 * (1.0 - q2))

        xs = np.array((grid.array[:, 1] * scale_factor).copy())
        ys = np.array((grid.array[:, 0] * scale_factor).copy())

        y_sign = np.sign(ys)
        ys = ys * y_sign

        z = xs + 1j * ys
        zq = axis_ratio * xs + 1j * ys / axis_ratio
        expv = -(xs**2.0) * (1.0 - q2) - ys**2.0 * (1.0 / q2 - 1.0)
        sigma_ = sigmas.reshape((-1,) + (1,) * xs.ndim)
        inv_sigma_ = 1 / sigma_
        amps_ = amps.reshape((-1,) + (1,) * xs.ndim)

        # process as one big vectorized calculation
        # could try `jax.lax.scan` instead if this is too much memory
        w = w_f_approx(inv_sigma_ * z)
        wq = w_f_approx(inv_sigma_ * zq)
        exp_factor = np.exp(inv_sigma_**2 * expv)

        sigma_func_real = w.imag - exp_factor * wq.imag
        sigma_func_imag = (-w.real + exp_factor * wq.real) * y_sign

        output_grid = sigma_ * amps_ * (sigma_func_real + 1j * sigma_func_imag)
        return output_grid.sum(axis=0)

    @staticmethod
    def kesi(p):
        """
        see Eq.(6) of 1906.08263
        """
        n_list = np.arange(0, 2 * p + 1, 1)
        return (2.0 * p * np.log(10) / 3.0 + 2.0 * np.pi * n_list * 1j) ** (0.5)

    @staticmethod
    def eta(p):
        """
        see Eq.(6) of 1906.00263
        """

        i = np.arange(1, p, 1)
        kesi_last = 1 / 2**p
        k = kesi_last + np.cumsum(np.cumprod((p + 1 - i) / i) * kesi_last)

        kesi_list = np.hstack(
            [np.array([0.5]), np.ones(p), k[::-1], np.array([kesi_last])]
        )
        coef = (-1) ** np.arange(0, 2 * p + 1, 1)
        eta_list = coef * 2.0 * np.sqrt(2.0 * np.pi) * 10 ** (p / 3.0) * kesi_list
        return eta_list

    def decompose_convergence_via_mge(self):
        raise NotImplementedError()

    def _decompose_convergence_via_mge(
        self, func, radii_min, radii_max, func_terms=28, func_gaussians=20
    ):
        """

        Parameters
        ----------
        func : func
            The function representing the profile that is decomposed into Gaussians.
        normalization
            A normalization factor tyh
        func_terms
            The number of terms used to approximate the input func.
        func_gaussians
            The number of Gaussians used to represent the input func.

        Returns
        -------
        """

        kesis = self.kesi(func_terms)  # kesi in Eq.(6) of 1906.08263
        etas = self.eta(func_terms)  # eta in Eqr.(6) of 1906.08263

        # sigma is sampled from logspace between these radii.

        log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), func_gaussians)
        d_log_sigma = log_sigmas[1] - log_sigmas[0]
        sigma_list = np.exp(log_sigmas)

        amplitude_list = np.zeros(func_gaussians)
        f_sigma = np.sum(
            etas * np.real(func(sigma_list.reshape(-1, 1) * kesis)), axis=1
        )
        amplitude_list = f_sigma * d_log_sigma / np.sqrt(2.0 * np.pi)
        amplitude_list = amplitude_list.at[0].multiply(0.5)
        amplitude_list = amplitude_list.at[-1].multiply(0.5)
        return amplitude_list, sigma_list

    def convergence_2d_via_mge_from(self, grid_radii):
        raise NotImplementedError()

    def _convergence_2d_via_mge_from(
        self, grid_radii, func_terms=28, func_gaussians=20
    ):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        amps, sigmas = self.decompose_convergence_via_mge(
            func_terms=func_terms, func_gaussians=func_gaussians
        )

        convergence = 0.0

        inv_sigma_ = 1 / sigmas.reshape((-1,) + (1,) * grid_radii.array.ndim)
        amps_ = amps.reshape((-1,) + (1,) * grid_radii.array.ndim)
        convergence = amps_ * np.exp(-0.5 * (grid_radii.array * inv_sigma_) ** 2)
        return convergence.sum(axis=0)

    def _deflections_2d_via_mge_from(
        self, grid, sigmas_factor=1.0, func_terms=28, func_gaussians=20
    ):
        axis_ratio = np.min(np.array([self.axis_ratio, 0.9999]))

        amps, sigmas = self.decompose_convergence_via_mge(
            func_terms=func_terms, func_gaussians=func_gaussians
        )
        sigmas *= sigmas_factor

        angle = self.zeta_from(
            grid=grid, amps=amps, sigmas=sigmas, axis_ratio=axis_ratio
        )

        angle *= np.sqrt((2.0 * np.pi) / (1.0 - axis_ratio**2.0))

        return self.rotated_grid_from_reference_frame_from(
            np.vstack((-angle.imag, angle.real)).T
        )
