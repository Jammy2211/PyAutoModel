from astropy import units
from autofit.jax_wrapper import numpy as np
from typing import Tuple


def ell_comps_from(axis_ratio: float, angle: float) -> Tuple[float, float]:
    """
    Returns the elliptical components e1 and e2 of a light or mass profile from an input angle in degrees and axis
    ratio.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-axis_ratio)/(1+axis_ratio) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-axis_ratio)/(1+axis_ratio) * cos(2 * angle)

    Which are the values this function returns.

    Parameters
    ----------
    axis_ratio
        Ratio of light profiles ellipse's minor and major axes (b/a).
    angle
        Rotation angle of light profile counter-clockwise from positive x-axis.
    """
    angle *= np.pi / 180.0
    fac = (1 - axis_ratio) / (1 + axis_ratio)
    ellip_y = fac * np.sin(2 * angle)
    ellip_x = fac * np.cos(2 * angle)
    return (ellip_y, ellip_x)


def axis_ratio_and_angle_from(ell_comps: Tuple[float, float]) -> Tuple[float, float]:
    """
    Returns the axis-ratio and position angle in degrees (-45 < angle < 135.0) from input elliptical components e1
    and e2 of a light or mass profile.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-axis_ratio)/(1+axis_ratio) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-axis_ratio)/(1+axis_ratio) * cos(2 * angle)

    The axis-ratio and angle are therefore given by:

    axis_ratio = (1 - fac) / (1 + fac)
    angle = 0.5 * arctan(ell_comps[0] / ell_comps[1])

    where `fac = sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2).

    This function returns the axis-ratio and angle in degrees.

    An additional check is performed which requires the angle is between -45 and 135 degrees. This ensures that
    for certain values of `ell_comps` the angle does not jump from one boundary to another (e.g. without this check
    certain values of `ell_comps` return -1.0 degrees and others 179.0 degrees). This ensures that when error
    estimates are computed from samples of a lens model via marginalization, the calculation is not biased by the
    angle jumping between these two values.

    Parameters
    ----------
    ell_comps
        The elliptical components of the light or mass profile which are converted to an angle.
    """
    angle = np.arctan2(ell_comps[0], ell_comps[1]) / 2
    angle *= 180.0 / np.pi

    if abs(angle) > 45 and angle < 0:
        angle += 180

    fac = np.sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2)
    if fac > 0.999:
        fac = 0.999  # avoid unphysical solution
    # if fac > 1: print('unphysical e1,e2')
    axis_ratio = (1 - fac) / (1 + fac)
    return axis_ratio, angle


def axis_ratio_from(ell_comps: Tuple[float, float]):
    """
    Returns the axis-ratio from input elliptical components e1 and e2 of a light or mass profile.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-axis_ratio)/(1+axis_ratio) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-axis_ratio)/(1+axis_ratio) * cos(2 * angle)

    The axis-ratio is therefore given by:

    axis_ratio = (1 - fac) / (1 + fac)
    angle = 0.5 * arctan(ell_comps[0] / ell_comps[1])

    where `fac = sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2).

    This function returns the axis-ratio.

    Parameters
    ----------
    ell_comps
        The elliptical components of the light or mass profile which are converted to an angle.
    """
    axis_ratio, angle = axis_ratio_and_angle_from(ell_comps=ell_comps)
    return axis_ratio


def angle_from(ell_comps: Tuple[float, float]) -> float:
    """
    Returns the position angle in degrees (-45 < angle < 135.0) from input elliptical components e1 and e2
    of a light or mass profile.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-q)/(1+q) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-q)/(1+q) * cos(2 * angle)

    axis_ratio = (1 - fac) / (1 + fac)
    angle = 0.5 * arctan(ell_comps[0] / ell_comps[1])

    where `fac = sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2).

    This function returns the angle in degrees.

    An additional check is performed which requires the angle is between -45 and 135 degrees. This ensures that
    for certain values of `ell_comps` the angle does not jump from one boundary to another (e.g. without this check
    certain values of `ell_comps` return -1.0 degrees and others 179.0 degrees).

    This ensures that when error estimates are computed from samples of a lens model via marginalization, the
    calculation is not biased by the angle jumping between these two values.

    Parameters
    ----------
    ell_comps
        The elliptical components of the light or mass profile which are converted to an angle.
    """
    axis_ratio, angle = axis_ratio_and_angle_from(ell_comps=ell_comps)

    return angle


def shear_gamma_1_2_from(magnitude: float, angle: float) -> Tuple[float, float]:
    """
    Returns the shear gamma 1 and gamma 2 values an input shear magnitude and angle in degrees.

    The gamma 1 and gamma 2 components of a shear are given by:

    gamma_1 = magnitude * np.cos(2 * angle * np.pi / 180.0)
    gamma_2 = magnitude * np.sin(2 * angle * np.pi / 180.0)

    Which are the values this function returns.

    Converting from gamma 1 and gamma 2 to magnitude and angle is given by:

    magnitude = np.sqrt(gamma_1**2 + gamma_2**2)
    angle = np.arctan2(gamma_2, gamma_1) / 2 * 180.0 / np.pi

    Parameters
    ----------
    axis_ratio
        Ratio of light profiles ellipse's minor and major axes (b/a).
    angle
        Rotation angle of light profile counter-clockwise from positive x-axis.
    """
    gamma_1 = magnitude * np.cos(2 * angle * np.pi / 180.0)
    gamma_2 = magnitude * np.sin(2 * angle * np.pi / 180.0)
    return (gamma_1, gamma_2)


def shear_magnitude_and_angle_from(
    gamma_1: float, gamma_2: float
) -> Tuple[float, float]:
    """
    Returns the shear magnitude and angle in degrees from input shear gamma 1 and gamma 2 values.

    The gamma 1 and gamma 2 components of a shear are given by:

    gamma_1 = magnitude * np.cos(2 * angle * np.pi / 180.0)
    gamma_2 = magnitude * np.sin(2 * angle * np.pi / 180.0)

    Converting from gamma 1 and gamma 2 to magnitude and angle is given by:

    magnitude = np.sqrt(gamma_1**2 + gamma_2**2)
    angle = np.arctan2(gamma_2, gamma_1) / 2 * 180.0 / np.pi

    Which are the values this function returns.

    Additional checks are performed which requires the angle is between -45 and 135 degrees. This ensures that
    for certain values of `gamma_1` and `gamma_2` the angle does not jump from one boundary to another (e.g. without
    this check certain values of `gamma_1` and `gamma_2` return -1.0 degrees and others 179.0 degrees).

    This ensures that when error estimates are computed from samples of a lens model via marginalization, the
    calculation is not biased by the angle jumping between these two values.

    Parameters
    ----------
    gamma_1
        The gamma 1 component of the shear.
    gamma_2
        The gamma 2 component of the shear.
    """
    angle = np.arctan2(gamma_2, gamma_1) / 2 * 180.0 / np.pi
    magnitude = np.sqrt(gamma_1**2 + gamma_2**2)

    if angle < 0:
        angle += 180.0

    if abs(angle - 90) > 45 and angle > 90:
        angle -= 180

    return magnitude, angle


def shear_magnitude_from(gamma_1: float, gamma_2: float) -> float:
    """
    Returns the shear magnitude and angle in degrees from input shear gamma 1 and gamma 2 values.

    The gamma 1 and gamma 2 components of a shear are given by:

    gamma_1 = magnitude * np.cos(2 * angle * np.pi / 180.0)
    gamma_2 = magnitude * np.sin(2 * angle * np.pi / 180.0)

    Converting from gamma 1 and gamma 2 to magnitude and angle is given by:

    magnitude = np.sqrt(gamma_1**2 + gamma_2**2)
    angle = np.arctan2(gamma_2, gamma_1) / 2 * 180.0 / np.pi

    The magnitude value is what this function returns.

    Parameters
    ----------
    gamma_1
        The gamma 1 component of the shear.
    gamma_2
        The gamma 2 component of the shear.
    """
    magnitude, angle = shear_magnitude_and_angle_from(gamma_1=gamma_1, gamma_2=gamma_2)
    return magnitude


def shear_angle_from(gamma_1: float, gamma_2: float) -> float:
    """
    Returns the shear magnitude and angle in degrees from input shear gamma 1 and gamma 2 values.

    The gamma 1 and gamma 2 components of a shear are given by:

    gamma_1 = magnitude * np.cos(2 * angle * np.pi / 180.0)
    gamma_2 = magnitude * np.sin(2 * angle * np.pi / 180.0)

    Converting from gamma 1 and gamma 2 to magnitude and angle is given by:

    magnitude = np.sqrt(gamma_1**2 + gamma_2**2)
    angle = np.arctan2(gamma_2, gamma_1) / 2 * 180.0 / np.pi

    The angle value is what this function returns.

    Parameters
    ----------
    gamma_1
        The gamma 1 component of the shear.
    gamma_2
        The gamma 2 component of the shear.
    """
    magnitude, angle = shear_magnitude_and_angle_from(gamma_1=gamma_1, gamma_2=gamma_2)
    return angle


def multipole_k_m_and_phi_m_from(
    multipole_comps: Tuple[float, float], m: int
) -> Tuple[float, float]:
    """
    Returns the multipole normalization value `k_m` and angle `phi` from the multipole component parameters.

    The normalization and angle are given by:

    .. math::
        \phi^{\rm mass}_m = \arctan{\frac{\epsilon_{\rm 2}^{\rm mp}}{\epsilon_{\rm 2}^{\rm mp}}}, \, \,
        k^{\rm mass}_m = \sqrt{{\epsilon_{\rm 1}^{\rm mp}}^2 + {\epsilon_{\rm 2}^{\rm mp}}^2} \, .

    The conversion depends on the multipole order `m`, to ensure that all possible rotationally symmetric
    multiple mass profiles are available in the conversion for multiple components spanning -inf to inf.

    Additional checks are performed which requires the angle `phi_m` is between -45 and 135 degrees. This ensures that
    for certain multipole component values the angle does not jump from one boundary to another (e.g. without
    this check certain values of `gamma_1` and `gamma_2` return -1.0 degrees and others 179.0 degrees).

    This ensures that when error estimates are computed from samples of a lens model via marginalization, the
    calculation is not biased by the angle jumping between these two values.

    Parameters
    ----------
    multipole_comps
        The first and second components of the multipole.

    Returns
    -------
    The normalization and angle parameters of the multipole.
    """
    phi_m = (
        np.arctan2(multipole_comps[0], multipole_comps[1]) * 180.0 / np.pi / float(m)
    )
    k_m = np.sqrt(multipole_comps[1] ** 2 + multipole_comps[0] ** 2)

    if phi_m < -90.0 / m:
        phi_m += 360.0 / m

    return k_m, phi_m


def multipole_comps_from(k_m: float, phi_m: float, m: int) -> Tuple[float, float]:
    """
    Returns the multipole component parameters from their normalization value `k_m` and angle `phi`.

    .. math::
        \phi^{\rm mass}_m = \arctan{\frac{\epsilon_{\rm 2}^{\rm mp}}{\epsilon_{\rm 2}^{\rm mp}}}, \, \,
        k^{\rm mass}_m = \sqrt{{\epsilon_{\rm 1}^{\rm mp}}^2 + {\epsilon_{\rm 2}^{\rm mp}}^2} \, .

    The conversion depends on the multipole order `m`, to ensure that all possible rotationally symmetric
    multiple mass profiles are available in the conversion for multiple components spanning -inf to inf.

    Parameters
    ----------
    k_m
        The magnitude of the multipole.
    phi_m
        The angle of the multipole.

    Returns
    -------
    The multipole component parameters.
    """
    multipole_comp_0 = k_m * np.sin(phi_m * float(m) * units.deg.to(units.rad))
    multipole_comp_1 = k_m * np.cos(phi_m * float(m) * units.deg.to(units.rad))

    return (multipole_comp_0, multipole_comp_1)
