from .nfw import NFW
from .mcr_util import physical_nfw_to_autogalaxy
from autogalaxy.cosmology.lensing import LensingCosmology
from autogalaxy.cosmology.wrap import Planck15
from typing import Tuple


class NFWPhysical(NFW):
    '''
    An NFW halo, sampled by it's physical parameters (mass, concentration).

    This class can interpret any mass definition known by `colossus`, specifed by the
    `mdef` argument. By default, this is `200c`, but one could use `200m`, `500c`, `vir`,
    etc.

    The `centre` and `ell_comps` are interpreted exactly the same as in the `NFW` halo defined
    in lensing units. i.e., `centre` is in arcsec.
    '''
    def __init__(
            self,
            centre: Tuple[float, float] = (0.0, 0.0),
            ell_comps: Tuple[float, float] = (0.0, 0.0),
            log10M: float = 12,
            concentration: float = 10,
            mdef: str = '200c',
            redshift_object: float = 0.5,
            redshift_source: float = 1.0,
            cosmo: LensingCosmology = Planck15(),
        ):
        self.params = {
            'log10M': log10M,
            'concentration': concentration,
            'mdef': mdef,
            'redshift_object': redshift_object,
            'redshift_source': redshift_source
        }
        kappa_s, scale_radius, radius_at_100 = physical_nfw_to_autogalaxy(
            10**log10M, concentration,
            mdef=mdef, redshift_object=redshift_object, redshift_source=redshift_source,
            cosmology=cosmo
        )
        super().__init__(
            centre, ell_comps,
            kappa_s, scale_radius,
        )
