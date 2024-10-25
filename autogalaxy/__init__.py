from autoconf.dictable import from_dict, from_json, output_to_json, to_dict
from autoarray.dataset import preprocess  # noqa
from autoarray.dataset.imaging.dataset import Imaging  # noqa
from autoarray.dataset.interferometer.dataset import Interferometer  # noqa
from autoarray.dataset.dataset_model import DatasetModel
from autoarray.dataset.over_sampling import OverSamplingDataset
from autoarray.inversion.pixelization import mesh  # noqa
from autoarray.inversion import regularization as reg  # noqa
from autoarray.inversion.pixelization import image_mesh
from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper  # noqa
from autoarray.inversion.inversion.settings import SettingsInversion  # noqa
from autoarray.inversion.inversion.factory import inversion_from as Inversion  # noqa
from autoarray.inversion.pixelization.image_mesh.abstract import AbstractImageMesh
from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.pixelization.pixelization import Pixelization  # noqa
from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.inversion.pixelization.mappers.mapper_grids import MapperGrids  # noqa
from autoarray.inversion.pixelization.mappers.factory import (
    mapper_from as Mapper,
)  # noqa
from autoarray.inversion.pixelization.border_relocator import BorderRelocator
from autoarray.mask.mask_1d import Mask1D  # noqa
from autoarray.mask.mask_2d import Mask2D  # noqa
from autoarray.operators.convolver import Convolver  # noqa
from autoarray.operators.convolver import Convolver  # noqa
from autoarray.operators.transformer import TransformerDFT  # noqa
from autoarray.operators.transformer import TransformerNUFFT  # noqa
from autoarray.layout.layout import Layout2D  # noqa
from autoarray.structures.arrays.uniform_1d import Array1D  # noqa
from autoarray.structures.arrays.uniform_2d import Array2D  # noqa
from autoarray.structures.arrays.irregular import ArrayIrregular  # noqa
from autoarray.structures.header import Header  # noqa
from autoarray.structures.grids.uniform_1d import Grid1D  # noqa
from autoarray.structures.grids.uniform_2d import Grid2D  # noqa
from autoarray.structures.grids.irregular_2d import Grid2DIrregular  # noqa
from autoarray.structures.grids.irregular_2d import Grid2DIrregularUniform  # noqa
from autoarray.operators.over_sampling.uniform import OverSamplingUniform  # noqa
from autoarray.operators.over_sampling.uniform import OverSamplerUniform  # noqa
from autoarray.operators.over_sampling.iterate import OverSamplingIterate
from autoarray.operators.over_sampling.iterate import OverSamplerIterate
from autoarray.structures.mesh.rectangular_2d import Mesh2DRectangular  # noqa
from autoarray.structures.mesh.voronoi_2d import Mesh2DVoronoi  # noqa
from autoarray.structures.mesh.delaunay_2d import Mesh2DDelaunay  # noqa
from autoarray.structures.vectors.uniform import VectorYX2D  # noqa
from autoarray.structures.vectors.irregular import VectorYX2DIrregular  # noqa
from autoarray.layout.region import Region1D  # noqa
from autoarray.layout.region import Region2D  # noqa
from autoarray.structures.arrays.kernel_2d import Kernel2D  # noqa
from autoarray.structures.visibilities import Visibilities  # noqa
from autoarray.structures.visibilities import VisibilitiesNoiseMap  # noqa

from .analysis.adapt_images.adapt_images import AdaptImages
from .analysis.adapt_images.adapt_image_maker import AdaptImageMaker
from . import aggregator as agg
from . import exc
from . import plot
from . import util
from .ellipse.dataset_interp import DatasetInterp
from .ellipse.ellipse.ellipse import Ellipse
from .ellipse.ellipse.ellipse_multipole import EllipseMultipole
from .ellipse.fit_ellipse import FitEllipse
from .ellipse.model.analysis import AnalysisEllipse
from .operate.image import OperateImage
from .operate.image import OperateImageList
from .operate.image import OperateImageGalaxies
from .operate.deflections import OperateDeflections
from .gui.scribbler import Scribbler
from .imaging.fit_imaging import FitImaging
from .imaging.model.analysis import AnalysisImaging
from .imaging.simulator import SimulatorImaging
from .interferometer.simulator import SimulatorInterferometer
from .interferometer.fit_interferometer import FitInterferometer
from .interferometer.model.analysis import AnalysisInterferometer

from .quantity.fit_quantity import FitQuantity
from .quantity.model.analysis import AnalysisQuantity
from .quantity.dataset_quantity import DatasetQuantity
from .galaxy.galaxy import Galaxy
from .galaxy.galaxies import Galaxies
from .galaxy.redshift import Redshift
from .galaxy.stellar_dark_decomp import StellarDarkDecomp
from .galaxy.to_inversion import AbstractToInversion
from .galaxy.to_inversion import GalaxiesToInversion
from .profiles.geometry_profiles import EllProfile
from .profiles import (
    point_sources as ps,
    mass as mp,
    light_and_mass_profiles as lmp,
    light_linear_and_mass_profiles as lmp_linear,
    scaling_relations as sr,
)
from .profiles.light.abstract import LightProfile
from .profiles.light import standard as lp
from .profiles import basis as lp_basis
from .profiles.light.linear import LightProfileLinearObjFuncList
from .profiles.light import linear as lp_linear
from .profiles.light import operated as lp_operated
from .profiles.light import (
    linear_operated as lp_linear_operated,
)
from .profiles.light import snr as lp_snr
from . import convert
from . import mock as m  # noqa
from .util.shear_field import ShearYX2D
from .util.shear_field import ShearYX2DIrregular
from . import cosmology as cosmo
from .gui.clicker import Clicker
from .gui.scribbler import Scribbler

from .analysis.clump_model import ClumpModel

from autoconf import conf

conf.instance.register(__file__)

__version__ = "2024.9.21.2"
