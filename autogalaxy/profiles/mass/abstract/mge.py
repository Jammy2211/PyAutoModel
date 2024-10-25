import os

if os.environ.get("USE_JAX", "0") == "1":
    USING_JAX = True
    from .mge_jax import MassProfileMGE
else:
    USING_JAX = False
    from .mge_numpy import MassProfileMGE
