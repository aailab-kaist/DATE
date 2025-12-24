__version__ = "0.16.0.dev0"

from .configuration_utils import ConfigMixin
from .utils import (
    OptionalDependencyNotAvailable,
    is_inflect_available,
    is_note_seq_available,
    is_scipy_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
    is_unidecode_available,
    logging,
)

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_pt_objects import *  # noqa F403
else:
    from .models import (
        AutoencoderKL,
        ControlNetModel,
        ModelMixin,
        PriorTransformer,
        T5FilmDecoder,
        Transformer2DModel,
        UNet1DModel,
        UNet2DConditionModel,
        UNet2DModel,
        UNet3DConditionModel,
        VQModel,
    )
    from .optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
    )
    from .schedulers import (
        DDIMInverseScheduler,
        DDIMScheduler,
        SDEScheduler,
        DDPMScheduler,
        DEISMultistepScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        IPNDMScheduler,
        KarrasVeScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        PNDMScheduler,
        RePaintScheduler,
        SchedulerMixin,
        ScoreSdeVeScheduler,
        UnCLIPScheduler,
        UniPCMultistepScheduler,
        VQDiffusionScheduler,
    )
    from .training_utils import EMAModel

try:
    if not (is_torch_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_scipy_objects import *  # noqa F403
else:
    from .schedulers import LMSDiscreteScheduler


try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .loaders import TextualInversionLoaderMixin
    from .pipelines import (
        StableDiffusionPipeline,
        StableDiffusionDATECLIPPipeline,
        StableDiffusionDATEIRPipeline,
    )
