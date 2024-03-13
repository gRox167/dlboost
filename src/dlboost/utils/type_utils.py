from typing import Any, Dict, Literal, NotRequired, Sequence, Tuple, TypedDict, Union

from jaxtyping import Array, Complex, Float, PyTree, Shaped
from torch import Size, Tensor

KspaceData = Complex[Tensor, "length"]
KspaceTraj = Float[Tensor, "2 length"]
ComplexImage2D = Complex[Tensor, "h w"] | Float[Tensor, "h w"]
ComplexImage3D = Shaped[ComplexImage2D, "d"]


Location = Sequence[slice]
Data_With_Location = Tuple[Any, Location]
PadSizes = Sequence[Tuple[int, int]]
NamedSize = Dict[str, int]
Patches_With_PadSizes = TypedDict(
    "Patches_With_PadSizes",
    {
        "patches": Sequence[Data_With_Location],
        "pad_sizes": NotRequired[PadSizes],
        "padded_size": NotRequired[Size],
    },
)

XArrayDevice = Literal["xarray", "disk", "xa"]
