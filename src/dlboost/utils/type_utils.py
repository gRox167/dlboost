from re import T
from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    NotRequired,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

from jaxtyping import Array, Complex, Float, PyTree, Shaped
from torch import Size, Tensor

KspaceData = Complex[Tensor, "length"]
KspaceSpokesData = Float[Tensor, "spokes_num spoke_length"]
KspaceTraj = Float[Tensor, "2 length"]
KspaceSpokesTraj = Float[Tensor, "2 spokes_num spoke_length"]
ComplexImage2D = Complex[Tensor, "h w"] | Float[Tensor, "h w"]
ComplexImage3D = Shaped[ComplexImage2D, "d"]

