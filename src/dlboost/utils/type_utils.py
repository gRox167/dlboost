from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Hashable,
    List,
    Mapping,
    NotRequired,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import torch

Location = Tuple[slice, ...]
Data_With_Location = Tuple[Any, Location]
Pad_Sizes = Sequence[Tuple[int, int]]
Patches_With_Pad_Sizes = TypedDict(
    "Patches_With_Pad_Sizes",
    {
        "patches": Sequence[Data_With_Location],
        "pad_sizes": NotRequired[Pad_Sizes],
        "padded_size": NotRequired[torch.Size],
    },
)
