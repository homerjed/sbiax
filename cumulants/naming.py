from typing import Callable, Optional, Tuple
from jaxtyping import PRNGKeyArray, Float, Array, jaxtyped
from beartype import beartype as typechecker    

typecheck = jaxtyped(typechecker=typechecker)

BulkTails = Literal["tails", "bulk", "bulk_pdf"]

CompressionType = Literal["linear", "nn"]

CompressionFn = Callable[[Float[Array, "d"], Float[Array, "p"]], Float[Array, "p"]]