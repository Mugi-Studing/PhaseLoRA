import abc
from typing import Any, Dict, Optional


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict, *, noise: Optional[Any] = None) -> Dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
