from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Callback(Protocol):
    """
    Protocol for training callbacks.
    Implement only the methods you need.
    """

    def on_train_begin(self, trainer: Any) -> None: ...
    def on_train_end(self, trainer: Any) -> None: ...
    def on_epoch_begin(self, trainer: Any) -> None: ...
    def on_epoch_end(self, trainer: Any, logs: dict[str, float]) -> None: ...
    def on_batch_begin(self, trainer: Any) -> None: ...
    def on_batch_end(self, trainer: Any) -> None: ...


class CallbackList:
    """Internal utility to fire events for a list of callbacks."""

    def __init__(self, callbacks: list[Callback] | None) -> None:
        self.callbacks = callbacks or []

    def fire(self, event: str, **kwargs: Any) -> None:
        """Fire a certain event for all the callbacks in this list."""
        for cb in self.callbacks:
            method = getattr(cb, event, None)
            if method:
                method(**kwargs)
