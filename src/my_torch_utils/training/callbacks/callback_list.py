from my_torch_utils.training.callbacks.base_callback import Callback


class CallbackList:
    """Container for managing a list of Callback instances."""

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks = callbacks or []

    def on_train_begin(self) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end()

    def on_epoch_begin(self, epoch: int) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch: int) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_batch_begin(self) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self, batch: int) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch)
