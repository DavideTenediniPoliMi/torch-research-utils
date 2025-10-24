class Callback:
    """
    Base class used to build new callbacks.

    The methods are called by the Trainer at different points during training.
    """

    def on_train_begin(self) -> None:
        """Called at the beginning of training."""

    def on_train_end(self) -> None:
        """Called at the end of training."""

    def on_epoch_begin(self, epoch: int) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch."""

    def on_batch_begin(self) -> None:
        """Called at the beginning of each training batch."""

    def on_batch_end(self, batch: int) -> None:
        """Called at the end of each training batch."""
