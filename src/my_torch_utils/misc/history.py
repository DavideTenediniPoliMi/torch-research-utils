from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from collections.abc import KeysView
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure

# A type alias for the raw data that can be passed to init or extend
HistoryInputData = dict[str, list[float] | list[tuple[float, float]]]
X_VAL = 0
Y_VAL = 1


class History:
    """
    Stores and visualizes time-series metrics from training processes.

    This class is designed to log metrics (e.g., loss, accuracy) where each
    data point is stored as an (time, value) tuple. This allows for different
    logging frequencies (e.g., per-step training loss, per-epoch validation
    loss).
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        history_data: HistoryInputData | None = None,
    ) -> None:
        """Initializes the History object.

        Args:
            metrics (list[str] | None): An optional list of metric keys to
                pre-initialize. Defaults to None.
            history_data (HistoryInputData | None): Optional dict to initialize
                time-series metrics. Defaults to None.
        """
        self.history: dict[str, list[tuple[float, float]]] = {}

        if metrics is not None:
            for metric in metrics:
                self.history[metric] = []

        if history_data is not None:
            self.extend(history_data)

    def __getitem__(self, key: str) -> list[tuple[float, float]]:
        """Retrieves the time-series history for a given metric key.

        Args:
            key (str): The name of the metric.

        Returns:
            list[tuple[float, float]]: A list of (time, value) tuples for the
                metric.
        """
        return self.history[key]

    def __contains__(self, key: str) -> bool:
        """Checks if a metric key exists in the time-series history.

        Args:
            key (str): The metric key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self.history

    def keys(self) -> KeysView[str]:
        """Returns the keys of the time-series history.

        Returns:
            KeysView[str]: A view of the metric keys.
        """
        return self.history.keys()

    def _get_delta(self, key: str) -> float:
        """Internal helper to compute the step size (delta) for a given metric.

        If fewer than two data points exist, defaults to 1.0 as the step size.
        Otherwise, computes the difference between the last two x values.

        Args:
            key (str): The metric key.

        Returns:
            float: The computed step size (delta).
        """
        if key in self.history and len(self.history[key]) >= 2:
            return abs(self.history[key][-1][X_VAL] - self.history[key][-2][X_VAL])
        return 1.0

    def _get_next_x(self, key: str) -> float:
        """Internal helper to compute the next x value for a given metric.
        If no previous data exists, returns 0.0. Otherwise, infers the next x
        based on the last recorded x value. If the last two points exist, uses
        their difference as the step size; otherwise defaults to 1.0 as step.

        Args:
            key (str): The metric key.

        Returns:
            float: The next x value.
        """
        if key not in self.history or len(self.history[key]) == 0:
            return 0.0
        return self.history[key][-1][X_VAL] + self._get_delta(key)

    def _process_values(
        self, key: str, values: list[float] | list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Internal helper to normalize input values to (x, y) tuples.

        Args:
            key (str): The metric key (used for inferring next x-value).
            values (list[float] | list[tuple[float, float]]): The list of
                values to process.

        Raises:
            ValueError: If the input values list is in an invalid format.

        Returns:
            list[tuple[float, float]]: A normalized list of (x, y) tuples.
        """
        if not values:
            return []

        # Check for list[float]
        if all(isinstance(v, int | float) for v in values):
            next_x = self._get_next_x(key)
            return [
                (next_x + i * self._get_delta(key), float(v))  # type: ignore
                for i, v in enumerate(values)
            ]

        # Check for list[tuple[float, float]]
        if all(
            isinstance(v, tuple)
            and len(v) == 2
            and all(isinstance(i, int | float) for i in v)
            for v in values
        ):
            return [(float(v[0]), float(v[1])) for v in values]  # type: ignore

        raise ValueError(
            f"Invalid values format for key '{key}'. "
            "Expected list[float] or list[tuple[float, float]]."
        )

    def extend(self, history: History | HistoryInputData) -> None:
        """Extend the history with another history object or dictionary.

        This adds new keys if they don't exist and appends to existing keys.

        Args:
            history (History | HistoryInputData): The history data to add.
        """
        history_data = history.history if isinstance(history, History) else history

        for key, values in history_data.items():
            processed_values = self._process_values(key, values)
            if key not in self.history:
                self.history[key] = []
            self.history[key].extend(processed_values)

    def append(self, key: str, value: float, x_value: float | None = None) -> None:
        """Append a single (x, y) data point to a specific metric.

        If `x_value` is None, it assumes the next sequential integer index
        based on the last x-axis values for that metric. If available, it
        infers the step size from the last two points, otherwise defaults to
        1.

        Args:
            key (str): The name of the metric.
            value (float): The y-axis value (e.g., loss, accuracy).
            x_value (float | None, optional): The x-axis value (e.g., epoch,
                iteration). Defaults to None.
        """
        if key not in self.history:
            self.history[key] = []

        if x_value is None:
            x_value = self._get_next_x(key)

        self.history[key].append((float(x_value), float(value)))

    def append_from_dict(
        self, history_dict: dict[str, float | tuple[float, float]]
    ) -> None:
        """Append a dictionary of single values for the current step.

        Keys in the dictionary are metric names.
        Values can be a single float (y-value, x is inferred) or a
        (x-value, y-value) tuple.

        Args:
            history_dict (dict[str, float | tuple[float, float]]): The
                dictionary of metrics to add.

        Raises:
            ValueError: If a value in the dictionary has an invalid format.
        """
        for key, value in history_dict.items():
            if isinstance(value, int | float):
                self.append(key, float(value))
            elif (
                isinstance(value, tuple)
                and len(value) == 2
                and all(isinstance(i, int | float) for i in value)
            ):
                self.append(key, float(value[1]), float(value[0]))
            else:
                raise ValueError(
                    f"Invalid value format for key '{key}' in append_from_dict. "
                    "Expected float or (float, float) tuple."
                )

    def save(self, filepath: str | Path) -> None:
        """Save the entire history to a file.

        Uses pickle for serialization.

        Args:
            filepath (str | Path): The path to the save file.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.history, f)

    @classmethod
    def load(cls, filepath: str | Path) -> History:
        """Load a history from a pickle file.

        Args:
            filepath (str | Path): The path to the file.

        Raises:
            ValueError: If the loaded file has an invalid format.

        Returns:
            History: The loaded History object.
        """
        with open(filepath, "rb") as f:
            loaded_data = pickle.load(f)

        if isinstance(loaded_data, dict):
            return cls(history_data=loaded_data)
        raise ValueError("Invalid history file format.")

    def visualize(
        self,
        metrics: list[str] | None = None,
        x_label: str = "Step",
        y_label: str = "Value",
        title: str = "History",
        log_y: bool = False,
        smoothing_window: int | None = None,
        plot_raw: bool = True,
        ax: Axes | None = None,
    ) -> Figure | SubFigure:
        """Plot the specified time-series metrics.

        Args:
            metrics (list[str] | None, optional): A list of metric names to
                plot. If None, all metrics in `.history` will be plotted.
                Defaults to None.
            x_label (str, optional): Label for the x-axis.
                Defaults to "Step".
            y_label (str, optional): Label for the y-axis.
                Defaults to "Value".
            title (str, optional): Title of the plot.
                Defaults to "History".
            log_y (bool, optional): Whether to use logarithmmic scale for y ax.
                Defaults to False.
            smoothing_window (int | None, optional): The window size for
                rolling average. If None, don't apply smoothing.
                Defaults to None.
            plot_raw (bool, optional): If smoothing, whether to also plot the
                raw (unsmoothed) data with high transparency.
                Defaults to True.
            ax (Axes | None, optional): A matplotlib Axes to plot on.
                If None, a new figure and axis are created.
                Defaults to None.

        Returns:
            Figure | SubFigure: The matplotlib Figure containing the plot.
        """
        fig: Figure | SubFigure
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            subfig = ax.get_figure()
            if subfig is None:
                raise ValueError("The Axes does not belong to a Figure.")
            fig = subfig

        metrics_to_plot = metrics if metrics is not None else self.history.keys()

        for key in metrics_to_plot:
            if key not in self.history or not self.history[key]:
                print(f"Warning: Metric '{key}' not found or is empty, skipping.")
                continue

            try:
                vals = np.array(self.history[key])
                if vals.ndim != 2 or vals.shape[1] != 2:
                    print(f"Warning: Data for '{key}' is malformed, skipping.")
                    continue

                x_data = vals[:, 0]
                y_data = vals[:, 1]
                label = key

                if smoothing_window is not None and smoothing_window > 0:
                    if len(y_data) > smoothing_window:
                        if plot_raw:
                            ax.plot(x_data, y_data, label=f"{key} (raw)", alpha=0.3)
                            label = f"{key} (smoothed)"

                        y_cumsum = np.cumsum(y_data)
                        y_cumsum[smoothing_window:] = (
                            y_cumsum[smoothing_window:] - y_cumsum[:-smoothing_window]
                        )

                        counts = np.arange(1, len(y_data) + 1)
                        counts[smoothing_window:] = smoothing_window

                        y_data = y_cumsum / counts
                    else:
                        print(
                            f"Warning: Not enough data for metric '{key}' "
                            f"to smooth with window {smoothing_window}"
                        )

                ax.plot(x_data, y_data, label=label)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: Could not plot metric '{key}'. Error: {e}")

        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        if log_y:
            ax.set_yscale("log")

        return fig
