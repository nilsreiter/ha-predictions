"""DataUpdateCoordinator for ha_predictions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import (
    CONF_FEATURE_ENTITY,
    CONF_TARGET_ENTITY,
    ENTITY_KEY_OPERATION_MODE,
    ENTITY_KEY_SAMPLING_STRATEGY,
    MIN_DATASET_SIZE,
    MSG_DATASET_CHANGED,
    MSG_PREDICTION_MADE,
    MSG_TRAINING_DONE,
    MSG_TRAINING_SETTINGS_CHANGED,
    SAMPLING_NONE,
    SAMPLING_RANDOM,
    SAMPLING_SMOTE,
    OperationMode,
)
from .ml.exceptions import ModelNotTrainedError
from .ml.model import Model, SamplingStrategy

if TYPE_CHECKING:
    from types import NoneType

    from homeassistant.core import Event, EventStateChangedData

    from .data import HAPredictionConfigEntry
    from .entity import HAPredictionEntity


class HAPredictionUpdateCoordinator(DataUpdateCoordinator):
    """Class to manage training/testing the model."""

    config_entry: HAPredictionConfigEntry

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the coordinator."""
        super().__init__(*args, **kwargs)
        # Initialize instance variables to avoid sharing between coordinator instances
        self.accuracy: float | NoneType = None
        self.entity_registry: list[HAPredictionEntity] = []
        # TODO: Use only numpy array as dataset representation
        self.dataset: pd.DataFrame | NoneType = None
        self.dataset_size: int = 0
        self.model: Model = Model(self.logger)
        self.operation_mode: OperationMode = OperationMode.TRAINING
        self.training_ready: bool = False
        self.current_prediction: tuple[str, float] | NoneType = None

    async def _async_update_data(self) -> Any:
        """Update data via library."""

    def register(self, entity: HAPredictionEntity) -> None:
        """Register an entity to be notified on changes."""
        self.entity_registry.append(entity)

    def remove_listeners(self) -> None:
        """Remove all listeners."""
        self.entity_registry.clear()

    def set_zscores(self, *, value: bool) -> NoneType:
        """En- or disable the use of zscores for training."""
        if value and "zscores" not in self.model.transformations:
            self.model.transformations["zscores"] = {}
            [e.notify(MSG_TRAINING_SETTINGS_CHANGED) for e in self.entity_registry]
            self.logger.info("Z-Score normalization enabled.")
        elif not value and "zscores" in self.model.transformations:
            self.model.transformations.pop("zscores", None)
            [e.notify(MSG_TRAINING_SETTINGS_CHANGED) for e in self.entity_registry]
            self.logger.info("Z-Score normalization disabled.")

    def get_option(self, key: str) -> str | NoneType:
        """Get the current option for a given key."""
        if key == ENTITY_KEY_OPERATION_MODE:
            return self.operation_mode.name
        if key == ENTITY_KEY_SAMPLING_STRATEGY:
            if "sampling" not in self.model.transformations:
                return SAMPLING_NONE
            sampling_type = self.model.transformations["sampling"]["type"]
            if sampling_type == SamplingStrategy.RANDOM_OVER:
                return SAMPLING_RANDOM
            if sampling_type == SamplingStrategy.SMOTE:
                return SAMPLING_SMOTE
            self.logger.warning(
                "Unknown sampling strategy '%s', defaulting to '%s'",
                sampling_type,
                SAMPLING_NONE,
            )
            return SAMPLING_NONE
        return None

    def select_option(self, key: str, value: str) -> NoneType:
        """Change the selected option."""
        if key == ENTITY_KEY_OPERATION_MODE:
            self._set_operation_mode(OperationMode[value])
        elif key == ENTITY_KEY_SAMPLING_STRATEGY:
            self._set_sampling_strategy(value)

    def _set_operation_mode(self, mode: OperationMode) -> None:
        """Set the operation mode."""
        if mode != self.operation_mode:
            self.operation_mode = mode
            self.logger.info("Operation mode has been changed to %s", mode)

    def _set_sampling_strategy(self, strategy: str) -> NoneType:
        """Set the sampling strategy for handling imbalanced datasets."""
        if strategy == SAMPLING_RANDOM:
            self.model.transformations["sampling"] = {
                "type": SamplingStrategy.RANDOM_OVER
            }
            [e.notify(MSG_TRAINING_SETTINGS_CHANGED) for e in self.entity_registry]
            self.logger.info("Random oversampling enabled.")
        elif strategy == SAMPLING_SMOTE:
            self.model.transformations["sampling"] = {
                "type": SamplingStrategy.SMOTE,
                "k_neighbors": 5,
            }
            [e.notify(MSG_TRAINING_SETTINGS_CHANGED) for e in self.entity_registry]
            self.logger.info("SMOTE enabled.")
        else:
            self.model.transformations.pop("sampling", None)
            [e.notify(MSG_TRAINING_SETTINGS_CHANGED) for e in self.entity_registry]
            self.logger.info("Sampling disabled.")

    def state_changed(self, event: Event[EventStateChangedData]) -> None:
        """Handle state changes of monitored entities."""
        new_state = event.data["new_state"]
        old_state = event.data["old_state"]

        # Only act if state actually changed
        if old_state and new_state and old_state.state != new_state.state:
            self.logger.info("Detecting changed states, storing current instance.")
            # Schedule data collection and prediction in executor to avoid
            # blocking event loop
            self.hass.async_create_task(self._async_collect_and_predict())

    async def _async_collect_and_predict(self) -> None:
        """
        Collect data and make prediction.

        Runs blocking operations in executor to avoid blocking event loop.
        """
        # Run blocking operations in executor
        await self.hass.async_add_executor_job(self._collect_data)

        # Notify entities on main event loop
        for entity in self.entity_registry:
            entity.notify(MSG_DATASET_CHANGED)

        # Make prediction if model is ready
        if self.model.prediction_ready and self.dataset is not None:
            self.logger.info("Making new prediction after state change.")
            await self._async_make_prediction()

    async def async_collect(self) -> None:
        """
        Collect the current state as a new data point.

        This is an async wrapper that runs blocking operations in executor.
        """
        # Run blocking operations in executor
        await self.hass.async_add_executor_job(self._collect_data)

        # Notify entities on main event loop
        for entity in self.entity_registry:
            entity.notify(MSG_DATASET_CHANGED)

    def _collect_data(self) -> None:
        """
        Collect the current state (blocking operations).

        Runs in executor to avoid blocking event loop.
        """
        xy = self._get_states_for_entities()
        if self.dataset is None:
            self._initialize_dataframe()
        if self.dataset is not None:
            self.dataset.loc[len(self.dataset)] = xy
            self.dataset_size = self.dataset.shape[0]
            self.logger.info(self.dataset)
        self.training_ready = self.dataset_size >= MIN_DATASET_SIZE

    async def _async_make_prediction(self) -> None:
        """
        Make a prediction.

        Runs blocking operations in executor to avoid blocking event loop.
        """
        # Run blocking operations in executor
        prediction = await self.hass.async_add_executor_job(self._compute_prediction)

        # Update state and notify entities on main event loop
        if prediction is not None:
            self.current_prediction = prediction
            self.logger.info("New prediction: %s", str(self.current_prediction))
            for entity in self.entity_registry:
                entity.notify(MSG_PREDICTION_MADE)

    def _compute_prediction(self) -> tuple[str, float] | None:
        """
        Compute prediction (blocking operations).

        Runs in executor to avoid blocking event loop.
        """
        try:
            return self.model.predict(
                np.array(self._get_states_for_entities(include_target=False)).reshape(
                    1, -1
                )
            )
        except ModelNotTrainedError as e:
            self.logger.warning(e)
            return None

    def _initialize_dataframe(self) -> NoneType:
        """Initialize empty dataframe for dataset."""
        self.dataset = pd.DataFrame(
            columns=[
                *list(self.config_entry.data[CONF_FEATURE_ENTITY]),
                self.config_entry.data[CONF_TARGET_ENTITY],
            ]
        )
        self.logger.debug("Initialized new dataframe: %s", str(self.dataset))

    def _get_state_for_entity(self, entity_id: str) -> str | float | NoneType:
        """Get the state for a given entity_id."""
        if state := self.hass.states.get(entity_id=entity_id):
            try:
                return float(state.state)
            except ValueError:
                return state.state

        return None

    def _get_states_for_entities(
        self, *, include_target: bool | NoneType = True
    ) -> list[str | float | NoneType]:
        """Get the states for all monitored entities."""
        features = [
            self._get_state_for_entity(e)
            for e in self.config_entry.data[CONF_FEATURE_ENTITY]
        ]
        if include_target:
            return [
                *features,
                self._get_state_for_entity(
                    entity_id=self.config_entry.data[CONF_TARGET_ENTITY]
                ),
            ]
        return features

    # TODO: Check that this doesn't block the event loop
    def read_table(self) -> NoneType:
        """Read dataset from file."""
        self.logger.info(
            "Reading dataset from file: %s",
            str(self.config_entry.runtime_data.datafile),
        )
        if Path.exists(self.config_entry.runtime_data.datafile):
            self.dataset = pd.read_csv(
                self.config_entry.runtime_data.datafile, header=0
            )
            self.dataset_size = self.dataset.shape[0]
            [e.notify(MSG_DATASET_CHANGED) for e in self.entity_registry]

    # TODO: handle possible IO errors
    # TODO: storing on disk should happend regularly in the background
    def store_table(self, df: pd.DataFrame | NoneType) -> None:
        """Store dataset to file."""
        self.config_entry.runtime_data.datafile.parent.mkdir(
            parents=True, exist_ok=True
        )
        if df is not None:
            df.to_csv(self.config_entry.runtime_data.datafile, index=False)

    async def train(self) -> NoneType:
        """Run the training process."""
        self.logger.info("training")

        # Store and read table on/from disk
        await self.hass.async_add_executor_job(self.store_table, self.dataset)
        await self.hass.async_add_executor_job(self.read_table)

        # Run actual training
        if self.dataset is None or not self.training_ready:
            self.logger.warning(
                "Not enough data points collected yet, need at least %i, have %d",
                MIN_DATASET_SIZE,
                self.dataset_size,
            )
            return

        # Convert DataFrame to numpy array
        data_numpy = self.dataset.copy().to_numpy()

        if self.operation_mode == OperationMode.TRAINING:
            await self.hass.async_add_executor_job(self.model.train_eval, data_numpy)
            self.accuracy = self.model.accuracy
            self.logger.info("Training complete, accuracy: %f", self.accuracy)
        elif self.operation_mode == OperationMode.PRODUCTION:
            await self.hass.async_add_executor_job(self.model.train_final, data_numpy)
        else:
            self.logger.error("Unknown operation mode: %s", self.operation_mode)
            return
        [e.notify(MSG_TRAINING_DONE) for e in self.entity_registry]

        if self.operation_mode == OperationMode.PRODUCTION:
            # Update prediction after training
            await self._async_make_prediction()
