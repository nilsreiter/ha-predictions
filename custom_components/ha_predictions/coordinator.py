"""DataUpdateCoordinator for ha_predictions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import (
    CONF_FEATURE_ENTITY,
    CONF_TARGET_ENTITY,
    LOGGER,
    MIN_DATASET_SIZE,
    MSG_DATASET_CHANGED,
    MSG_PREDICTION_MADE,
    MSG_TRAINING_DONE,
    OP_MODE_PROD,
    OP_MODE_TRAIN,
)
from .ml.model import Model

if TYPE_CHECKING:
    from types import NoneType

    from homeassistant.core import Event, EventStateChangedData

    from .data import HAPredictionConfigEntry
    from .entity import HAPredictionEntity


# https://developers.home-assistant.io/docs/integration_fetching_data#coordinated-single-api-poll-for-data-for-all-entities
class HAPredictionUpdateCoordinator(DataUpdateCoordinator):
    """Class to manage training/testing the model."""

    config_entry: HAPredictionConfigEntry
    accuracy: float | NoneType = None
    entity_registry: list[HAPredictionEntity] = []
    dataset: pd.DataFrame | NoneType = None
    dataset_size: int = 0
    model: Model = Model(LOGGER)
    operation_mode: str = OP_MODE_TRAIN
    training_ready: bool = False
    current_prediction: tuple[str, float] | NoneType = None

    async def _async_update_data(self) -> Any:
        """Update data via library."""

    def register(self, entity: HAPredictionEntity):
        self.entity_registry.append(entity)

    def remove_listeners(self) -> None:
        self.entity_registry.clear()

    def set_operation_mode(self, mode: str) -> None:
        if mode != self.operation_mode:
            self.operation_mode = mode
            self.logger.info("Operation mode has been changed to %s", mode)

    def state_changed(self, event: Event[EventStateChangedData]) -> None:
        new_state = event.data["new_state"]
        old_state = event.data["old_state"]

        # Only act if state actually changed
        if old_state and new_state and old_state.state != new_state.state:
            self.logger.info("Detecting changed states, storing current instance.")
            self.collect()
            if self.model.prediction_ready and self.dataset is not None:
                self.logger.info("Making new prediction after state change.")
                instance_data = pd.DataFrame(
                    columns=self.dataset.columns[:-1],
                    data=[self._get_states_for_entities(include_target=False)],
                )
                self.logger.debug(
                    "Instance data for prediction: %s", str(instance_data)
                )
                pred = self.model.predict(instance_data)
                if pred is not None:
                    self.current_prediction = pred
                    self.logger.info("New prediction: %s", str(self.current_prediction))
                    [e.notify(MSG_PREDICTION_MADE) for e in self.entity_registry]

    def _initialize_dataframe(self) -> NoneType:
        self.dataset = pd.DataFrame(
            columns=[
                *list(self.config_entry.data[CONF_FEATURE_ENTITY]),
                self.config_entry.data[CONF_TARGET_ENTITY],
            ]
        )
        self.logger.debug("Initialized new dataframe: %s", str(self.dataset))

    def _get_state_for_entity(self, entity_id: str) -> str | float | NoneType:
        if state := self.hass.states.get(entity_id=entity_id):
            try:
                return float(state.state)
            except ValueError:
                return state.state

        return None

    def _get_states_for_entities(
        self, include_target: bool | NoneType = True
    ) -> list[str | float | NoneType]:
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

    def read_table(self) -> NoneType:
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

    def store_table(self, df: pd.DataFrame | NoneType):
        self.config_entry.runtime_data.datafile.parent.mkdir(
            parents=True, exist_ok=True
        )
        if df is not None:
            df.to_csv(self.config_entry.runtime_data.datafile, index=False)

    def collect(self) -> NoneType:
        """Collect the current situation as a new data point."""
        xy = self._get_states_for_entities()
        if self.dataset is None:
            self._initialize_dataframe()
        if self.dataset is not None:
            self.dataset.loc[len(self.dataset)] = xy
            self.dataset_size = self.dataset.shape[0]
            self.logger.info(self.dataset)
        [e.notify(MSG_DATASET_CHANGED) for e in self.entity_registry]
        self.training_ready = self.dataset_size >= MIN_DATASET_SIZE

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

        if self.operation_mode == OP_MODE_TRAIN:
            await self.hass.async_add_executor_job(
                self.model.train_eval, self.dataset.copy()
            )
            self.accuracy = self.model.accuracy
            self.logger.info("Training complete, accuracy: %f", self.accuracy)
        elif self.operation_mode == OP_MODE_PROD:
            await self.hass.async_add_executor_job(
                self.model.train_final, self.dataset.copy()
            )
        else:
            self.logger.error("Unknown operation mode: %s", self.operation_mode)
            return
        [e.notify(MSG_TRAINING_DONE) for e in self.entity_registry]
