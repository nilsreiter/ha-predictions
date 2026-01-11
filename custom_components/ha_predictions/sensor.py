"""Sensor platform for integration_blueprint."""

from __future__ import annotations

from types import NoneType
from typing import TYPE_CHECKING, Union

from homeassistant.components.sensor import SensorEntity, SensorEntityDescription

from .const import (
    CONF_FEATURE_ENTITY,
    ET_PERFORMANCE_SENSOR,
    MSG_PREDICTION_MADE,
    MSG_TRAINING_DONE,
    MSG_DATASET_CHANGED,
    MIN_DATASET_SIZE,
)
from .entity import HAPredictionEntity

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

    from .coordinator import HAPredictionUpdateCoordinator
    from .data import HAPredictionConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,  # noqa: ARG001 Unused function argument: `hass`
    entry: HAPredictionConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sensor platform."""
    async_add_entities(
        [
            PredictionPerformanceSensor(
                coordinator=entry.runtime_data.coordinator,
                entity_description=SensorEntityDescription(
                    key="prediction_performance",
                    name="Prediction Performance",
                    icon="mdi:percent-box-outline",
                    suggested_display_precision=1,
                    native_unit_of_measurement="%",
                ),
            ),
            DatasetSensor(
                coordinator=entry.runtime_data.coordinator,
                entity_description=SensorEntityDescription(
                    key="dataset_size",
                    name="Dataset size",
                    icon="mdi:database",
                    suggested_display_precision=0,
                ),
            ),
            CurrentPredictionSensor(
                coordinator=entry.runtime_data.coordinator,
                entity_description=SensorEntityDescription(
                    key="current_prediction",
                    name="Current Prediction",
                    icon="mdi:lightbulb-on",
                ),
            ),
        ]
    )


class DatasetSensor(HAPredictionEntity, SensorEntity):
    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: SensorEntityDescription,
    ):
        super().__init__(coordinator)
        self.entity_description = entity_description
        self._attr_unique_id = self.coordinator.config_entry.entry_id + "-dataset-size"
        coordinator.register(self)

    @property
    def native_value(self) -> float | None:
        """Return the native value of the sensor."""
        if self.coordinator.dataset_size is not None:
            return self.coordinator.dataset_size
        return None

    @property
    def should_poll(self) -> bool:
        return False

    @property
    def extra_state_attributes(self) -> dict[str, str | list | float | NoneType]:
        """Return the state attributes of the sensor."""
        return {
            "minimal_dataset_size": MIN_DATASET_SIZE,
            "feature_entities": list(
                self.coordinator.config_entry.data.get(CONF_FEATURE_ENTITY, [])
            ),
        }

    def notify(self, msg: str) -> None:
        if msg is MSG_DATASET_CHANGED:
            self.schedule_update_ha_state()


class CurrentPredictionSensor(HAPredictionEntity, SensorEntity):
    prediction_label: str | NoneType = None
    prediction_probability: float | NoneType = None

    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: SensorEntityDescription,
    ):
        super().__init__(coordinator)
        self.entity_description = entity_description
        self._attr_unique_id = (
            self.coordinator.config_entry.entry_id + "-current-prediction"
        )
        coordinator.register(self)

    @property
    def native_value(self) -> str | None:
        """Return the native value of the sensor."""
        # Implement logic to return the current prediction value
        return self.prediction_label

    @property
    def extra_state_attributes(self) -> dict[str, float | NoneType]:
        """Return the state attributes of the sensor."""
        return {
            "probability": self.prediction_probability,
        }

    @property
    def available(self) -> bool:
        return self.prediction_label is not None

    @property
    def should_poll(self) -> bool:
        return False

    def notify(self, msg: str) -> None:
        if (
            msg is MSG_PREDICTION_MADE
            and self.coordinator.current_prediction is not None
        ):
            self.prediction_label = self.coordinator.current_prediction[0]
            self.prediction_probability = self.coordinator.current_prediction[1]
            self.schedule_update_ha_state()


class PredictionPerformanceSensor(HAPredictionEntity, SensorEntity):
    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: SensorEntityDescription,
    ):
        super().__init__(coordinator)
        self.entity_description = entity_description
        self._attr_unique_id = self.coordinator.config_entry.entry_id + "-accuracy"
        coordinator.register(self)

    @property
    def native_value(self) -> float | None:
        """Return the native value of the sensor."""
        if self.coordinator.accuracy is not None:
            return self.coordinator.accuracy * 100
        return None

    @property
    def available(self) -> bool:
        return self.coordinator.accuracy is not None

    @property
    def should_poll(self) -> bool:
        return False

    def notify(self, msg: str) -> None:
        if msg is MSG_TRAINING_DONE:
            self.schedule_update_ha_state()
