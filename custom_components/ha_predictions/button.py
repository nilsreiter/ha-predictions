"""Button entities for HA Predictions integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.button import ButtonEntity, ButtonEntityDescription
from slugify import slugify

from custom_components.ha_predictions.const import (
    ENTITY_KEY_RUN_TRAINING,
    ENTITY_KEY_STORE_INSTANCE,
    MSG_DATASET_CHANGED,
    UNDERSCORE,
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
    """Set up the button platform."""
    async_add_entities(
        [
            RunTrainingButton(
                coordinator=entry.runtime_data.coordinator,
                entity_description=ButtonEntityDescription(
                    key=ENTITY_KEY_RUN_TRAINING,
                    name="Run Training",
                    icon="mdi:cog-play",
                ),
            ),
            StoreInstanceButton(
                coordinator=entry.runtime_data.coordinator,
                entity_description=ButtonEntityDescription(
                    key=ENTITY_KEY_STORE_INSTANCE,
                    name="Store Instance",
                    icon="mdi:table-plus",
                ),
            ),
        ]
    )


class StoreInstanceButton(HAPredictionEntity, ButtonEntity):
    """Button entity to store a new instance."""

    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: ButtonEntityDescription,
    ) -> None:
        """Initialize the button entity."""
        super().__init__(coordinator)
        self.entity_description = entity_description
        self._attr_unique_id = slugify(
            self.coordinator.config_entry.runtime_data.target_entity_name
            + UNDERSCORE
            + ENTITY_KEY_STORE_INSTANCE
        )

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.coordinator.async_collect()


class RunTrainingButton(HAPredictionEntity, ButtonEntity):
    """Button entity to run model training."""

    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: ButtonEntityDescription,
    ) -> None:
        """Initialize the button entity."""
        super().__init__(coordinator)
        self.entity_description = entity_description

        self._attr_unique_id = slugify(
            self.coordinator.config_entry.runtime_data.target_entity_name
            + UNDERSCORE
            + ENTITY_KEY_RUN_TRAINING
        )

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.coordinator.train()

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self.coordinator.training_ready

    def notify(self, msg: str) -> None:
        """Handle notifications from the coordinator."""
        if msg == MSG_DATASET_CHANGED:
            self.schedule_update_ha_state()
