"""Select entity to choose operation mode for HA Predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.select import SelectEntity, SelectEntityDescription
from propcache import cached_property

from custom_components.ha_predictions.switch import EntityCategory

from .const import (
    ENTITY_KEY_OPERATION_MODE,
    ENTITY_KEY_SAMPLING_STRATEGY,
    OP_MODE_PROD,
    OP_MODE_TRAIN,
    SAMPLING_NONE,
    SAMPLING_RANDOM,
    SAMPLING_SMOTE,
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
    """Set up the select platform."""
    async_add_entities(
        [
            # SelectModeEntity(
            #     coordinator=entry.runtime_data.coordinator,
            #     entity_description=HAPredictionSelectEntityDescription(
            #         key=ENTITY_KEY_OPERATION_MODE,
            #         name="Mode",
            #         icon="mdi:form-dropdown",
            #     ),
            # ),
            HAPredictionSelectEntity(
                coordinator=entry.runtime_data.coordinator,
                entity_description=HAPredictionSelectEntityDescription(
                    key=ENTITY_KEY_SAMPLING_STRATEGY,
                    name="Sampling Strategy",
                    icon="mdi:form-dropdown",
                    options=[SAMPLING_NONE, SAMPLING_RANDOM, SAMPLING_SMOTE],
                    entity_category=EntityCategory.CONFIG,
                ),
            ),
            HAPredictionSelectEntity(
                coordinator=entry.runtime_data.coordinator,
                entity_description=HAPredictionSelectEntityDescription(
                    key=ENTITY_KEY_OPERATION_MODE,
                    name="Operation Mode",
                    icon="mdi:form-dropdown",
                    options=[OP_MODE_TRAIN, OP_MODE_PROD],
                ),
            ),
        ]
    )


class HAPredictionSelectEntityDescription(SelectEntityDescription):
    """Describe select entity for HA Predictions."""


class HAPredictionSelectEntity(HAPredictionEntity, SelectEntity):
    """Base class for select entities in HA Predictions."""

    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: HAPredictionSelectEntityDescription,
    ) -> None:
        """Initialize the select entity."""
        super().__init__(coordinator)

        self.entity_description: HAPredictionSelectEntityDescription = (
            entity_description
        )
        self.coordinator.register(self)

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        self.coordinator.select_option(self.entity_description.key, option)
        self.schedule_update_ha_state()

    @cached_property
    def unique_id(self) -> str | None:
        """Return a unique ID."""
        return (
            self.coordinator.config_entry.entry_id
            + UNDERSCORE
            + self.entity_description.key
        )

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return True

    @property
    def current_option(self) -> str:
        """Return the current option."""
        option = self.coordinator.get_option(self.entity_description.key)
        if option is not None:
            return option
        self.coordinator.logger.warning(
            "Current option for %s is None, defaulting to first option",
            self.entity_description.key,
        )
        return self.options[0]


class SelectModeEntity(HAPredictionSelectEntity):
    """Select entity to choose operation mode."""

    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: HAPredictionSelectEntityDescription,
    ) -> None:
        """Initialize the select entity."""
        super().__init__(coordinator, entity_description)
        self._attr_options: list[str] = [OP_MODE_TRAIN, OP_MODE_PROD]

    @cached_property
    def unique_id(self) -> str | None:
        """Return a unique ID."""
        return (
            self.coordinator.config_entry.entry_id
            + UNDERSCORE
            + ENTITY_KEY_OPERATION_MODE
        )

    @property
    def current_option(self) -> str:
        """Return the current option."""
        return self.coordinator.operation_mode

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return True

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        self.coordinator.select_option(self.entity_description.key, option)
        self.schedule_update_ha_state()
