"""Switch to enable/disable z-score normalization for training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.const import EntityCategory
from slugify import slugify

from custom_components.ha_predictions.const import ENTITY_KEY_ZSCORES_SWITCH

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
    """Set up the switch platform."""
    async_add_entities(
        ZScoreSwitch(
            coordinator=entry.runtime_data.coordinator,
            entity_description=entity_description,
        )
        for entity_description in [
            SwitchEntityDescription(
                key="ml_setting_switch_zscores",
                name="Use Z-Score Normalization",
                icon="mdi:math-norm-box",
                entity_category=EntityCategory.CONFIG,
            )
        ]
    )


class ZScoreSwitch(HAPredictionEntity, SwitchEntity):
    """Switch to enable/disable z-score normalization for training."""

    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: SwitchEntityDescription,
    ) -> None:
        """Initialize the switch class."""
        super().__init__(coordinator)
        self.entity_description = entity_description
        self._attr_unique_id = slugify(
            self.coordinator.config_entry.runtime_data.target_entity_name
            + ENTITY_KEY_ZSCORES_SWITCH
        )

    @property
    def is_on(self) -> bool:
        """Return true if the switch is on."""
        return "zscores" in self.coordinator.model.transformations

    def turn_on(self, **_: Any) -> None:
        """Turn on the switch."""
        self._attr_is_on = True
        self.coordinator.set_zscores(value=True)
        self.schedule_update_ha_state()

    def turn_off(self, **_: Any) -> None:
        """Turn off the switch."""
        self._attr_is_on = False
        self.coordinator.set_zscores(value=False)
        self.schedule_update_ha_state()
