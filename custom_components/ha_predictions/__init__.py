"""
Custom integration to integrate integration_blueprint with Home Assistant.

For more details about this integration, please refer to
https://github.com/nilsreiter/ml
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from homeassistant.const import Platform
from homeassistant.core import (
    callback,
)
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.loader import async_get_loaded_integration

from .const import (
    CONF_FEATURE_ENTITY,
    CONF_TARGET_ENTITY,
    DOMAIN,
    LOGGER,
    OPT_FEATURES_CHANGED,
)
from .coordinator import HAPredictionUpdateCoordinator
from .data import HAPredictionData

if TYPE_CHECKING:
    from homeassistant.core import (
        Event,
        EventStateChangedData,
        HomeAssistant,
    )

    from .data import HAPredictionConfigEntry

PLATFORMS: list[Platform] = [
    Platform.BUTTON,
    Platform.SENSOR,
    Platform.SELECT,
    #    Platform.BINARY_SENSOR,
    #    Platform.SWITCH,
]


# https://developers.home-assistant.io/docs/config_entries_index/#setting-up-an-entry
async def async_setup_entry(
    hass: HomeAssistant,
    entry: HAPredictionConfigEntry,
) -> bool:
    """Set up this integration using UI."""
    coordinator = HAPredictionUpdateCoordinator(
        hass,
        logger=LOGGER,
        name=DOMAIN,
        update_interval=timedelta(hours=1),
    )

    entities_for_subscription = [entry.data[CONF_TARGET_ENTITY]] + entry.data[
        CONF_FEATURE_ENTITY
    ]

    @callback
    def state_changed(event: Event[EventStateChangedData]) -> None:
        coordinator.state_changed(event)

    unsub = async_track_state_change_event(
        hass, entity_ids=entities_for_subscription, action=state_changed
    )

    target_entity_state = hass.states.get(entry.data[CONF_TARGET_ENTITY])
    if target_entity_state is not None:
        suggested_area = target_entity_state.attributes.get("area_id")
    else:
        suggested_area = None
    entry.runtime_data = HAPredictionData(
        integration=async_get_loaded_integration(hass, entry.domain),
        coordinator=coordinator,
        datafile=Path(hass.config.config_dir, DOMAIN, entry.entry_id + ".csv"),
        unsubscribe=[unsub],
        device_info=DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=f"HA Predictions {entry.title}",
            manufacturer="Nils Reiter",
            model="HA Predictions Integration",
            suggested_area=suggested_area,
        ),
    )
    await hass.async_add_executor_job(coordinator.read_table)
    await coordinator.async_config_entry_first_refresh()
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    LOGGER.info("Setup complete")
    LOGGER.debug(
        "Subscribed to state changes for entities: %s", entities_for_subscription
    )
    return True


async def async_unload_entry(
    hass: HomeAssistant,
    entry: HAPredictionConfigEntry,
) -> bool:
    """Handle removal of an entry."""
    [unsub() for unsub in entry.runtime_data.unsubscribe]
    entry.runtime_data.coordinator.remove_listeners()
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_reload_entry(
    hass: HomeAssistant,
    entry: HAPredictionConfigEntry,
) -> None:
    """Reload config entry."""
    # Check if we need to reset training data due to feature entity changes
    if entry.options.get(OPT_FEATURES_CHANGED, False):
        # Delete the training data file if it exists
        if entry.runtime_data:
            datafile = entry.runtime_data.datafile
            LOGGER.info(
                "Feature entities changed, deleting training data file: %s",
                datafile,
            )
            # Use lambda to pass missing_ok parameter
            await hass.async_add_executor_job(lambda: datafile.unlink(missing_ok=True))

        # Clear the features_changed flag after processing
        hass.config_entries.async_update_entry(
            entry,
            options={**entry.options, OPT_FEATURES_CHANGED: False},
        )

    await hass.config_entries.async_reload(entry.entry_id)
