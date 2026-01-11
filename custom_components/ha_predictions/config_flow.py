"""Adds config flow for Blueprint."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector
from slugify import slugify

from .const import CONF_FEATURE_ENTITY, CONF_TARGET_ENTITY, DOMAIN, OPT_FEATURES_CHANGED


class HAPredictionsFlowHandler(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for HA Predictions."""

    VERSION = 2

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> HAPredictionsOptionsFlowHandler:
        """Get the options flow for this handler."""
        return HAPredictionsOptionsFlowHandler(config_entry)

    async def async_step_user(
        self,
        user_input: dict | None = None,
    ) -> config_entries.ConfigFlowResult:
        """Handle a flow initialized by the user."""
        _errors = {}
        if user_input is not None:
            await self.async_set_unique_id(
                ## Do NOT use this in production code
                ## The unique_id should never be something that can change
                ## https://developers.home-assistant.io/docs/config_entries_config_flow_handler#unique-ids
                unique_id=slugify(user_input[CONF_TARGET_ENTITY])
            )
            self._abort_if_unique_id_configured()

            # Validate entities exist and are of correct type
            target_entity = user_input[CONF_TARGET_ENTITY]
            feature_entities = user_input[CONF_FEATURE_ENTITY]

            # Check if target entity exists
            if self.hass.states.get(target_entity) is None:
                _errors["base"] = "target_entity_not_found"

            # Check if target entity is of correct domain
            if not _errors:
                target_domain = target_entity.split(".")[0]
                if target_domain not in ["light", "switch", "input_boolean"]:
                    _errors["base"] = "target_entity_wrong_domain"

            # Check if all feature entities exist
            if not _errors:
                for feature_entity in feature_entities:
                    if self.hass.states.get(feature_entity) is None:
                        _errors["base"] = "feature_entity_not_found"
                        break

            # If no errors, create the entry
            if not _errors:
                return self.async_create_entry(
                    title="Prediction for " + user_input[CONF_TARGET_ENTITY],
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_TARGET_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            filter=[{"domain": ["light", "switch", "input_boolean"]}]
                        ),
                    ),
                    vol.Required(CONF_FEATURE_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(multiple=True),
                    ),
                },
            ),
            errors=_errors,
        )


class HAPredictionsOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for HA Predictions."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            # Check if feature entities have changed
            old_features = set(self.config_entry.data.get(CONF_FEATURE_ENTITY, []))
            new_features = set(user_input.get(CONF_FEATURE_ENTITY, []))
            features_changed = old_features != new_features

            # Update the config entry with new data
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data={
                    **self.config_entry.data,
                    CONF_FEATURE_ENTITY: user_input[CONF_FEATURE_ENTITY],
                },
                options={
                    **self.config_entry.options,
                    OPT_FEATURES_CHANGED: features_changed,
                },
            )

            return self.async_create_entry(title="", data={})

        # Get current feature entities from config entry
        current_features = self.config_entry.data.get(CONF_FEATURE_ENTITY, [])

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_FEATURE_ENTITY,
                        default=current_features,
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(multiple=True),
                    ),
                }
            ),
        )
