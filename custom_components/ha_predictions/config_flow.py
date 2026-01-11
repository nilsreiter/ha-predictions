"""Adds config flow for Blueprint."""

from __future__ import annotations

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.helpers import selector
from slugify import slugify

from .const import CONF_FEATURE_ENTITY, CONF_TARGET_ENTITY, DOMAIN, LOGGER


# TODO: Make options flow editable after initial setup, allow adding feature entities.
#       Adding a feature entity resets the training data.
class HAPredictionsFlowHandler(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for HA Predictions."""

    VERSION = 2

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
