"""Constants for integration_blueprint."""

from ctypes.wintypes import MSG
from logging import Logger, getLogger
from re import M

LOGGER: Logger = getLogger(__package__)
DOMAIN = "ha_predictions"
ATTRIBUTION = "Data provided by http://jsonplaceholder.typicode.com/"

ACTION_DATA_ENTITY_ID = "entity_id"

CONF_TARGET_ENTITY = "CONF_TARGET_ENTITY"
CONF_FEATURE_ENTITY = "CONF_FEATURE_ENTITY"

ET_PERFORMANCE_SENSOR = "PERFORMANCE_SENSOR"

MSG_TRAINING_DONE = "TRAINING_DONE"
MSG_DATASET_CHANGED = "DATASET_CHANGED"
MSG_OPERATION_MODE_CHANGED = "OPERATION_MODE_CHANGED"
MSG_PREDICTION_MADE = "PREDICTION_MADE"

OP_MODE_TRAIN = "TRAINING"
OP_MODE_PROD = "PRODUCTION"

MIN_DATASET_SIZE = 10
