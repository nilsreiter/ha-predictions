# HA Predictions

[![GitHub Release][releases-shield]][releases]
[![GitHub Activity][commits-shield]][commits]
[![License][license-shield]](LICENSE)
[![hacs][hacsbadge]][hacs]

A Home Assistant custom integration that uses machine learning to predict entity states based on feature entities. Train models to predict when lights turn on/off, switches change state, or other automations trigger based on the state of other entities in your home.

> ⚠️ **Alpha Stage**: This integration is currently in alpha development. Features may change, and there may be bugs or incomplete functionality. Use at your own risk and please report any issues you encounter.

## Features

- 🤖 **Machine Learning Integration**: Uses logistic regression to learn patterns from your Home Assistant entities
- 📊 **Training Mode**: Collect data from your entities to build prediction models
- 🎯 **Production Mode**: Make real-time predictions based on trained models
- 📈 **Performance Monitoring**: Track model accuracy and dataset size
- 🔄 **Flexible Configuration**: Choose target entity and any number of feature entities
- 💾 **Persistent Storage**: Training data is saved and persists across restarts

## Installation

### HACS (Recommended)

1. Open HACS in Home Assistant
2. Go to "Integrations"
3. Click the three dots in the top right corner
4. Select "Custom repositories"
5. Add this repository URL: `https://github.com/nilsreiter/ha_predictions`
6. Select category "Integration"
7. Click "Add"
8. Search for "HA Predictions" and install

### Manual Installation

1. Using the tool of your choice, open the directory (folder) for your Home Assistant configuration (where you find `configuration.yaml`)
2. If you do not have a `custom_components` directory, create it
3. In the `custom_components` directory, create a new folder called `ha_predictions`
4. Download all the files from the `custom_components/ha_predictions/` directory in this repository
5. Place the files in the new `ha_predictions` directory
6. Restart Home Assistant

## Configuration

### Adding the Integration

1. Go to **Settings** → **Devices & Services**
2. Click **+ Add Integration**
3. Search for **HA Predictions**
4. Follow the configuration steps:
   - **Target Entity**: Select the entity you want to predict (e.g., a light or switch)
   - **Feature Entities**: Select one or more entities to use as features for prediction (e.g., motion sensors, time-based sensors, other lights)

### Changing Feature Entities

You can modify the feature entities after initial setup:

1. Go to **Settings** → **Devices & Services**
2. Find the **HA Predictions** integration
3. Click **Configure**
4. Update the feature entities

⚠️ **Note**: Changing feature entities will reset your training data, and you'll need to collect new data and retrain the model.

## Usage

Once installed and configured, the integration creates several entities:

### Sensors

- **Prediction Performance**: Shows the accuracy of your trained model as a percentage
- **Dataset Size**: Displays the number of training samples collected
- **Current Prediction**: Shows the predicted state and confidence level

### Buttons

- **Store Instance**: Manually save the current state of all configured entities as a training sample
- **Run Training**: Train the model using collected data (requires at least 10 samples)

### Select

- **Mode**: Switch between operation modes:
  - **TRAINING**: Automatically collect data when entities change state
  - **PRODUCTION**: Use the trained model to make predictions

## Workflow

### 1. Training Phase

1. Set the **Mode** selector to **TRAINING**
2. Let your home operate normally for a period of time (days or weeks work best)
3. The integration will automatically collect data when the target entity or feature entities change state
4. You can also manually store instances using the **Store Instance** button
5. Monitor the **Dataset Size** sensor to see how many samples have been collected
6. Once you have at least 10 samples, the **Run Training** button will become available

### 2. Training the Model

1. Click the **Run Training** button
2. The integration will train a logistic regression model using your collected data
3. Check the **Prediction Performance** sensor to see the model's accuracy

### 3. Making Predictions

1. Set the **Mode** selector to **PRODUCTION**
2. The integration will now make predictions based on the current state of your feature entities
3. View predictions in the **Current Prediction** sensor
4. The sensor shows both the predicted state and the confidence level

## Example Use Cases

- **Presence Prediction**: Predict when someone will arrive home based on time of day, day of week, and other sensors
- **Lighting Automation**: Predict when lights should turn on based on motion sensors, ambient light, and time
- **Climate Control**: Predict when heating/cooling should activate based on occupancy and external temperature
- **Energy Management**: Predict high-usage periods based on historical patterns and current conditions

## Requirements

- Home Assistant 2025.2.4 or later
- Python packages (automatically installed):
  - pandas
  - numpy

## Development

This integration is based on the [integration_blueprint](https://github.com/ludeeus/integration_blueprint) template.

For development setup:
1. Clone this repository
2. Open in Visual Studio Code with Dev Containers
3. Run `scripts/develop` to start a development Home Assistant instance

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details on contributing.

## Support

- [Report bugs and request features](https://github.com/nilsreiter/ha_predictions/issues)
- [Documentation](https://github.com/nilsreiter/ha_predictions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- Created by [@nilsreiter](https://github.com/nilsreiter)
- Based on [integration_blueprint](https://github.com/ludeeus/integration_blueprint)

---

[releases-shield]: https://img.shields.io/github/release/nilsreiter/ha_predictions.svg?style=for-the-badge
[releases]: https://github.com/nilsreiter/ha_predictions/releases
[commits-shield]: https://img.shields.io/github/commit-activity/y/nilsreiter/ha_predictions.svg?style=for-the-badge
[commits]: https://github.com/nilsreiter/ha_predictions/commits/main
[license-shield]: https://img.shields.io/github/license/nilsreiter/ha_predictions.svg?style=for-the-badge
[hacs]: https://github.com/hacs/integration
[hacsbadge]: https://img.shields.io/badge/HACS-Custom-orange.svg?style=for-the-badge
