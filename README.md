# HA Predictions

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
2. Go to "Integrations" → Three dots menu → "Custom repositories"
3. Add this repository URL: `https://github.com/nilsreiter/ha-predictions`
4. Select category "Integration" and click "Add"
5. Search for "HA Predictions" and install

### Manual Installation

1. Copy the `custom_components/ha_predictions/` directory to your Home Assistant `config/custom_components/` folder
2. Restart Home Assistant

## Configuration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for **HA Predictions**
3. Select:
   - **Target Entity**: The entity to predict (e.g., light or switch)
   - **Feature Entities**: Entities to use as prediction features (e.g., sensors, time, other lights)

You can modify feature entities later via **Configure**, but this will reset your training data.

## Usage

The integration creates these entities:

- **Sensors**: Prediction Performance (accuracy %), Dataset Size (sample count), Current Prediction (state + confidence)
- **Buttons**: Store Instance (manual save), Run Training (requires 10+ samples)
- **Mode Selector**: TRAINING (collect data) / PRODUCTION (make predictions)

## Workflow

1. **Training Phase**: Set mode to TRAINING and let your home operate normally for days/weeks. Data is automatically collected. Monitor Dataset Size sensor.
2. **Train Model**: Once you have 10+ samples, click **Run Training**. Check Prediction Performance sensor for accuracy.
3. **Production**: Set mode to PRODUCTION to make real-time predictions based on trained model.
4. **Automation**: Create a Home Assistant automation that triggers when the prediction changes to control your target entity (e.g., switch lights). This is a security measure to ensure predictions don't directly control devices.

## Example Use Cases

- Predict presence/arrivals based on time and sensors
- Automate lighting based on motion, ambient light, and time
- Control climate based on occupancy and temperature
- Manage energy usage based on historical patterns

## Requirements

- Home Assistant 2025.2.4 or later
- Python packages: pandas, numpy (auto-installed)

## Development

Based on [integration_blueprint](https://github.com/ludeeus/integration_blueprint). See [CONTRIBUTING.md](CONTRIBUTING.md) for setup details.

## Support

- [Report bugs and request features](https://github.com/nilsreiter/ha-predictions/issues)
- [Documentation](https://github.com/nilsreiter/ha-predictions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- Created by [@nilsreiter](https://github.com/nilsreiter)
- Based on [integration_blueprint](https://github.com/ludeeus/integration_blueprint)

---

[releases-shield]: https://img.shields.io/github/release/nilsreiter/ha-predictions.svg?style=for-the-badge
[releases]: https://github.com/nilsreiter/ha-predictions/releases
[commits-shield]: https://img.shields.io/github/commit-activity/y/nilsreiter/ha-predictions.svg?style=for-the-badge
[commits]: https://github.com/nilsreiter/ha-predictions/commits/main
[license-shield]: https://img.shields.io/github/license/nilsreiter/ha-predictions.svg?style=for-the-badge
[hacs]: https://github.com/hacs/integration
[hacsbadge]: https://img.shields.io/badge/HACS-Custom-orange.svg?style=for-the-badge
