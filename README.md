# Reinforcement Learning Trading Bot

A production-grade cryptocurrency trading bot that employs deep reinforcement learning to execute trades on the Binance exchange. This system combines advanced machine learning techniques with robust risk management and real-time market analysis to make informed trading decisions.

## Overview

This trading bot implements a Deep Q-Network (DQN) architecture to learn and adapt to market conditions. It features comprehensive risk management, real-time market data analysis, and multiple notification channels for monitoring trading activities. The system is designed for production deployment with emphasis on reliability, safety, and performance.

## Key Features

### Trading Capabilities

- Deep Q-Network (DQN) based decision making
- Real-time market data analysis with technical indicators
- Position sizing and management
- Automated entry and exit strategies
- Support for multiple trading pairs

### Risk Management

- Dynamic position sizing based on account equity
- Maximum drawdown protection
- Daily trade limits
- Comprehensive risk metrics tracking
- Slippage protection
- Correlation analysis

### Technical Analysis

- Multiple timeframe support
- Real-time technical indicator calculation
- Volume analysis
- Price action patterns recognition
- Market volatility assessment

### Monitoring and Notifications

- Real-time performance monitoring
- Multi-channel notifications (Telegram, Email, Discord)
- Prometheus metrics integration
- Detailed logging system
- Trade execution reports

### Production Features

- Asynchronous operation
- Rate limiting and API optimization
- Error handling and recovery
- State persistence
- Resource cleanup

## System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for enhanced performance)
- 4GB RAM minimum (8GB recommended)
- Stable internet connection
- Operating System: Linux (recommended), macOS, or Windows

## Installation

1. Clone the repository:

```bash
git clone https://github.com/artbm/BinanceDQNBot.git
cd trading-bot
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install additional system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev ta-lib

# macOS
brew install ta-lib

# Windows
# Download and install ta-lib from: http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip
```

## Configuration

1. Copy the example configuration files:

```bash
cp config/config.yaml.example config/config.yaml
cp .env.example .env
```

2. Edit the configuration files:

- `config.yaml`: Trading parameters, risk limits, and bot settings
- `.env`: API keys, secrets, and environment variables

3. Configure notification channels (optional):

- Telegram bot token and chat ID
- Email SMTP settings
- Discord webhook URL

## Usage

1. Start the trading bot:

```bash
python -m trading_bot.main
```

2. Monitor the bot:

- Access metrics dashboard: http://localhost:8000
- Check logs in the `logs` directory
- Monitor notifications through configured channels

3. Stop the bot gracefully:

```bash
# Press Ctrl+C or send SIGTERM
```

## Project Structure

```
trading_bot/
├── .env.example
├── .gitignore
├── config
│   ├── init.py
│   ├── config.yaml
│   └── model_config.yaml
├── logs
│   └── .gitkeep
├── models
│   └── .gitkeep
├── notebooks
│   └── backtest.ipynb
├── README.md
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── agent.py
│   ├── data_manager.py
│   ├── environment.py
│   ├── main.py
│   ├── risk_manager.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── metrics.py
│       └── notifications.py ├── tests
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_environment.py
│   └── test_risk_manager.py └── venv
    ├── bin/
    ├── include/
    ├── lib/
    ├── lib64/
    ├── pyvenv.cfg
    └── share/
```

## Safety and Risk Management

This trading bot includes multiple safety measures to protect against excessive losses:

1. Position Sizing: Dynamic calculation based on account equity
2. Stop Loss: Automatic stop-loss orders for all positions
3. Maximum Drawdown: Automatic trading suspension if drawdown limit is reached
4. Daily Limits: Maximum number of trades and position sizes per day
5. Error Recovery: Automatic error handling and position recovery

## Monitoring and Metrics

The system provides comprehensive monitoring through:

1. Prometheus Metrics:

   - Trading performance metrics
   - System resource utilization
   - API request statistics

2. Logging:

   - Detailed activity logs
   - Error tracking
   - Performance monitoring

3. Notifications:
   - Real-time trade notifications
   - Error alerts
   - Performance reports

## Development and Testing

1. Run tests:

```bash
pytest tests/
```

2. Format code:

```bash
black src/
```

3. Type checking:

```bash
mypy src/
```

## Contributing

We welcome contributions to improve the trading bot. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is provided for educational and research purposes only. Trading cryptocurrencies carries significant risk of loss. Use this software at your own risk. The authors and contributors are not responsible for any financial losses incurred through the use of this system.

## Support

For questions and support:

- Open an issue on GitHub
- Join our Discord community
- Contact the maintainers

## Acknowledgments

We thank the following projects and communities:

- Binance API Team
- OpenAI Gym
- PyTorch
- TA-Lib

---

**Note**: Always test the trading bot thoroughly in a testnet environment before deploying it with real funds. Ensure you understand the risks involved in automated trading and cryptocurrency markets.
