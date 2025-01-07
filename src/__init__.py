"""
Trading Bot Package
------------------
A reinforcement learning based trading bot for cryptocurrency markets.
"""

import os
from typing import Dict, Optional
import yaml
from pathlib import Path
import logging
from datetime import datetime

# Version information
__version__ = '1.0.0'
__author__ = 'Your Name'
__license__ = 'MIT'

# Package imports
from .environment import TradingEnvironment
from .agent import DQNAgent, AgentConfig
from .risk_manager import RiskManager
from .data_manager import MarketDataManager
from .utils.logger import setup_logger
from .utils.metrics import setup_metrics

class TradingBot:
    """Main trading bot class that orchestrates all components."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize trading bot with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = setup_logger()
        self.metrics = setup_metrics()
        
        # Initialize components
        self.data_manager = None
        self.risk_manager = None
        self.environment = None
        self.agent = None
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults."""
        try:
            if config_path is None:
                config_path = os.path.join(
                    Path(__file__).parent.parent,
                    'config',
                    'config.yaml'
                )
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validate configuration
            self._validate_config(config)
            return config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
            
    def _validate_config(self, config: Dict) -> None:
        """Validate configuration parameters."""
        required_keys = {
            'trading': [
                'symbol',
                'base_asset',
                'quote_asset',
                'position_size',
                'max_position_size'
            ],
            'risk': [
                'max_drawdown',
                'risk_per_trade',
                'max_trades_per_day'
            ],
            'agent': [
                'batch_size',
                'learning_rate',
                'memory_size'
            ]
        }
        
        for section, keys in required_keys.items():
            if section not in config:
                raise ValueError(f"Missing configuration section: {section}")
            
            for key in keys:
                if key not in config[section]:
                    raise ValueError(f"Missing configuration key: {section}.{key}")

    async def initialize(self, api_key: str, api_secret: str) -> None:
        """
        Initialize all trading components.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
        """
        try:
            # Initialize data manager
            self.data_manager = MarketDataManager(
                api_key=api_key,
                api_secret=api_secret,
                config=self.config['trading']
            )
            
            # Initialize risk manager
            self.risk_manager = RiskManager(
                config=self.config,
                initial_balance=await self.data_manager.get_account_balance()
            )
            
            # Initialize trading environment
            self.environment = TradingEnvironment(
                data_manager=self.data_manager,
                risk_manager=self.risk_manager,
                config=self.config['trading']
            )
            
            # Initialize trading agent
            self.agent = DQNAgent(
                state_size=self.environment.observation_space.shape[0],
                action_size=self.environment.action_space.n,
                config=AgentConfig(**self.config['agent'])
            )
            
            self.logger.info("Trading bot initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading bot: {e}")
            raise

    async def start(self) -> None:
        """Start the trading bot."""
        try:
            self.logger.info("Starting trading bot...")
            
            # Start data manager's websocket connection
            await self.data_manager.start_websocket()
            
            # Log initial state
            self.logger.info(f"Initial configuration: {self.config}")
            self.logger.info("Trading bot started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start trading bot: {e}")
            raise

    async def stop(self) -> None:
        """Stop the trading bot and cleanup resources."""
        try:
            self.logger.info("Stopping trading bot...")
            
            # Stop data manager's websocket connection
            await self.data_manager.stop_websocket()
            
            # Close environment and cleanup positions
            await self.environment.close()
            
            # Save agent's model and cleanup
            self.agent.cleanup()
            
            # Final metrics update
            self.metrics.record_shutdown()
            
            self.logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    def get_status(self) -> Dict:
        """Get current status of the trading bot."""
        return {
            'version': __version__,
            'start_time': self.metrics.start_time,
            'uptime': (datetime.now() - self.metrics.start_time).total_seconds(),
            'trading_enabled': True if self.environment else False,
            'agent_status': self.agent.get_model_summary() if self.agent else None,
            'risk_metrics': self.risk_manager.get_risk_report() if self.risk_manager else None
        }

# Convenience imports
from .environment import TradingEnvironment
from .agent import DQNAgent
from .risk_manager import RiskManager
from .data_manager import MarketDataManager

__all__ = [
    'TradingBot',
    'TradingEnvironment',
    'DQNAgent',
    'RiskManager',
    'MarketDataManager',
]
