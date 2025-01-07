#!/usr/bin/env python3
"""
CLI interface for the trading bot.
"""

import click
import asyncio
import yaml
import os
from pathlib import Path
from typing import Dict, Optional
import sys
from datetime import datetime
import signal
from dotenv import load_dotenv

from trading_bot import TradingBot, setup_logger

logger = setup_logger('trading_cli')

DEFAULT_CONFIG_PATH = 'config/config.yaml'

class TradingBotCLI:
    def __init__(self):
        self.bot: Optional[TradingBot] = None
        self.running = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received, stopping bot...")
        self.running = False
        if self.bot:
            asyncio.create_task(self.bot.stop())

    async def start_bot(self, config_path: str):
        """Start the trading bot with given configuration."""
        try:
            # Load environment variables
            load_dotenv()
            
            # Initialize bot
            self.bot = TradingBot(config_path)
            await self.bot.initialize(
                api_key=os.getenv('BINANCE_API_KEY'),
                api_secret=os.getenv('BINANCE_API_SECRET')
            )
            
            # Start bot
            await self.bot.start()
            self.running = True
            
            # Main loop
            while self.running:
                status = self.bot.get_status()
                logger.info(f"Bot status: {status}")
                await asyncio.sleep(60)
                
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            self.running = False
            raise
        finally:
            if self.bot:
                await self.bot.stop()

@click.group()
def cli():
    """Trading bot command line interface."""
    pass

@cli.command()
@click.option('--config', '-c', default=DEFAULT_CONFIG_PATH, help='Path to configuration file')
def start(config):
    """Start the trading bot."""
    click.echo(f"Starting trading bot with config: {config}")
    
    if not os.path.exists(config):
        click.echo(f"Error: Configuration file not found: {config}")
        sys.exit(1)
        
    bot_cli = TradingBotCLI()
    asyncio.run(bot_cli.start_bot(config))

@cli.command()
@click.option('--config', '-c', default=DEFAULT_CONFIG_PATH, help='Path to configuration file')
def validate(config):
    """Validate configuration file."""
    try:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Validate configuration structure
        required_sections = ['trading', 'risk', 'agent']
        for section in required_sections:
            if section not in config_data:
                click.echo(f"Error: Missing required section '{section}' in config")
                sys.exit(1)
                
        click.echo(f"Configuration file {config} is valid")
        
    except Exception as e:
        click.echo(f"Error validating configuration: {e}")
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', default=DEFAULT_CONFIG_PATH, help='Path to configuration file')
def init(config):
    """Initialize a new configuration file."""
    if os.path.exists(config):
        click.echo(f"Error: Configuration file already exists: {config}")
        if not click.confirm("Do you want to overwrite it?"):
            return
    
    default_config = {
        'trading': {
            'symbol': 'BTCUSDT',
            'base_asset': 'BTC',
            'quote_asset': 'USDT',
            'position_size': 0.01,
            'max_position_size': 0.1
        },
        'risk': {
            'max_drawdown': 10,
            'risk_per_trade': 1,
            'max_trades_per_day': 10
        },
        'agent': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'memory_size': 10000
        }
    }
    
    try:
        os.makedirs(os.path.dirname(config), exist_ok=True)
        with open(config, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        click.echo(f"Created new configuration file: {config}")
        
    except Exception as e:
        click.echo(f"Error creating configuration file: {e}")
        sys.exit(1)

@cli.command()
def status():
    """Check the status of the trading bot."""
    pid_file = Path('.bot.pid')
    
    if not pid_file.exists():
        click.echo("Trading bot is not running")
        return
        
    try:
        pid = int(pid_file.read_text())
        os.kill(pid, 0)  # Check if process exists
        click.echo(f"Trading bot is running (PID: {pid})")
    except (ProcessLookupError, ValueError):
        click.echo("Trading bot is not running (stale PID file)")
        pid_file.unlink()
    except Exception as e:
        click.echo(f"Error checking bot status: {e}")

@cli.command()
@click.option('--config', '-c', default=DEFAULT_CONFIG_PATH, help='Path to configuration file')
def backtest(config):
    """Run backtesting simulation."""
    click.echo("Backtesting feature not implemented yet")
    # TODO: Implement backtesting functionality

def main():
    cli()

if __name__ == '__main__':
    main()
