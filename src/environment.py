from typing import Dict, Tuple, Optional, Any
import numpy as np
import gym
from gym import spaces
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *
import pandas as pd
import logging
from datetime import datetime
import asyncio

from .data_manager import MarketDataManager
from .risk_manager import RiskManager
from .utils.logger import get_logger
from .utils.metrics import EnvironmentMetrics

logger = get_logger(__name__)

class Position:
    def __init__(self, entry_price: float, quantity: float, side: str):
        self.entry_price = entry_price
        self.quantity = quantity
        self.side = side
        self.entry_time = datetime.utcnow()
        self.orders: Dict[str, Dict] = {}  # Stores related orders (stop loss, take profit)

class TradingEnvironment(gym.Env):
    def __init__(
        self,
        client: Client,
        market_data_manager: MarketDataManager,
        risk_manager: RiskManager,
        config: Dict[str, Any]
    ):
        super(TradingEnvironment, self).__init__()

        # Initialize components
        self.client = client
        self.market_data_manager = market_data_manager
        self.risk_manager = risk_manager
        self.config = config
        self.metrics = EnvironmentMetrics()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: HOLD, 1: BUY, 2: SELL
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(50, 10),  # Timestamp, OHLCV, + technical indicators
            dtype=np.float32
        )

        # Initialize trading state
        self.positions: Dict[str, Position] = {}
        self.balance = 0.0
        self.initial_balance = 0.0

    async def reset(self) -> np.ndarray:
        """Reset the trading environment to initial state."""
        try:
            # Get account balance
            account = await self.client.get_account()
            self.balance = float(
                next(
                    balance['free']
                    for balance in account['balances']
                    if balance['asset'] == self.config['quote_asset']
                )
            )
            self.initial_balance = self.balance

            # Reset positions and metrics
            self.positions.clear()
            self.metrics.reset()

            # Get initial observation
            observation = await self._get_observation()
            return observation

        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            raise

    async def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action (int): 0: HOLD, 1: BUY, 2: SELL
            
        Returns:
            observation (np.ndarray): Market state
            reward (float): Trading reward/profit
            done (bool): Whether episode is finished
            info (dict): Additional information
        """
        try:
            reward = 0.0
            done = False
            info = {}

            # Get current market state
            current_price = await self.market_data_manager.get_current_price()
            
            # Check if trading is allowed
            can_trade, reason = self.risk_manager.can_trade(
                current_price=current_price,
                balance=self.balance
            )
            
            if not can_trade:
                logger.warning(f"Trading restricted: {reason}")
                return await self._get_observation(), 0, False, {"reason": reason}

            # Execute trading action
            if action == 1:  # BUY
                reward = await self._execute_buy(current_price)
            elif action == 2:  # SELL
                reward = await self._execute_sell(current_price)

            # Update positions and check for stop loss/take profit
            await self._update_positions(current_price)

            # Check if episode should end
            done = await self._check_episode_end()

            # Get new observation
            observation = await self._get_observation()

            # Update metrics
            self.metrics.update(
                action=action,
                reward=reward,
                balance=self.balance,
                position_count=len(self.positions)
            )

            info = {
                "balance": self.balance,
                "positions": len(self.positions),
                "current_price": current_price,
                "action": action,
                "reward": reward
            }

            return observation, reward, done, info

        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return await self._get_observation(), -1, True, {"error": str(e)}
        except Exception as e:
            logger.error(f"Error in trading step: {e}")
            return await self._get_observation(), -1, True, {"error": str(e)}

    async def _execute_buy(self, current_price: float) -> float:
        """Execute buy order with position sizing and risk management."""
        try:
            # Calculate position size
            quantity = self.risk_manager.calculate_position_size(
                balance=self.balance,
                current_price=current_price
            )

            # Place market buy order
            order = await self._place_order(
                side=SIDE_BUY,
                quantity=quantity,
                current_price=current_price
            )

            if order:
                # Create new position
                position = Position(
                    entry_price=float(order['price']),
                    quantity=float(order['executedQty']),
                    side=SIDE_BUY
                )
                self.positions[order['orderId']] = position

                # Place stop loss and take profit orders
                await self._place_stop_loss(order['orderId'], current_price)
                await self._place_take_profit(order['orderId'], current_price)

                return 0  # Initial reward for opening position is 0

        except Exception as e:
            logger.error(f"Error executing buy order: {e}")
            return -1

    async def _execute_sell(self, current_price: float) -> float:
        """Execute sell order and calculate profit/loss."""
        try:
            total_reward = 0
            positions_to_remove = []

            for position_id, position in self.positions.items():
                # Place market sell order
                order = await self._place_order(
                    side=SIDE_SELL,
                    quantity=position.quantity,
                    current_price=current_price
                )

                if order:
                    # Calculate profit/loss
                    profit = (float(order['price']) - position.entry_price) * position.quantity
                    reward = (profit / position.entry_price) * 100
                    total_reward += reward

                    # Cancel any existing stop loss/take profit orders
                    await self._cancel_position_orders(position_id)
                    positions_to_remove.append(position_id)

            # Remove closed positions
            for position_id in positions_to_remove:
                del self.positions[position_id]

            return total_reward

        except Exception as e:
            logger.error(f"Error executing sell order: {e}")
            return -1

    async def _place_order(
        self,
        side: str,
        quantity: float,
        current_price: float
    ) -> Optional[Dict]:
        """Place order with retry logic and slippage protection."""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Check for price slippage
                new_price = await self.market_data_manager.get_current_price()
                slippage = abs(new_price - current_price) / current_price * 100

                if slippage > self.config['max_slippage']:
                    logger.warning(f"Price slippage ({slippage}%) exceeds maximum allowed")
                    return None

                # Place order
                order = await self.client.create_order(
                    symbol=self.config['symbol'],
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity,
                    newOrderRespType='FULL'
                )

                logger.info(f"Order placed: {side} {quantity} @ {current_price}")
                return order

            except BinanceAPIException as e:
                logger.error(f"Order placement error: {e}")
                retry_count += 1
                if retry_count == max_retries:
                    raise
                await asyncio.sleep(1)

    async def _place_stop_loss(self, position_id: str, current_price: float):
        """Place stop loss order for position."""
        position = self.positions[position_id]
        stop_price = current_price * (1 - self.config['stop_loss_pct'] / 100)

        try:
            order = await self.client.create_order(
                symbol=self.config['symbol'],
                side=SIDE_SELL,
                type=ORDER_TYPE_STOP_LOSS_LIMIT,
                quantity=position.quantity,
                price=stop_price * 0.999,  # Slight buffer for execution
                stopPrice=stop_price,
                timeInForce=TIME_IN_FORCE_GTC
            )

            position.orders['stop_loss'] = order
            logger.info(f"Stop loss placed at {stop_price}")

        except Exception as e:
            logger.error(f"Error placing stop loss: {e}")

    async def _place_take_profit(self, position_id: str, current_price: float):
        """Place take profit order for position."""
        position = self.positions[position_id]
        take_profit_price = current_price * (1 + self.config['take_profit_pct'] / 100)

        try:
            order = await self.client.create_order(
                symbol=self.config['symbol'],
                side=SIDE_SELL,
                type=ORDER_TYPE_TAKE_PROFIT_LIMIT,
                quantity=position.quantity,
                price=take_profit_price * 1.001,  # Slight buffer for execution
                stopPrice=take_profit_price,
                timeInForce=TIME_IN_FORCE_GTC
            )

            position.orders['take_profit'] = order
            logger.info(f"Take profit placed at {take_profit_price}")

        except Exception as e:
            logger.error(f"Error placing take profit: {e}")

    async def _cancel_position_orders(self, position_id: str):
        """Cancel all orders associated with a position."""
        try:
            position = self.positions[position_id]
            for order_id in position.orders.values():
                await self.client.cancel_order(
                    symbol=self.config['symbol'],
                    orderId=order_id
                )
        except Exception as e:
            logger.error(f"Error canceling orders: {e}")

    async def _update_positions(self, current_price: float):
        """Update position status and check for stop loss/take profit triggers."""
        for position_id, position in list(self.positions.items()):
            profit_pct = (current_price - position.entry_price) / position.entry_price * 100

            # Check for stop loss
            if profit_pct <= -self.config['stop_loss_pct']:
                await self._execute_sell(current_price)
                logger.info(f"Stop loss triggered for position {position_id}")

            # Check for take profit
            elif profit_pct >= self.config['take_profit_pct']:
                await self._execute_sell(current_price)
                logger.info(f"Take profit triggered for position {position_id}")

    async def _get_observation(self) -> np.ndarray:
        """Get current market state observation."""
        try:
            # Get market data
            df = await self.market_data_manager.get_market_data()
            
            # Select features for observation
            features = df[[
                'close', 'volume', 'sma_20', 'sma_50', 'rsi',
                'macd', 'bb_upper', 'bb_middle', 'bb_lower'
            ]].values

            # Add position indicator
            position_indicator = np.full((len(features), 1), len(self.positions))
            
            return np.column_stack((features, position_indicator))

        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            raise

    async def _check_episode_end(self) -> bool:
        """Check if trading episode should end."""
        # Check for maximum drawdown
        if self.balance <= self.initial_balance * (1 - self.config['max_drawdown'] / 100):
            logger.warning("Maximum drawdown reached")
            return True

        # Check for minimum balance
        if self.balance < self.config['min_balance']:
            logger.warning("Minimum balance reached")
            return True

        # Check for maximum episode length
        if self.metrics.step_count >= self.config['max_steps']:
            logger.info("Maximum steps reached")
            return True

        return False

    async def close(self):
        """Cleanup and close all positions."""
        try:
            # Close all open positions
            current_price = await self.market_data_manager.get_current_price()
            await self._execute_sell(current_price)

            # Cancel all open orders
            for position_id in self.positions:
                await self._cancel_position_orders(position_id)

            logger.info("Environment closed, all positions cleared")

        except Exception as e:
            logger.error(f"Error closing environment: {e}")
            raise
