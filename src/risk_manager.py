from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from prometheus_client import Counter, Gauge, Histogram
from .utils.logger import get_logger
import threading
import time

logger = get_logger(__name__)

@dataclass
class RiskLimits:
    max_position_size: float
    max_trades_per_day: int
    max_daily_drawdown: float
    max_total_drawdown: float
    risk_per_trade: float
    min_risk_reward_ratio: float
    max_correlation: float
    max_leverage: float
    position_timeout: int
    cooldown_period: int
    max_slippage: float
    min_volume: float
    max_open_positions: int

class RiskMetrics:
    def __init__(self):
        # Prometheus metrics
        self.trade_counter = Counter('trades_total', 'Total number of trades', ['outcome'])
        self.position_size_gauge = Gauge('position_size', 'Current position size')
        self.drawdown_gauge = Gauge('drawdown', 'Current drawdown percentage')
        self.risk_exposure_gauge = Gauge('risk_exposure', 'Current risk exposure')
        self.trade_duration = Histogram('trade_duration_seconds', 'Trade duration')
        
        # Internal metrics
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
        self.positions: Dict[str, Dict] = {}
        
        # Thread safety
        self.metrics_lock = threading.Lock()

class RiskManager:
    def __init__(self, config: Dict, initial_balance: float):
        """
        Initialize the risk manager.
        
        Args:
            config: Risk configuration parameters
            initial_balance: Initial account balance
        """
        self.limits = RiskLimits(**config['risk_limits'])
        self.metrics = RiskMetrics()
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        
        # Position tracking
        self.open_positions: Dict[str, Dict] = {}
        self.position_history: List[Dict] = []
        
        # Time tracking
        self.last_trade_time = None
        self._reset_daily_metrics()

    def _reset_daily_metrics(self):
        """Reset daily trading metrics at UTC midnight."""
        with self.metrics_lock:
            self.metrics.daily_trades = 0
            self.metrics.daily_pnl = 0.0
        
        # Schedule next reset
        now = datetime.utcnow()
        next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        delay = (next_midnight - now).total_seconds()
        threading.Timer(delay, self._reset_daily_metrics).start()

    def can_trade(self, balance: float, current_price: float, volume: float) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk parameters.
        
        Args:
            balance: Current account balance
            current_price: Current asset price
            volume: Current trading volume
            
        Returns:
            Tuple of (can_trade, reason)
        """
        try:
            with self.metrics_lock:
                # Check daily trade limit
                if self.metrics.daily_trades >= self.limits.max_trades_per_day:
                    return False, "Daily trade limit reached"

                # Check cooldown period
                if (self.last_trade_time and 
                    time.time() - self.last_trade_time < self.limits.cooldown_period):
                    return False, "In cooldown period"

                # Check open positions limit
                if len(self.open_positions) >= self.limits.max_open_positions:
                    return False, "Maximum open positions reached"

                # Check volume
                if volume < self.limits.min_volume:
                    return False, "Insufficient volume"

                # Check drawdown limits
                current_drawdown = self._calculate_drawdown(balance)
                if current_drawdown >= self.limits.max_total_drawdown:
                    return False, "Maximum drawdown reached"

                # Check daily drawdown
                if self.metrics.daily_pnl / self.initial_balance * 100 <= -self.limits.max_daily_drawdown:
                    return False, "Maximum daily drawdown reached"

                # Check risk exposure
                total_exposure = sum(pos['size'] * current_price for pos in self.open_positions.values())
                if total_exposure + self.calculate_position_size(balance, current_price) > balance * self.limits.max_leverage:
                    return False, "Maximum risk exposure reached"

                return True, "OK"

        except Exception as e:
            logger.error(f"Error in can_trade check: {e}")
            return False, f"Error in risk check: {str(e)}"

    def calculate_position_size(self, balance: float, current_price: float) -> float:
        """
        Calculate safe position size based on risk parameters.
        
        Args:
            balance: Current account balance
            current_price: Current asset price
            
        Returns:
            Position size in base asset
        """
        try:
            # Calculate position size based on risk per trade
            risk_amount = balance * (self.limits.risk_per_trade / 100)
            
            # Convert to position size using stop loss
            position_size = risk_amount / (current_price * (self.limits.max_daily_drawdown / 100))
            
            # Apply maximum position size limit
            max_size = self.limits.max_position_size
            position_size = min(position_size, max_size)
            
            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            raise

    def validate_order(self, side: str, quantity: float, price: float, 
                      current_price: float) -> Tuple[bool, str]:
        """
        Validate order parameters against risk limits.
        
        Args:
            side: Order side (buy/sell)
            quantity: Order quantity
            price: Order price
            current_price: Current market price
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check slippage
            slippage = abs(price - current_price) / current_price * 100
            if slippage > self.limits.max_slippage:
                return False, f"Slippage ({slippage:.2f}%) exceeds maximum allowed"

            # Check position size
            if quantity > self.limits.max_position_size:
                return False, "Position size exceeds maximum allowed"

            # Additional order validation logic here
            return True, "OK"

        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False, str(e)

    def record_trade(self, trade_id: str, side: str, quantity: float, 
                    price: float, timestamp: float) -> None:
        """
        Record trade for risk tracking.
        
        Args:
            trade_id: Unique trade identifier
            side: Trade side (buy/sell)
            quantity: Trade quantity
            price: Trade price
            timestamp: Trade timestamp
        """
        try:
            with self.metrics_lock:
                if side == 'buy':
                    self.open_positions[trade_id] = {
                        'size': quantity,
                        'entry_price': price,
                        'entry_time': timestamp
                    }
                    self.metrics.position_size_gauge.inc(quantity)
                    
                elif side == 'sell' and trade_id in self.open_positions:
                    position = self.open_positions[trade_id]
                    pnl = (price - position['entry_price']) * quantity
                    duration = timestamp - position['entry_time']
                    
                    # Update metrics
                    self.metrics.daily_pnl += pnl
                    self.metrics.trade_duration.observe(duration)
                    self.metrics.position_size_gauge.dec(quantity)
                    self.metrics.trade_counter.labels(
                        outcome='profit' if pnl > 0 else 'loss'
                    ).inc()
                    
                    # Record trade history
                    self.position_history.append({
                        'trade_id': trade_id,
                        'entry_price': position['entry_price'],
                        'exit_price': price,
                        'quantity': quantity,
                        'pnl': pnl,
                        'duration': duration
                    })
                    
                    del self.open_positions[trade_id]

                self.metrics.daily_trades += 1
                self.last_trade_time = timestamp

        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            raise

    def check_position_timeout(self) -> List[str]:
        """
        Check for positions that have exceeded maximum hold time.
        
        Returns:
            List of position IDs that should be closed
        """
        current_time = time.time()
        positions_to_close = []
        
        for trade_id, position in self.open_positions.items():
            duration = current_time - position['entry_time']
            if duration > self.limits.position_timeout:
                positions_to_close.append(trade_id)
                logger.warning(f"Position {trade_id} exceeded maximum hold time")
                
        return positions_to_close

    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate current portfolio risk metrics."""
        try:
            with self.metrics_lock:
                metrics = {
                    'open_positions': len(self.open_positions),
                    'daily_trades': self.metrics.daily_trades,
                    'daily_pnl': self.metrics.daily_pnl,
                    'current_drawdown': self.metrics.current_drawdown,
                    'peak_balance': self.peak_balance
                }
                
                # Calculate win rate
                if self.position_history:
                    winning_trades = sum(1 for trade in self.position_history if trade['pnl'] > 0)
                    metrics['win_rate'] = winning_trades / len(self.position_history)
                    
                    # Calculate Sharpe ratio
                    returns = [trade['pnl'] / trade['entry_price'] for trade in self.position_history]
                    if returns:
                        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
                
                return metrics

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            raise

    def _calculate_drawdown(self, current_balance: float) -> float:
        """Calculate current drawdown percentage."""
        try:
            self.peak_balance = max(self.peak_balance, current_balance)
            drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100
            
            with self.metrics_lock:
                self.metrics.current_drawdown = drawdown
                self.metrics.drawdown_gauge.set(drawdown)
            
            return drawdown

        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            raise

    def _calculate_sharpe_ratio(self, returns: List[float], 
                              risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from returns."""
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
        
        if len(excess_returns) < 2:
            return 0.0
            
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(252)

    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report."""
        try:
            metrics = self.calculate_portfolio_metrics()
            
            report = {
                'risk_metrics': metrics,
                'limits': {
                    'max_position_size': self.limits.max_position_size,
                    'max_trades_per_day': self.limits.max_trades_per_day,
                    'max_drawdown': self.limits.max_total_drawdown,
                    'risk_per_trade': self.limits.risk_per_trade
                },
                'positions': {
                    'open': len(self.open_positions),
                    'total_exposure': sum(pos['size'] for pos in self.open_positions.values()),
                    'avg_duration': np.mean([time.time() - pos['entry_time'] 
                                          for pos in self.open_positions.values()]) 
                                          if self.open_positions else 0
                }
            }
            
            if self.position_history:
                report['statistics'] = {
                    'total_trades': len(self.position_history),
                    'win_rate': metrics['win_rate'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'avg_profit': np.mean([trade['pnl'] for trade in self.position_history]),
                    'max_profit': max(trade['pnl'] for trade in self.position_history),
                    'max_loss': min(trade['pnl'] for trade in self.position_history)
                }
                
            return report

        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            raise
