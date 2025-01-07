from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import asyncio
import time
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
import threading
from .utils.logger import get_logger
from .utils.metrics import DataMetrics

logger = get_logger(__name__)

class MarketDataCache:
    def __init__(self, max_age: int = 60):
        self.data: Dict[str, Dict[str, Any]] = {}
        self.max_age = max_age
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[pd.DataFrame]:
        with self.lock:
            cached_data = self.data.get(key)
            if cached_data is None:
                return None

            if time.time() - cached_data['timestamp'] > self.max_age:
                del self.data[key]
                return None

            return cached_data['data']

    def set(self, key: str, data: pd.DataFrame):
        with self.lock:
            self.data[key] = {
                'data': data,
                'timestamp': time.time()
            }

class MarketDataManager:
    def __init__(
        self,
        client: Client,
        config: Dict[str, Any],
        cache_duration: int = 60
    ):
        """
        Initialize the market data manager.
        
        Args:
            client: Binance client instance
            config: Trading configuration
            cache_duration: Duration to cache data in seconds
        """
        self.client = client
        self.config = config
        self.cache = MarketDataCache(max_age=cache_duration)
        self.metrics = DataMetrics()
        
        # Thread pool for concurrent operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Websocket connection for real-time data
        self._ws_connection = None
        self._last_kline = None
        self._ws_lock = threading.Lock()

    async def get_current_price(self) -> float:
        """Get current price with caching."""
        try:
            cache_key = f"price_{self.config['symbol']}"
            cached_price = self.cache.get(cache_key)

            if cached_price is not None:
                return cached_price

            ticker = await self.client.get_symbol_ticker(symbol=self.config['symbol'])
            price = float(ticker['price'])
            
            self.cache.set(cache_key, price)
            self.metrics.record_price_update()
            
            return price

        except BinanceAPIException as e:
            logger.error(f"Error fetching current price: {e}")
            raise

    async def get_market_data(
        self,
        interval: str = Client.KLINE_INTERVAL_1MINUTE,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Get historical market data with technical indicators.
        
        Args:
            interval: Kline interval
            limit: Number of klines to fetch
            
        Returns:
            DataFrame with OHLCV data and indicators
        """
        try:
            cache_key = f"klines_{self.config['symbol']}_{interval}_{limit}"
            cached_data = self.cache.get(cache_key)

            if cached_data is not None:
                return self._update_realtime_data(cached_data)

            # Fetch historical klines
            klines = await self.client.get_historical_klines(
                symbol=self.config['symbol'],
                interval=interval,
                limit=limit
            )

            # Convert to DataFrame
            df = await self._process_klines(klines)
            
            # Calculate indicators
            df = await self._calculate_indicators(df)
            
            self.cache.set(cache_key, df)
            self.metrics.record_data_fetch()
            
            return df

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            raise

    async def _process_klines(self, klines: List) -> pd.DataFrame:
        """Process raw kline data into DataFrame."""
        try:
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignored'
            ])

            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                             'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            # Set index
            df.set_index('timestamp', inplace=True)
            
            return df

        except Exception as e:
            logger.error(f"Error processing klines: {e}")
            raise

    async def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators in parallel."""
        try:
            # Create indicator instances
            bb = BollingerBands(df['close'])
            macd = MACD(df['close'])
            rsi = RSIIndicator(df['close'])
            vwap = VolumeWeightedAveragePrice(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            )

            # Calculate basic indicators
            df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
            df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = EMAIndicator(df['close'], window=26).ema_indicator()

            # Calculate advanced indicators in parallel
            tasks = [
                self.thread_pool.submit(lambda: {
                    'bb_high': bb.bollinger_hband(),
                    'bb_mid': bb.bollinger_mavg(),
                    'bb_low': bb.bollinger_lband()
                }),
                self.thread_pool.submit(lambda: {
                    'macd_line': macd.macd(),
                    'macd_signal': macd.macd_signal(),
                    'macd_hist': macd.macd_diff()
                }),
                self.thread_pool.submit(lambda: {
                    'rsi': rsi.rsi(),
                    'vwap': vwap.volume_weighted_average_price()
                })
            ]

            # Gather results
            results = [task.result() for task in tasks]
            
            # Update DataFrame
            for result in results:
                for key, value in result.items():
                    df[key] = value

            # Calculate additional features
            df['volatility'] = df['close'].rolling(window=20).std()
            df['returns'] = df['close'].pct_change()
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            return df

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise

    async def start_websocket(self):
        """Start websocket connection for real-time data."""
        try:
            if self._ws_connection is None:
                self._ws_connection = await self.client.start_kline_socket(
                    symbol=self.config['symbol'],
                    interval=Client.KLINE_INTERVAL_1MINUTE,
                    callback=self._handle_websocket_message
                )
                logger.info("Websocket connection established")

        except Exception as e:
            logger.error(f"Error starting websocket: {e}")
            raise

    async def stop_websocket(self):
        """Stop websocket connection."""
        try:
            if self._ws_connection:
                await self.client.stop_socket(self._ws_connection)
                self._ws_connection = None
                logger.info("Websocket connection closed")

        except Exception as e:
            logger.error(f"Error stopping websocket: {e}")

    def _handle_websocket_message(self, msg: Dict):
        """Handle incoming websocket messages."""
        try:
            with self._ws_lock:
                self._last_kline = {
                    'timestamp': msg['k']['t'],
                    'open': float(msg['k']['o']),
                    'high': float(msg['k']['h']),
                    'low': float(msg['k']['l']),
                    'close': float(msg['k']['c']),
                    'volume': float(msg['k']['v'])
                }
                self.metrics.record_websocket_update()

        except Exception as e:
            logger.error(f"Error handling websocket message: {e}")

    def _update_realtime_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Update cached data with real-time websocket data."""
        try:
            with self._ws_lock:
                if self._last_kline is None:
                    return df

                last_kline_time = pd.to_datetime(self._last_kline['timestamp'], unit='ms')
                
                if last_kline_time > df.index[-1]:
                    new_row = pd.DataFrame(
                        [self._last_kline],
                        index=[last_kline_time]
                    )
                    df = pd.concat([df[:-1], new_row])

                    # Update indicators for the new data
                    df = asyncio.run(self._calculate_indicators(df))

            return df

        except Exception as e:
            logger.error(f"Error updating real-time data: {e}")
            return df

    async def get_order_book(self, limit: int = 100) -> pd.DataFrame:
        """Get current order book data."""
        try:
            cache_key = f"orderbook_{self.config['symbol']}_{limit}"
            cached_data = self.cache.get(cache_key)

            if cached_data is not None:
                return cached_data

            depth = await self.client.get_order_book(
                symbol=self.config['symbol'],
                limit=limit
            )

            df_bids = pd.DataFrame(depth['bids'], columns=['price', 'quantity'])
            df_asks = pd.DataFrame(depth['asks'], columns=['price', 'quantity'])

            df_bids['side'] = 'bid'
            df_asks['side'] = 'ask'

            df = pd.concat([df_bids, df_asks])
            df[['price', 'quantity']] = df[['price', 'quantity']].astype(float)

            self.cache.set(cache_key, df)
            self.metrics.record_orderbook_fetch()

            return df

        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            raise

    async def get_ticker_info(self) -> Dict:
        """Get 24-hour ticker information."""
        try:
            cache_key = f"ticker_{self.config['symbol']}"
            cached_data = self.cache.get(cache_key)

            if cached_data is not None:
                return cached_data

            ticker = await self.client.get_ticker(symbol=self.config['symbol'])
            self.cache.set(cache_key, ticker)
            
            return ticker

        except Exception as e:
            logger.error(f"Error fetching ticker info: {e}")
            raise

    def cleanup(self):
        """Cleanup resources."""
        try:
            self.thread_pool.shutdown(wait=True)
            asyncio.create_task(self.stop_websocket())
            logger.info("Data manager cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
