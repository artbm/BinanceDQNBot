"""
Notification system for the trading bot.
Supports multiple channels including Telegram, Email, Discord, and Slack.
"""

import asyncio
from typing import Optional, List, Dict, Any, Union
import telegram
from telegram.error import TelegramError
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import json
from datetime import datetime, timedelta
import time
from collections import deque
import threading
from .logger import get_logger

logger = get_logger(__name__)

class RateLimiter:
    """Rate limiter for notifications."""
    
    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()

    def can_proceed(self) -> bool:
        """Check if request can proceed under rate limits."""
        current_time = time.time()
        
        with self.lock:
            # Remove old requests
            while self.requests and current_time - self.requests[0] > self.time_window:
                self.requests.popleft()

            if len(self.requests) < self.max_requests:
                self.requests.append(current_time)
                return True

            return False

class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, config: Dict[str, Any], rate_limiter: RateLimiter):
        self.config = config
        self.rate_limiter = rate_limiter
        self.enabled = True

    async def send(self, message: str, level: str) -> bool:
        """Send notification through the channel."""
        if not self.enabled or not self.rate_limiter.can_proceed():
            return False
        return await self._send(message, level)

    async def _send(self, message: str, level: str) -> bool:
        raise NotImplementedError

    def _format_message(self, message: str, level: str) -> str:
        """Format message with timestamp and level."""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        return f"[{timestamp}] [{level}] {message}"

class TelegramChannel(NotificationChannel):
    """Telegram notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, RateLimiter(max_requests=30, time_window=60))
        self.bot: Optional[telegram.Bot] = None
        self._initialize()

    def _initialize(self):
        """Initialize Telegram bot."""
        try:
            if self.config.get('telegram_token') and self.config.get('telegram_chat_id'):
                self.bot = telegram.Bot(self.config['telegram_token'])
            else:
                self.enabled = False
                logger.warning("Telegram configuration incomplete")
        except Exception as e:
            self.enabled = False
            logger.error(f"Failed to initialize Telegram bot: {e}")

    async def _send(self, message: str, level: str) -> bool:
        """Send message via Telegram."""
        try:
            if not self.bot:
                return False

            formatted_message = self._format_message(message, level)
            
            # Add emoji based on level
            level_emoji = {
                'INFO': '📝',
                'WARNING': '⚠️',
                'ERROR': '🚨',
                'CRITICAL': '🔥',
                'SUCCESS': '✅',
                'TRADE': '💰'
            }
            
            formatted_message = f"{level_emoji.get(level, '')} {formatted_message}"
            
            await self.bot.send_message(
                chat_id=self.config['telegram_chat_id'],
                text=formatted_message,
                parse_mode='HTML'
            )
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False

class EmailChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, RateLimiter(max_requests=100, time_window=3600))
        self._initialize()

    def _initialize(self):
        """Initialize email configuration."""
        required_keys = ['smtp_server', 'smtp_port', 'username', 'password', 
                        'from_addr', 'to_addr']
        
        if not all(k in self.config.get('email', {}) for k in required_keys):
            self.enabled = False
            logger.warning("Email configuration incomplete")

    async def _send(self, message: str, level: str) -> bool:
        """Send message via email."""
        try:
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from_addr']
            msg['To'] = email_config['to_addr']
            msg['Subject'] = f"Trading Bot Notification - {level}"
            
            # Create HTML message
            html_message = f"""
            <html>
                <body>
                    <div style="font-family: Arial, sans-serif; padding: 20px;">
                        <h2 style="color: #333;">Trading Bot Notification</h2>
                        <div style="padding: 15px; background-color: {self._get_level_color(level)};">
                            <p style="color: white;">{self._format_message(message, level)}</p>
                        </div>
                    </div>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html_message, 'html'))
            
            # Send email asynchronously
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._send_email_sync,
                email_config,
                msg
            )
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    def _send_email_sync(self, config: Dict[str, Any], msg: MIMEMultipart):
        """Synchronous email sending."""
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)

    def _get_level_color(self, level: str) -> str:
        """Get color for notification level."""
        return {
            'INFO': '#2196F3',
            'WARNING': '#FFA000',
            'ERROR': '#F44336',
            'CRITICAL': '#B71C1C',
            'SUCCESS': '#4CAF50',
            'TRADE': '#9C27B0'
        }.get(level, '#757575')

class DiscordChannel(NotificationChannel):
    """Discord notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, RateLimiter(max_requests=5, time_window=5))
        self._initialize()

    def _initialize(self):
        """Initialize Discord configuration."""
        if not self.config.get('discord_webhook_url'):
            self.enabled = False
            logger.warning("Discord configuration incomplete")

    async def _send(self, message: str, level: str) -> bool:
        """Send message via Discord webhook."""
        try:
            webhook_url = self.config['discord_webhook_url']
            formatted_message = self._format_message(message, level)
            
            payload = {
                'content': None,
                'embeds': [{
                    'title': f'Trading Bot Notification - {level}',
                    'description': formatted_message,
                    'color': self._get_level_color(level),
                    'timestamp': datetime.utcnow().isoformat()
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 204

        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def _get_level_color(self, level: str) -> int:
        """Get Discord color code for notification level."""
        return {
            'INFO': 0x2196F3,
            'WARNING': 0xFFA000,
            'ERROR': 0xF44336,
            'CRITICAL': 0xB71C1C,
            'SUCCESS': 0x4CAF50,
            'TRADE': 0x9C27B0
        }.get(level, 0x757575)

class NotificationManager:
    """Manages all notification channels."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize notification manager.
        
        Args:
            config: Configuration dictionary containing settings for all channels
        """
        self.channels: Dict[str, NotificationChannel] = {}
        self._initialize_channels(config)
        
        # Deduplication cache
        self.recent_messages = deque(maxlen=100)
        self.message_lock = threading.Lock()

    def _initialize_channels(self, config: Dict[str, Any]):
        """Initialize all configured notification channels."""
        channel_classes = {
            'telegram': TelegramChannel,
            'email': EmailChannel,
            'discord': DiscordChannel
        }
        
        for channel_name, channel_class in channel_classes.items():
            try:
                channel = channel_class(config)
                if channel.enabled:
                    self.channels[channel_name] = channel
            except Exception as e:
                logger.error(f"Failed to initialize {channel_name} channel: {e}")

    def _is_duplicate(self, message: str, level: str, window: int = 300) -> bool:
        """Check if message is a duplicate within time window."""
        current_time = time.time()
        message_key = f"{level}:{message}"
        
        with self.message_lock:
            # Clean old messages
            while (self.recent_messages and 
                   current_time - self.recent_messages[0][1] > window):
                self.recent_messages.popleft()
            
            # Check for duplicates
            for recent_key, _ in self.recent_messages:
                if recent_key == message_key:
                    return True
            
            # Add new message
            self.recent_messages.append((message_key, current_time))
            return False

    async def send_notification(
        self,
        message: str,
        level: str = 'INFO',
        channels: Optional[List[str]] = None,
        deduplicate: bool = True
    ) -> Dict[str, bool]:
        """
        Send notification through specified channels.
        
        Args:
            message: Notification message
            level: Message importance level
            channels: List of channels to use (default: all configured channels)
            deduplicate: Whether to prevent duplicate messages
            
        Returns:
            Dictionary of channel names and their success status
        """
        if deduplicate and self._is_duplicate(message, level):
            logger.debug(f"Skipping duplicate message: {message}")
            return {}

        results = {}
        channels = channels or list(self.channels.keys())
        
        for channel_name in channels:
            if channel := self.channels.get(channel_name):
                try:
                    success = await channel.send(message, level)
                    results[channel_name] = success
                except Exception as e:
                    logger.error(f"Error in {channel_name} channel: {e}")
                    results[channel_name] = False
        
        return results

    async def send_trade_notification(
        self,
        trade_type: str,
        symbol: str,
        quantity: float,
        price: float,
        profit: Optional[float] = None
    ) -> Dict[str, bool]:
        """
        Send formatted trade notification.
        
        Args:
            trade_type: Type of trade (BUY/SELL)
            symbol: Trading pair symbol
            quantity: Trade quantity
            price: Trade price
            profit: Optional profit/loss amount
        """
        message = (
            f"Trade: {trade_type}\n"
            f"Symbol: {symbol}\n"
            f"Quantity: {quantity}\n"
            f"Price: {price}"
        )
        
        if profit is not None:
            message += f"\nProfit/Loss: {profit:+.8f}"
        
        return await self.send_notification(
            message=message,
            level='TRADE',
            deduplicate=False
        )
