# src/main.py
import os
import asyncio
from dotenv import load_dotenv
from binance.client import Client
import yaml

from utils.logger import setup_logger
from utils.metrics import MetricsManager
from data_manager import MarketDataManager
from risk_manager import RiskManager
from environment import EnhancedTradingEnvironment
from agent import EnhancedDQNAgent

# Load environment variables
load_dotenv()
logger = setup_logger()

async def main():
    try:
        # Load configurations
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Initialize Binance client
        client = Client(
            os.getenv("BINANCE_API_KEY"),
            os.getenv("BINANCE_API_SECRET"),
            testnet=config["trading"].get("testnet")
        )

        # Initialize components
        metrics_manager = MetricsManager()
        market_data_manager = MarketDataManager(client, config["trading"])
        risk_manager = RiskManager(config["trading"], config["risk"])

        # Initialize trading environment
        env = EnhancedTradingEnvironment(
            client=client,
            config=config["trading"],
            risk_manager=risk_manager,
            market_data_manager=market_data_manager
        )

        # Initialize agent
        state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        action_size = env.action_space.n
        agent = EnhancedDQNAgent(state_size, action_size)

        # Trading loop
        while True:
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                # Get action from agent
                action = agent.act(state)
                
                # Execute trading step
                next_state, reward, done, info = await env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
                
                state = next_state
                total_reward += reward

                # Log metrics
                metrics_manager.update_trading_metrics(reward, info)

                await asyncio.sleep(60)  # 1-minute trading interval

    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
