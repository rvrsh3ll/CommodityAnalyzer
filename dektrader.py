#!/usr/bin/env python3

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import json
from dataclasses import dataclass
from enum import Enum
import requests
import re
import pytz
from discord_webhook import DiscordWebhook, DiscordEmbed

class MarketSession(Enum):
    PRE_MARKET = "PRE_MARKET"  # 4AM-9:30AM
    MARKET_OPEN = "MARKET_OPEN"  # 9:30AM-11AM
    MIDDAY = "MIDDAY"  # 11AM-3PM
    MARKET_CLOSE = "MARKET_CLOSE"  # 3PM-4PM
    AFTER_HOURS = "AFTER_HOURS"
    CLOSED = "CLOSED"

@dataclass
class TradingPlan:
    symbol: str
    why: str
    entry_price: float
    stop_loss: float
    profit_target: float
    position_size: int
    datetime: datetime
    volume_ratio: float
    float_category: str
    current_volume: int
    avg_volume: int

class TradeResult:
    def __init__(self, plan: TradingPlan, exit_price: float, exit_time: datetime):
        self.plan = plan
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.profit_loss = (exit_price - plan.entry_price) * plan.position_size
        self.profit_loss_percent = ((exit_price - plan.entry_price) / plan.entry_price) * 100

class TradingSystem:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.stocks = set()
        self.daily_trades = []
        self.monthly_performance = {}
        self.consecutive_green_months = 0
        self.last_scrape_time = datetime.min
        
        # Discord webhook URL
        self.webhook_url = "https://discord.com/api/webhooks/1286420702173597807/hNgcuYY68fm6t0ncWSSGt2QwrQvEybW5uRrr2nXZCMiizQnq6Wguhm41SBJcO8TicQWy"
        
        # Configure logging
        logging.basicConfig(
            filename='trading_system.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        # Load paper trading history
        self.paper_trading_history = self._load_paper_trading_history()

    def _load_paper_trading_history(self) -> Dict:
        """Load paper trading history from file"""
        try:
            with open('paper_trading_history.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'monthly_results': {},
                'consecutive_green_months': 0,
                'total_trades': 0,
                'winning_trades': 0
            }

    def send_discord_alert(self, plan: TradingPlan, session: MarketSession):
        """Send formatted alert to Discord"""
        try:
            webhook = DiscordWebhook(url=self.webhook_url)
            
            # Create embed
            embed = DiscordEmbed(
                title=f"🎯 Trading Setup Alert: {plan.symbol}",
                color="45CEA2" if plan.volume_ratio > 10 else "5865F2"
            )
            
            # Basic trade info
            embed.add_field(
                name="💰 Price Information",
                value=(
                    f"Entry Price: ${plan.entry_price:.2f}\n"
                    f"Profit Target (+10%): ${plan.profit_target:.2f}\n"
                    f"Stop Loss (-5%): ${plan.stop_loss:.2f}\n"
                    f"Position Size: {plan.position_size:,} shares"
                ),
                inline=False
            )
            
            # Volume analysis
            volume_color = "🔴" if plan.volume_ratio > 20 else "🟡" if plan.volume_ratio > 10 else "🟢"
            embed.add_field(
                name="📊 Volume Analysis",
                value=(
                    f"{volume_color} Volume Ratio: {plan.volume_ratio:.1f}x average\n"
                    f"Current Volume: {plan.current_volume:,}\n"
                    f"Average Volume: {plan.avg_volume:,}\n"
                    f"Float Category: {plan.float_category}"
                ),
                inline=False
            )
            
            # Market context
            embed.add_field(
                name="📈 Market Context",
                value=(
                    f"Session: {session.value}\n"
                    f"Time: {plan.datetime.strftime('%I:%M:%S %p ET')}\n"
                    f"Reason: {plan.why}"
                ),
                inline=False
            )
            
            # Calculate potential profit and loss
            max_profit = (plan.profit_target - plan.entry_price) * plan.position_size
            max_loss = (plan.entry_price - plan.stop_loss) * plan.position_size
            
            embed.add_field(
                name="💵 Potential P/L",
                value=(
                    f"Max Profit: ${max_profit:,.2f}\n"
                    f"Max Loss: ${max_loss:,.2f}\n"
                    f"Risk/Reward: 2:1"
                ),
                inline=False
            )
            
            # Add timestamp
            embed.set_timestamp()
            
            # Set footer
            embed.set_footer(text="Based on Deck's Trading Strategy | Volume = Activity = Profit")
            
            # Add embed to webhook
            webhook.add_embed(embed)
            
            # Send webhook
            webhook.execute()
            
        except Exception as e:
            logging.error(f"Error sending Discord alert: {str(e)}")

    def _get_scraping_interval(self, session: MarketSession) -> int:
        """Get appropriate scraping interval based on market session"""
        if session == MarketSession.PRE_MARKET:
            return 120  # 2 minutes during pre-market
        elif session in [MarketSession.MARKET_OPEN, MarketSession.MARKET_CLOSE]:
            return 60   # 1 minute during open and close
        elif session == MarketSession.MIDDAY:
            return 300  # 5 minutes during midday
        else:
            return 600  # 10 minutes outside trading hours

    def _scrape_stock_symbols(self) -> set:
        """Scrape stock symbols from Biztoc"""
        try:
            headers = {
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'accept-language': 'en-US,en;q=0.9',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
            }
            
            response = requests.get('https://biztoc.com/', headers=headers)
            if response.status_code != 200:
                logging.error(f"Failed to fetch symbols: Status code {response.status_code}")
                return set()

            symbols = set(re.findall(r'<div class="stock_symbol">(.*?)</div>', response.text))
            logging.info(f"Scraped {len(symbols)} symbols")
            return symbols

        except Exception as e:
            logging.error(f"Error scraping symbols: {str(e)}")
            return set()

    def _categorize_float(self, shares_float: float) -> str:
        """Categorize float size per video guidelines"""
        if shares_float < 5_000_000:
            return "🔥 VERY LOW FLOAT (<5M)"
        elif shares_float < 15_000_000:
            return "⭐ LOW FLOAT (5M-15M)"
        elif shares_float < 30_000_000:
            return "📊 AVERAGE FLOAT (15M-30M)"
        else:
            return "🔷 HIGH FLOAT (>30M)"

    def get_market_session(self) -> MarketSession:
        """Determine current market session in ET"""
        et_tz = pytz.timezone('US/Eastern')
        now = datetime.now(et_tz)
        
        # Define session times
        pre_market_start = now.replace(hour=7, minute=0, second=0)
        market_open = now.replace(hour=9, minute=30, second=0)
        midday_start = now.replace(hour=11, minute=0, second=0)
        market_close_start = now.replace(hour=15, minute=0, second=0)
        market_end = now.replace(hour=16, minute=0, second=0)
        
        if now.weekday() >= 5:  # Weekend
            return MarketSession.CLOSED
            
        if pre_market_start <= now < market_open:
            return MarketSession.PRE_MARKET
        elif market_open <= now < midday_start:
            return MarketSession.MARKET_OPEN
        elif midday_start <= now < market_close_start:
            return MarketSession.MIDDAY
        elif market_close_start <= now < market_end:
            return MarketSession.MARKET_CLOSE
        else:
            return MarketSession.CLOSED

    def update_watchlist(self, session: MarketSession):
        """Update watchlist if enough time has passed"""
        now = datetime.now()
        interval = self._get_scraping_interval(session)
        
        if (now - self.last_scrape_time).total_seconds() >= interval:
            new_symbols = self._scrape_stock_symbols()
            if new_symbols:
                self.stocks.update(new_symbols)
                self.last_scrape_time = now
                logging.info(f"Updated watchlist: {len(self.stocks)} total symbols")

    def _calculate_position_size(self, price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management"""
        risk_per_trade = self.current_capital * 0.01  # 1% risk per trade
        risk_per_share = price - stop_loss
        return int(risk_per_trade / risk_per_share) if risk_per_share > 0 else 0

    def analyze_stock(self, symbol: str, session: MarketSession) -> Optional[TradingPlan]:
        """Analyze stock for potential trade setup"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get current data
            hist = stock.history(period='1d', interval='1m')
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].sum()
            
            # Get float information
            try:
                shares_float = stock.info['floatShares']
                float_category = self._categorize_float(shares_float)
            except:
                shares_float = None
                float_category = "UNKNOWN"
            
            # Volume analysis
            daily_hist = stock.history(period='1mo')
            avg_volume = daily_hist['Volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Only proceed if we have significant volume (5x average)
            if volume_ratio > 5:
                # Calculate targets using video's 2:1 ratio
                profit_target = current_price * 1.10  # 10% gain
                stop_loss = current_price * 0.95     # 5% loss
                
                # Position sizing
                position_size = self._calculate_position_size(current_price, stop_loss)
                
                # Create trading plan
                plan = TradingPlan(
                    symbol=symbol,
                    why=f"Volume surge ({volume_ratio:.1f}x avg), {float_category} float",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    profit_target=profit_target,
                    position_size=position_size,
                    datetime=datetime.now(pytz.timezone('US/Eastern')),
                    volume_ratio=volume_ratio,
                    float_category=float_category,
                    current_volume=int(current_volume),
                    avg_volume=int(avg_volume)
                )
                
                # Send Discord alert
                self.send_discord_alert(plan, session)
                
                return plan
                
        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {str(e)}")
            return None

    def run(self):
        """Main system loop"""
        print("Starting Trading System with Integrated Stock Scraping...")
        print("Following video guidelines for timing, volume, and risk management")
        
        while True:
            try:
                session = self.get_market_session()
                
                # Update watchlist based on current session
                self.update_watchlist(session)
                
                if session == MarketSession.CLOSED:
                    print("\nMarket closed. Waiting for next session...")
                    time.sleep(300)  # Check every 5 minutes during closed market
                    continue
                
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"\n=== Analyzing {len(self.stocks)} stocks at {current_time} ({session.value}) ===")
                
                # Analyze each stock in the watchlist
                for symbol in self.stocks:
                    plan = self.analyze_stock(symbol, session)
                    if plan:
                        alert = (
                            f"\nPOTENTIAL TRADE SETUP: {symbol}\n"
                            f"Reason: {plan.why}\n"
                            f"Entry: ${plan.entry_price:.2f}\n"
                            f"Target: ${plan.profit_target:.2f}\n"
                            f"Stop Loss: ${plan.stop_loss:.2f}\n"
                            f"Position Size: {plan.position_size} shares\n"
                        )
                        print(alert)
                        logging.info(alert)
                
                # Sleep based on current session
                time.sleep(self._get_scraping_interval(session))
                
            except KeyboardInterrupt:
                print("\nStopping trading system...")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                time.sleep(60)

if __name__ == "__main__":
    # Initialize system with $10,000 starting capital
    system = TradingSystem(10000)
    system.run()
