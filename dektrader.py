#!/usr/bin/env python3

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Set
import json
from dataclasses import dataclass
from enum import Enum
import requests
import re
import pytz
from discord_webhook import DiscordWebhook
from itertools import islice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)

# Discord webhook URL
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1286420702173597807/hNgcuYY68fm6t0ncWSSGt2QwrQvEybW5uRrr2nXZCMiizQnq6Wguhm41SBJcO8TicQWy"

class MarketSession(Enum):
    PRE_MARKET = "PRE_MARKET"      # 4AM-9:30AM
    MARKET_OPEN = "MARKET_OPEN"    # 9:30AM-11AM
    MIDDAY = "MIDDAY"             # 11AM-3PM
    MARKET_CLOSE = "MARKET_CLOSE"  # 3PM-4PM
    AFTER_HOURS = "AFTER_HOURS"
    CLOSED = "CLOSED"

@dataclass
class TradingCosts:
    commission: float = 0  
    sec_fee: float      
    finra_fee: float    
    total_cost: float

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
    costs: TradingCosts

class TradingSystem:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.stocks: Set[str] = set()
        self.last_scrape_time = datetime.min
        self.webhook_url = DISCORD_WEBHOOK_URL
        self.blacklist = {'MLECW'}  # Add known problematic symbols
        self.min_price = 1.00  # Minimum price filter
        self.max_price = 50.00  # Maximum price filter
        self.min_volume = 50000  # Minimum volume filter
        
        logging.info(f"Trading System initialized with ${initial_capital:,.2f}")

    def send_discord_alert(self, plan: TradingPlan, session: MarketSession):
        """Send formatted alert to Discord"""
        try:
            webhook = DiscordWebhook(
                url=self.webhook_url,
                rate_limit_retry=True,
                timeout=5
            )
            
            # Calculate potential profit/loss after fees
            max_profit = (plan.profit_target - plan.entry_price) * plan.position_size - plan.costs.total_cost
            max_loss = (plan.entry_price - plan.stop_loss) * plan.position_size + plan.costs.total_cost
            
            message = (
                f"🎯 TRADING SETUP: {plan.symbol}\n\n"
                f"💰 Entry: ${plan.entry_price:.2f}\n"
                f"🎯 Target: ${plan.profit_target:.2f}\n"
                f"⛔ Stop: ${plan.stop_loss:.2f}\n"
                f"📊 Size: {plan.position_size:,} shares\n\n"
                f"💵 Cost Analysis:\n"
                f"SEC Fee: ${plan.costs.sec_fee:.2f}\n"
                f"FINRA Fee: ${plan.costs.finra_fee:.2f}\n"
                f"Total Fees: ${plan.costs.total_cost:.2f}\n"
                f"Max Profit (after fees): ${max_profit:.2f}\n"
                f"Max Loss (after fees): ${max_loss:.2f}\n\n"
                f"📈 Volume: {plan.volume_ratio:.1f}x average\n"
                f"Float: {plan.float_category}\n"
                f"Session: {session.value}\n"
                f"Reason: {plan.why}"
            )
            
            webhook.content = message
            response = webhook.execute()
            
            if response and response.status_code not in [200, 204]:
                logging.error(f"Discord alert failed with status {response.status_code}")
                
        except Exception as e:
            logging.error(f"Discord alert failed for {plan.symbol}: {str(e)}")

    def _get_scraping_interval(self, session: MarketSession) -> int:
        """Get appropriate scraping interval based on market session"""
        intervals = {
            MarketSession.PRE_MARKET: 120,    # 2 min
            MarketSession.MARKET_OPEN: 60,    # 1 min
            MarketSession.MARKET_CLOSE: 60,   # 1 min
            MarketSession.MIDDAY: 300,        # 5 min
            MarketSession.AFTER_HOURS: 600,   # 10 min
            MarketSession.CLOSED: 600         # 10 min
        }
        return intervals.get(session, 300)

    def _scrape_stock_symbols(self) -> set:
        """Scrape stock symbols with filtering"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get('https://biztoc.com/',
                                 headers=headers,
                                 timeout=10)
            
            if response.status_code == 200:
                # Extract symbols and filter out blacklisted ones
                symbols = set(re.findall(r'<div class="stock_symbol">(.*?)</div>', 
                                       response.text))
                filtered_symbols = {sym for sym in symbols if sym not in self.blacklist}
                
                if len(filtered_symbols) > 0:
                    logging.info(f"Scraped {len(filtered_symbols)} valid symbols")
                    return filtered_symbols
                
        except Exception as e:
            logging.error(f"Scraping failed: {str(e)}")
            
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

    def _is_valid_stock(self, symbol: str) -> bool:
        """Validate stock against filters"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1d')
            
            if hist.empty:
                return False
                
            price = hist['Close'].iloc[-1]
            volume = hist['Volume'].sum()
            
            # Apply filters
            return (
                self.min_price <= price <= self.max_price and
                volume >= self.min_volume and
                symbol not in self.blacklist
            )
            
        except Exception:
            return False

    def _calculate_trading_costs(self, price: float, shares: int) -> TradingCosts:
        """Calculate Robinhood trading costs"""
        principal = price * shares
        sec_fee = (principal / 1_000_000) * 8.00
        finra_fee = (principal / 1_000) * 0.02
        total_cost = sec_fee + finra_fee
        
        return TradingCosts(
            sec_fee=sec_fee,
            finra_fee=finra_fee,
            total_cost=total_cost
        )

    def _calculate_position_size(self, price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management"""
        risk_per_trade = self.current_capital * 0.01  # 1% risk
        risk_per_share = price - stop_loss
        return int(risk_per_trade / risk_per_share) if risk_per_share > 0 else 0

    def get_market_session(self) -> MarketSession:
        """Determine current market session"""
        et_tz = pytz.timezone('US/Eastern')
        now = datetime.now(et_tz)
        
        # Return closed for weekends
        if now.weekday() >= 5:
            return MarketSession.CLOSED
            
        # Define session times
        pre_market_start = now.replace(hour=4, minute=0, second=0)
        market_open = now.replace(hour=9, minute=30, second=0)
        midday_start = now.replace(hour=11, minute=0, second=0)
        market_close_start = now.replace(hour=15, minute=0, second=0)
        market_end = now.replace(hour=16, minute=0, second=0)
        after_hours_end = now.replace(hour=20, minute=0, second=0)
        
        if pre_market_start <= now < market_open:
            return MarketSession.PRE_MARKET
        elif market_open <= now < midday_start:
            return MarketSession.MARKET_OPEN
        elif midday_start <= now < market_close_start:
            return MarketSession.MIDDAY
        elif market_close_start <= now < market_end:
            return MarketSession.MARKET_CLOSE
        elif market_end <= now < after_hours_end:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.CLOSED

    def analyze_stock(self, symbol: str, session: MarketSession) -> Optional[TradingPlan]:
        """Analyze stock with enhanced filtering"""
        try:
            if not self._is_valid_stock(symbol):
                return None
                
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1d', interval='1m')
            
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].sum()
            
            # Get float if available
            try:
                shares_float = stock.info.get('floatShares', 0)
                float_category = self._categorize_float(shares_float)
            except:
                float_category = "UNKNOWN FLOAT"
                
            # Volume analysis
            daily_hist = stock.history(period='5d')
            if daily_hist.empty:
                return None
                
            avg_volume = daily_hist['Volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Enhanced setup criteria
            if (volume_ratio > 5 and 
                current_price >= self.min_price and 
                current_volume >= self.min_volume):
                
                profit_target = current_price * 1.10
                stop_loss = current_price * 0.95
                
                position_size = self._calculate_position_size(current_price, stop_loss)
                costs = self._calculate_trading_costs(current_price, position_size)
                
                return TradingPlan(
                    symbol=symbol,
                    why=f"Volume surge ({volume_ratio:.1f}x avg), {float_category}",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    profit_target=profit_target,
                    position_size=position_size,
                    datetime=datetime.now(pytz.timezone('US/Eastern')),
                    volume_ratio=volume_ratio,
                    float_category=float_category,
                    current_volume=int(current_volume),
                    avg_volume=int(avg_volume),
                    costs=costs
                )
                
        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {str(e)}")
            return None

    def _chunk_symbols(self, symbols: Set[str], size: int = 10):
        """Split symbols into chunks for processing"""
        it = iter(symbols)
        return iter(lambda: tuple(islice(it, size)), ())

    def update_watchlist(self, session: MarketSession):
        """Update watchlist with validation"""
        try:
            now = datetime.now()
            interval = self._get_scraping_interval(session)
            
            if (now - self.last_scrape_time).total_seconds() >= interval:
                new_symbols = self._scrape_stock_symbols()
                
                if new_symbols:
                    self.stocks = new_symbols  # Replace instead of update
                    self.last_scrape_time = now
                    logging.info(f"Updated watchlist: {len(self.stocks)} symbols")
                    
        except Exception as e:
            logging.error(f"Error updating watchlist: {str(e)}")

    def run(self):
        """Main system loop"""
        logging.info("Starting Trading System...")
        
        while True:
            try:
                session = self.get_market_session()
                
                if session == MarketSession.CLOSED:
                    logging.info("Market closed, waiting...")
                    time.sleep(300)
                    continue
                
                # Update watchlist
                self.update_watchlist(session)
                
                # Process stocks in chunks
                for chunk in self._chunk_symbols(self.stocks):
                    for symbol in chunk:
                        plan = self.analyze_stock(symbol, session)
                        if plan:
                            self.send_discord_alert(plan, session)
                    time.sleep(2)  # Rate limiting
                    
                # Sleep based on session
                time.sleep(self._get_scraping_interval(session))
                
            except KeyboardInterrupt:
                logging.info("Shutting down trading system...")
                break
            except Exception as e:
                logging.error(f"System error: {str(e)}")
                time.sleep(60)

if __name__ == "__main__":
    # Initialize system with $10,000 starting capital
    system = TradingSystem(10000)
    system.run()
