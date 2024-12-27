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

# Discord webhook URL for alerts
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1286420702173597807/hNgcuYY68fm6t0ncWSSGt2QwrQvEybW5uRrr2nXZCMiizQnq6Wguhm41SBJcO8TicQWy"

class MarketSession(Enum):
    PRE_MARKET = "PRE_MARKET"      # 4AM-9:30AM  
    MARKET_OPEN = "MARKET_OPEN"    # 9:30AM-11AM - Primary trading window
    MIDDAY = "MIDDAY"             # 11AM-3PM - Avoid trading
    MARKET_CLOSE = "MARKET_CLOSE"  # 3PM-4PM
    AFTER_HOURS = "AFTER_HOURS"    # 4PM-8PM
    CLOSED = "CLOSED"

@dataclass
class TradingCosts:
    sec_fee: float      
    finra_fee: float    
    total_cost: float
    commission: float = 0

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
    pattern: Optional[str] = None

class TradingSystem:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.stocks: Set[str] = set()
        self.last_scrape_time = datetime.min
        self.webhook_url = DISCORD_WEBHOOK_URL
        self.blacklist = {'MLECW'}  # Known problematic symbols

        # Trading session parameters based on Ross's strategy
        self.trade_session = {
            'start_time': '09:30',  # Market open
            'end_time': '11:30',    # End active trading after 2 hours
            'best_hours': ['09:30', '10:30']  # Most profitable hour
        }
        
        # Price and float filters - "2020 Rule" from Ross
        self.trading_filters = {
            'min_price': 2.00,      # Min $2 based on transcript
            'max_price': 20.00,     # "2020 Rule" - $20 max price
            'min_volume': 500000,   # Minimum daily volume
            'max_float': 20000000,  # "2020 Rule" - 20M float max
            'min_rel_volume': 5.0   # 500% relative volume minimum
        }
        
        # Risk management parameters from Ross's strategy
        self.risk_rules = {
            'max_loss_day': 2000,     # Stop trading if down $2k
            'max_size_capital': 0.95,  # Max 95% of buying power
            'initial_stop': 0.95,      # 5% stop loss from entry
            'profit_target': 1.10,     # 10% profit target
            'risk_per_trade': 0.01     # 1% account risk per trade
        }
        
        logging.info(f"Trading System initialized with ${initial_capital:,.2f}")

    def send_discord_alert(self, plan: TradingPlan, session: MarketSession):
        """Send formatted alert to Discord with enhanced pattern info"""
        try:
            webhook = DiscordWebhook(
                url=self.webhook_url,
                rate_limit_retry=True,
                timeout=5
            )
            
            max_profit = (plan.profit_target - plan.entry_price) * plan.position_size - plan.costs.total_cost
            max_loss = (plan.entry_price - plan.stop_loss) * plan.position_size + plan.costs.total_cost
            
            pattern_info = f"\nPattern: {plan.pattern}" if plan.pattern else ""
            
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
                f"Session: {session.value}"
                f"{pattern_info}\n"
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
            MarketSession.MARKET_OPEN: 60,    # 1 min - Most active
            MarketSession.MARKET_CLOSE: 60,   # 1 min
            MarketSession.MIDDAY: 300,        # 5 min - Less active
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
        """Categorize float size based on Ross's guidelines"""
        if shares_float < 5_000_000:
            return "🔥 VERY LOW FLOAT (<5M)"
        elif shares_float < 15_000_000:
            return "⭐ LOW FLOAT (5M-15M)"
        elif shares_float < 30_000_000:
            return "📊 AVERAGE FLOAT (15M-30M)"
        else:
            return "🔷 HIGH FLOAT (>30M)"

    def _is_valid_stock(self, symbol: str) -> bool:
        """Validate stock against enhanced filters"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1d')
            
            if hist.empty:
                return False
                
            price = hist['Close'].iloc[-1]
            volume = hist['Volume'].sum()
            
            # Apply enhanced filters from Ross's strategy
            return (
                self.trading_filters['min_price'] <= price <= self.trading_filters['max_price'] and
                volume >= self.trading_filters['min_volume'] and
                symbol not in self.blacklist
            )
            
        except Exception:
            return False

    def _calculate_trading_costs(self, price: float, shares: int) -> TradingCosts:
        """Calculate trading costs with standard fees"""
        principal = price * shares
        sec_fee = (principal / 1_000_000) * 8.00  # SEC fee
        finra_fee = (principal / 1_000) * 0.02    # FINRA fee
        total_cost = sec_fee + finra_fee
        
        return TradingCosts(
            sec_fee=sec_fee,
            finra_fee=finra_fee,
            total_cost=total_cost
        )

    def _calculate_position_size(self, price: float, stop_loss: float) -> int:
        """Calculate position size based on Ross's 1% risk rule"""
        risk_per_trade = self.current_capital * self.risk_rules['risk_per_trade']
        risk_per_share = price - stop_loss
        if risk_per_share <= 0:
            return 0
            
        position = int(risk_per_trade / risk_per_share)
        
        # Limit to 95% max of buying power
        max_shares = int((self.current_capital * self.risk_rules['max_size_capital']) / price)
        return min(position, max_shares)

    def _analyze_pattern(self, hist: pd.DataFrame) -> Optional[str]:
        """Analyze for Ross's favorite patterns"""
        try:
            # Calculate 9 EMA for his setups
            hist['EMA9'] = hist['Close'].ewm(span=9, adjust=False).mean()
            
            # Look for bull flag pattern - First pullback
            if (hist['High'].iloc[-5:].max() > hist['High'].iloc[-10:-5].max() and
                hist['Low'].iloc[-3:].min() > hist['Low'].iloc[-6:-3].min()):
                return "Bull Flag - First Pullback"
            
            # Look for flat top breakout
            recent_highs = hist['High'].iloc[-20:]
            if len(set(round(recent_highs, 2))) < 3:  # Similar highs
                return "Flat Top Breakout"
            
            # Look for red to green move
            if (hist['Open'].iloc[-1] > hist['Close'].iloc[0] and
                hist['Close'].iloc[-2] < hist['Close'].iloc[0]):
                return "Red to Green Move"
                
            return None
            
        except Exception as e:
            logging.error(f"Error in pattern analysis: {str(e)}")
            return None

    def get_market_session(self) -> MarketSession:
        """Determine current market session"""
        et_tz = pytz.timezone('US/Eastern')
        now = datetime.now(et_tz)
        
        if now.weekday() >= 5:
            return MarketSession.CLOSED
            
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
        """Analyze stock with enhanced criteria from Ross's strategy"""
        try:
            if not self._is_valid_stock(symbol):
                return None
                
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1d', interval='1m')
            
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].sum()
            
            try:
                shares_float = stock.info.get('floatShares', 0)
                if shares_float > self.trading_filters['max_float']:
                    return None
                float_category = self._categorize_float(shares_float)
            except:
                float_category = "UNKNOWN FLOAT"
                
            daily_hist = stock.history(period='5d')
            if daily_hist.empty:
                return None
                
            avg_volume = daily_hist['Volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            if (volume_ratio > self.trading_filters['min_rel_volume'] and 
                current_volume >= self.trading_filters['min_volume']):
                
                # Identify pattern
                pattern = self._analyze_pattern(hist)
                
                profit_target = current_price * self.risk_rules['profit_target']
                stop_loss = current_price * self.risk_rules['initial_stop']
                
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
                    costs=costs,
                    pattern=pattern
                )
                
        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {str(e)}")
            return None

    def _chunk_symbols(self, symbols: Set[str], size: int = 10):
        """Split symbols into chunks for processing to avoid rate limits"""
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
                    # Replace instead of update based on Ross's strategy of focusing on fresh setups
                    self.stocks = new_symbols
                    self.last_scrape_time = now
                    logging.info(f"Updated watchlist: {len(self.stocks)} symbols")
                    
        except Exception as e:
            logging.error(f"Error updating watchlist: {str(e)}")

    def run(self):
        """Main system loop with Ross's trading hours focus"""
        logging.info("Starting Trading System...")
        daily_pnl = 0  # Track daily P&L for max loss rule
        
        while True:
            try:
                session = self.get_market_session()
                
                # Reset P&L at market open
                if session == MarketSession.MARKET_OPEN:
                    daily_pnl = 0
                
                if session == MarketSession.CLOSED:
                    logging.info("Market closed, waiting...")
                    time.sleep(300)
                    continue
                
                # Stop trading if max loss reached
                if daily_pnl < -self.risk_rules['max_loss_day']:
                    logging.warning(f"Max daily loss of ${self.risk_rules['max_loss_day']} reached. Stopping for the day.")
                    time.sleep(300)
                    continue
                
                # Avoid midday chop
                if session == MarketSession.MIDDAY:
                    logging.info("Midday session - reduced trading")
                    time.sleep(self._get_scraping_interval(session))
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
