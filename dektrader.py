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
        logging.FileHandler('trading_scanner.log'),
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

@dataclass 
class AlertHistory:
    last_price: float
    last_volume: int
    last_alert_time: datetime
    alert_count: int
    daily_alerts: int
    last_alert_date: datetime

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
    def __init__(self):
        self.stocks: Set[str] = set()
        self.last_scrape_time = datetime.min
        self.webhook_url = DISCORD_WEBHOOK_URL
        self.blacklist = {'MLECW'}  # Known problematic symbols
        self.alert_history: Dict[str, AlertHistory] = {}

        # Trading session parameters
        self.trade_session = {
            'start_time': '09:30',  # Market open
            'end_time': '11:30',    # End active trading after 2 hours
            'best_hours': ['09:30', '10:30']  # Most profitable hour
        }
        
        # Price and float filters
        self.trading_filters = {
            'min_price': 2.00,      # Min $2
            'max_price': 20.00,     # $20 max price
            'min_volume': 500000,   # Minimum daily volume
            'max_float': 20000000,  # 20M float max
            'min_rel_volume': 5.0   # 500% relative volume minimum
        }
        
        logging.info("Trading Scanner initialized")

    def _should_alert(self, symbol: str, current_price: float, current_volume: int, timestamp: datetime) -> bool:
        """Determine if we should send a new alert based on history"""
        # Check if we have any history for this symbol
        if symbol not in self.alert_history:
            return True
            
        history = self.alert_history[symbol]
        
        # Check if it's a new day
        current_date = timestamp.date()
        if current_date != history.last_alert_date.date():
            history.daily_alerts = 0
            
        # Don't alert if we've already sent 5 alerts today
        if history.daily_alerts >= 5:
            logging.info(f"Skipping {symbol} - max daily alerts reached")
            return False
            
        # Don't alert if less than 15 minutes have passed
        time_since_last = (timestamp - history.last_alert_time).total_seconds()
        if time_since_last < 900:  # 15 minutes
            logging.info(f"Skipping {symbol} - too soon since last alert ({time_since_last:.0f}s)")
            return False
            
        # Calculate price movement since last alert
        price_change_pct = abs(current_price - history.last_price) / history.last_price
        if price_change_pct < 0.03:  # 3% minimum price movement
            logging.info(f"Skipping {symbol} - insufficient price movement ({price_change_pct:.1%})")
            return False
            
        # Check for significant new volume
        volume_since_last = current_volume - history.last_volume
        if volume_since_last < 100000:  # Minimum new volume
            logging.info(f"Skipping {symbol} - insufficient new volume")
            return False
            
        return True

    def _record_alert(self, symbol: str, price: float, volume: int, timestamp: datetime):
        """Record alert details for future reference"""
        # Get existing history or create new
        history = self.alert_history.get(symbol)
        if history:
            # Update existing history
            history.last_price = price
            history.last_volume = volume
            history.last_alert_time = timestamp
            history.alert_count += 1
            
            # Update daily alerts if same day, reset if new day
            if timestamp.date() == history.last_alert_date.date():
                history.daily_alerts += 1
            else:
                history.daily_alerts = 1
                
            history.last_alert_date = timestamp
        else:
            # Create new history
            self.alert_history[symbol] = AlertHistory(
                last_price=price,
                last_volume=volume,
                last_alert_time=timestamp,
                alert_count=1,
                daily_alerts=1,
                last_alert_date=timestamp
            )
        
        logging.info(f"Recorded alert for {symbol} - Price: ${price:.2f}, Volume: {volume:,}")

    def send_discord_alert(self, plan: TradingPlan, session: MarketSession):
        """Send formatted alert to Discord with tracking"""
        # Check if we should alert
        if not self._should_alert(plan.symbol, plan.entry_price, plan.current_volume, plan.datetime):
            return
            
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
            
            if response and response.status_code in [200, 204]:
                self._record_alert(plan.symbol, plan.entry_price, plan.current_volume, plan.datetime)
            else:
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
        """Categorize float size"""
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
        """Calculate theoretical position size for max $100 risk"""
        risk_per_share = price - stop_loss
        if risk_per_share <= 0:
            return 0
            
        # Target $100 risk per trade    
        position = int(100 / risk_per_share)
        
        # Limit position to keep max loss around $100
        max_shares = int(10000 / price)  # Using $10k theoretical capital
        return min(position, max_shares)

    def _analyze_pattern(self, hist: pd.DataFrame) -> Optional[str]:
        """Analyze for specific patterns"""
        try:
            # Calculate 9 EMA
            hist['EMA9'] = hist['Close'].ewm(span=9, adjust=False).mean()
            
            # Look for bull flag pattern
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

# PART 2

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
        """Analyze stock with enhanced criteria"""
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
                
                profit_target = current_price * 1.10  # 10% target
                stop_loss = current_price * 0.95     # 5% stop
                
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
                    self.stocks = new_symbols
                    self.last_scrape_time = now
                    logging.info(f"Updated watchlist: {len(self.stocks)} symbols")
                    
        except Exception as e:
            logging.error(f"Error updating watchlist: {str(e)}")

    def run(self):
        """Main system loop with enhanced logging"""
        logging.info("Starting Trading Scanner...")
        
        while True:
            try:
                session = self.get_market_session()
                
                if session == MarketSession.CLOSED:
                    logging.info("Market closed, waiting...")
                    time.sleep(300)
                    continue
                    
                # Avoid midday chop
                if session == MarketSession.MIDDAY:
                    logging.info("Midday session - reduced scanning")
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
                logging.info("Shutting down scanner...")
                break
            except Exception as e:
                logging.error(f"System error: {str(e)}")
                time.sleep(60)

if __name__ == "__main__":
    system = TradingSystem()
    system.run()
