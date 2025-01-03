#!/usr/bin/env python3

import yfinance as yf 
import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta 
import time 
import logging 
from typing import Dict, List, Optional, Set 
import json 
from dataclasses import dataclass, field
from enum import Enum
import requests
import re
import pytz
from discord_webhook import DiscordWebhook
from itertools import islice
import ollama
from concurrent.futures import ThreadPoolExecutor
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO for cleaner logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_scanner.log'),
        logging.StreamHandler()
    ]
)

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

@dataclass
class Position:
    symbol: str
    entry_price: float
    entry_time: datetime
    initial_pattern: str
    stop_loss: float
    profit_target: float
    size: int
    last_check: datetime
    scale_out_levels: List[float]
    risk_level: str
    last_llm_advice: Optional[Dict] = None
    partial_exits: List[float] = field(default_factory=list)
    monitoring_enabled: bool = True

class LLMAnalyzer:
    """Handles LLM-based pattern analysis"""
    
    def __init__(self):
        self.model = LLM_CONFIG['model']
        self._init_llm()
        self.executor = ThreadPoolExecutor(max_workers=LLM_CONFIG['max_workers'])
        
    def _init_llm(self):
        """Initialize LLM connection"""
        try:
            ollama.generate(model=self.model, 
                          prompt="Test connection")
            logging.info(f"🤖 LLM initialized: {self.model}")
        except Exception as e:
            logging.error(f"❌ LLM initialization failed: {str(e)}")
            raise

    def _format_candle_data(self, hist: pd.DataFrame) -> str:
        """Format recent price action for LLM analysis"""
        try:
            recent = hist.tail(5)  # Last 5 candles
            candles = []
            for _, row in recent.iterrows():
                candle = {
                    'o': round(row['Open'], 2),
                    'h': round(row['High'], 2),
                    'l': round(row['Low'], 2),
                    'c': round(row['Close'], 2),
                    'v': int(row['Volume'])
                }
                candles.append(candle)
            return json.dumps(candles)
        except Exception as e:
            logging.error(f"❌ Error formatting candle data: {str(e)}")
            raise

    def analyze_pattern(self, symbol: str, hist: pd.DataFrame) -> Optional[tuple[str, str]]:
        """Get LLM analysis of price pattern"""
        try:
            logging.info(f"🤖 Starting LLM analysis for {symbol}")
            data = self._format_candle_data(hist)
            
            prompt = f"""Act as an expert day trader. 
            Analyze these 5 recent price candles for a day trade setup: {data}

            Consider:
            1. Price momentum and trend direction
            2. Volume patterns and accumulation
            3. Support/resistance levels
            4. Risk vs reward potential

            Identify ONE primary pattern and its characteristics:
            BULL FLAG, BREAKOUT, FADE, TREND REVERSAL, MOMENTUM SURGE

            Format response as JSON:
            {{
                "pattern": "pattern_name",
                "volume": "INCREASING|DECREASING|SPIKE|NORMAL",
                "risk": "LOW|MEDIUM|HIGH",
                "confidence": "percentage between 0-100"
            }}

            One line JSON only."""
            
            logging.info(f"🤖 Sending analysis request for {symbol}")
            future = self.executor.submit(
                ollama.generate, model=self.model, prompt=prompt
            )
            response = future.result(timeout=LLM_CONFIG['timeout'])
            result = json.loads(response.strip())
            
            logging.info(f"🤖 Analysis Results for {symbol}:")
            logging.info(f"  - Pattern: {result['pattern']}")
            logging.info(f"  - Volume: {result['volume']}")
            logging.info(f"  - Risk: {result['risk']}")
            logging.info(f"  - Confidence: {result['confidence']}%")
            
            pattern = f"{result['pattern']} ({result['volume']} volume, {result['confidence']}% confidence)"
            return pattern, result['risk']
            
        except Exception as e:
            logging.error(f"❌ LLM analysis failed for {symbol}: {str(e)}")
            return None, None

    def get_exit_plan(self, symbol: str, entry_price: float, pattern: str, risk_level: str) -> Dict:
        """Get LLM suggested exit strategy"""
        try:
            logging.info(f"🤖 Generating exit plan for {symbol}")
            prompt = f"""Given a day trade entry:
            Entry Price: ${entry_price:.2f}
            Pattern: {pattern}
            Risk Level: {risk_level}

            Provide a complete exit strategy with exact prices.
            Consider current market volatility and pattern reliability.

            Format response as JSON:
            {{
                "stop_loss_pct": float,
                "profit_target_pct": float,
                "scale_out_levels": [float, float, float]
            }}

            One line JSON only."""
            
            future = self.executor.submit(
                ollama.generate, model=self.model, prompt=prompt
            )
            response = future.result(timeout=LLM_CONFIG['timeout'])
            plan = json.loads(response.strip())
            
            result = {
                "stop_loss": entry_price * (1 - plan["stop_loss_pct"]/100),
                "profit_target": entry_price * (1 + plan["profit_target_pct"]/100),
                "scale_out_levels": [
                    entry_price * (1 + pct/100) 
                    for pct in plan["scale_out_levels"]
                ]
            }
            
            logging.info(f"🤖 Exit Plan for {symbol}:")
            logging.info(f"  - Stop Loss: ${result['stop_loss']:.2f} ({plan['stop_loss_pct']:.1f}%)")
            logging.info(f"  - Target: ${result['profit_target']:.2f} ({plan['profit_target_pct']:.1f}%)")
            logging.info(f"  - Scale Out Levels: {', '.join([f'${x:.2f}' for x in result['scale_out_levels']])}")
            
            return result
            
        except Exception as e:
            logging.error(f"❌ Exit plan error for {symbol}: {str(e)}")
            return {
                "stop_loss": entry_price * 0.95,
                "profit_target": entry_price * 1.10,
                "scale_out_levels": [
                    entry_price * 1.05,
                    entry_price * 1.07,
                    entry_price * 1.09
                ]
            }

    def monitor_position(self, position: Position, current_price: float, 
                        current_volume: int) -> Dict:
        """Get LLM advice for open position"""
        try:
            time_held = (datetime.now() - position.entry_time).total_seconds() / 60
            pnl = ((current_price/position.entry_price - 1) * 100)
            logging.info(f"🤖 Monitoring {position.symbol}:")
            logging.info(f"  - Time Held: {time_held:.1f}m")
            logging.info(f"  - P&L: {pnl:.1f}%")
            
            prompt = f"""Analyze active day trade position:
            Symbol: {position.symbol}
            Entry: ${position.entry_price:.2f}
            Current: ${current_price:.2f}
            P&L: {pnl:.1f}%
            Time Held: {time_held:.1f} minutes
            Pattern: {position.initial_pattern}
            Risk Level: {position.risk_level}

            What action should be taken? Consider:
            1. Position age (day trade)
            2. Profit/loss %
            3. Initial setup validity

            Format response as JSON:
            {{
                "action": "HOLD|SELL|PARTIAL_EXIT",
                "reason": "brief explanation",
                "urgency": "LOW|MEDIUM|HIGH"
            }}

            One line JSON only."""
            
            future = self.executor.submit(
                ollama.generate, model=self.model, prompt=prompt
            )
            response = future.result(timeout=LLM_CONFIG['timeout'])
            result = json.loads(response.strip())
            
            logging.info(f"🤖 Position Advice for {position.symbol}:")
            logging.info(f"  - Action: {result['action']}")
            logging.info(f"  - Reason: {result['reason']}")
            logging.info(f"  - Urgency: {result['urgency']}")
            
            return result
            
        except Exception as e:
            logging.error(f"❌ Monitor error for {position.symbol}: {str(e)}")
            return {
                "action": "HOLD",
                "reason": "monitoring error",
                "urgency": "LOW"
            }

class PositionManager:
    """Manages active trading positions"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.last_position_check = {}
        
    def add_position(self, symbol: str, plan: TradingPlan, llm_risk: str, 
                    scale_levels: List[float]):
        """Add new position to track"""
        if len(self.positions) >= MONITORING_CONFIG['max_positions']:
            logging.warning(f"Maximum positions reached, cannot add {symbol}")
            return
            
        self.positions[symbol] = Position(
            symbol=symbol,
            entry_price=plan.entry_price,
            entry_time=datetime.now(pytz.timezone('US/Eastern')),
            initial_pattern=plan.pattern,
            stop_loss=plan.stop_loss,
            profit_target=plan.profit_target,
            size=plan.position_size,
            last_check=datetime.now(),
            scale_out_levels=scale_levels,
            risk_level=llm_risk
        )
        logging.info(f"Added position: {symbol} at ${plan.entry_price:.2f}")
        
    def get_active_positions(self) -> List[Position]:
        """Get all active positions"""
        return list(self.positions.values())
        
    def remove_position(self, symbol: str):
        """Remove closed position"""
        if symbol in self.positions:
            del self.positions[symbol]
            logging.info(f"Removed position: {symbol}")
            
    def should_check_position(self, symbol: str) -> bool:
        """Determine if position should be checked"""
        now = datetime.now()
        last_check = self.last_position_check.get(symbol, datetime.min)
        
        return (now - last_check).total_seconds() >= MONITORING_CONFIG['min_check_interval']

    def update_position_check(self, symbol: str):
        """Update last check time for position"""
        self.last_position_check[symbol] = datetime.now()
		
class TradingSystem:
    """Main trading system with LLM integration"""
    
    def __init__(self):
        self.stocks: Set[str] = set()
        self.last_scrape_time = datetime.min
        self.webhook_url = DISCORD_WEBHOOK_URL
        self.blacklist = {'MLECW'}  # Known problematic symbols
        self.alert_history: Dict[str, AlertHistory] = {}
        self.llm = LLMAnalyzer()
        self.position_manager = PositionManager()
        
        logging.info(f"🚀 Trading Scanner initialized with {LLM_CONFIG['model']}")
        
    def get_market_session(self) -> MarketSession:
        """Determine current market session"""
        et_tz = pytz.timezone('US/Eastern')
        now = datetime.now(et_tz)
        
        if now.weekday() >= 5:  # Weekend check
            logging.info("📅 Weekend - Market Closed")
            return MarketSession.CLOSED
            
        current_time = now.strftime('%H:%M')
        logging.debug(f"⏰ Current time (ET): {current_time}")
        
        if TRADING_SESSIONS['pre_market_start'] <= current_time < TRADING_SESSIONS['market_open']:
            return MarketSession.PRE_MARKET
        elif TRADING_SESSIONS['market_open'] <= current_time < TRADING_SESSIONS['midday_start']:
            return MarketSession.MARKET_OPEN
        elif TRADING_SESSIONS['midday_start'] <= current_time < '15:00':
            return MarketSession.MIDDAY
        elif '15:00' <= current_time < TRADING_SESSIONS['market_close']:
            return MarketSession.MARKET_CLOSE
        elif TRADING_SESSIONS['market_close'] <= current_time < TRADING_SESSIONS['after_hours_end']:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.CLOSED

    def _is_valid_stock(self, symbol: str) -> bool:
        """Validate stock against filters with detailed logging"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1d')
            
            if hist.empty:
                logging.debug(f"❌ {symbol}: No price data available")
                return False
                
            price = hist['Close'].iloc[-1]
            volume = hist['Volume'].sum()
            
            # Calculate price change
            if len(hist) > 1:
                price_change = ((price - hist['Open'].iloc[0]) / hist['Open'].iloc[0]) * 100
            else:
                price_change = 0
                
            logging.info(f"📊 {symbol} Initial Analysis:")
            logging.info(f"  - Price: ${price:.2f}")
            logging.info(f"  - Volume: {volume:,}")
            logging.info(f"  - Price Change: {price_change:.1f}%")
            
            # Price checks
            if price < TRADING_FILTERS['min_price']:
                logging.info(f"❌ {symbol}: Price too low (${price:.2f})")
                return False
                
            if price > TRADING_FILTERS['max_price']:
                logging.info(f"❌ {symbol}: Price too high (${price:.2f})")
                return False
                
            # Volume check
            if volume < TRADING_FILTERS['min_volume']:
                logging.info(f"❌ {symbol}: Volume too low ({volume:,})")
                return False
                
            # Blacklist check
            if symbol in self.blacklist:
                logging.info(f"❌ {symbol}: Blacklisted")
                return False
                
            # If price change is significant, flag it
            if abs(price_change) > 20:
                logging.info(f"⚡ {symbol}: Large move detected ({price_change:.1f}%)")
                
            logging.info(f"✅ {symbol}: Passed initial validation")
            return True
            
        except Exception as e:
            logging.error(f"❌ {symbol}: Validation error - {str(e)}")
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
            
        position = int(RISK_CONFIG['max_loss_per_trade'] / risk_per_share)
        max_shares = int(10000 / price)  # Using $10k theoretical capital
        return min(position, max_shares)

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

    def _should_use_llm(self, symbol: str, volume_ratio: float, shares_float: float, 
                       session: MarketSession) -> bool:
        """Determine if setup warrants LLM analysis"""
        sessions_allowed = [
            MarketSession.MARKET_OPEN, 
            MarketSession.PRE_MARKET, 
            MarketSession.MARKET_CLOSE,
            MarketSession.AFTER_HOURS
        ]
        
        should_use = (
            session in sessions_allowed and
            volume_ratio >= LLM_FILTERS['min_volume_ratio'] and
            shares_float <= LLM_FILTERS['max_float']
        )
        
        logging.info(f"🤖 LLM Eligibility Check for {symbol}:")
        logging.info(f"  - Session: {session.value}")
        logging.info(f"  - Volume Ratio: {volume_ratio:.1f}x (min: {LLM_FILTERS['min_volume_ratio']}x)")
        logging.info(f"  - Float: {shares_float:,} (max: {LLM_FILTERS['max_float']:,})")
        logging.info(f"  - Using LLM: {'✅' if should_use else '❌'}")
        
        return should_use

    def _should_alert(self, symbol: str, current_price: float, current_volume: int, 
                     timestamp: datetime) -> bool:
        """Determine if we should send alert"""
        if symbol not in self.alert_history:
            logging.debug(f"📢 {symbol}: First alert")
            return True
            
        history = self.alert_history[symbol]
        
        current_date = timestamp.date()
        if current_date != history.last_alert_date.date():
            history.daily_alerts = 0
            logging.debug(f"📅 {symbol}: New trading day - resetting alert count")
            
        if history.daily_alerts >= ALERT_CONFIG['max_daily_alerts']:
            logging.debug(f"⛔ {symbol}: Max daily alerts reached ({history.daily_alerts})")
            return False
            
        time_since_last = (timestamp - history.last_alert_time).total_seconds()
        if time_since_last < ALERT_CONFIG['min_alert_interval']:
            logging.debug(f"⏳ {symbol}: Alert interval not met ({time_since_last:.0f}s)")
            return False
            
        price_change_pct = abs(current_price - history.last_price) / history.last_price
        if price_change_pct < ALERT_CONFIG['min_price_movement']:
            logging.debug(f"📉 {symbol}: Insufficient price movement ({price_change_pct:.1%})")
            return False
            
        logging.debug(f"✅ {symbol}: Alert conditions met")
        return True

    def _record_alert(self, symbol: str, price: float, volume: int, timestamp: datetime):
        """Record alert details"""
        history = self.alert_history.get(symbol)
        if history:
            history.last_price = price
            history.last_volume = volume
            history.last_alert_time = timestamp
            history.alert_count += 1
            
            if timestamp.date() == history.last_alert_date.date():
                history.daily_alerts += 1
            else:
                history.daily_alerts = 1
                
            history.last_alert_date = timestamp
        else:
            self.alert_history[symbol] = AlertHistory(
                last_price=price,
                last_volume=volume,
                last_alert_time=timestamp,
                alert_count=1,
                daily_alerts=1,
                last_alert_date=timestamp
            )
        
        logging.info(f"📝 Alert recorded for {symbol} - Price: ${price:.2f}, Volume: {volume:,}")

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
                    logging.info(f"🔍 Scraped {len(filtered_symbols)} valid symbols")
                    return filtered_symbols
                
        except Exception as e:
            logging.error(f"❌ Scraping failed: {str(e)}")
            
        return set()

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
                    logging.info(f"📝 Updated watchlist: {len(self.stocks)} symbols")
                    
        except Exception as e:
            logging.error(f"❌ Error updating watchlist: {str(e)}")

    def send_discord_alert(self, plan: TradingPlan, session: MarketSession):
        """Send Discord alert"""
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
                logging.info(f"📢 Discord alert sent for {plan.symbol}")
            else:
                logging.error(f"❌ Discord alert failed with status {response.status_code}")
                
        except Exception as e:
            logging.error(f"❌ Discord alert failed for {plan.symbol}: {str(e)}")

    def analyze_stock(self, symbol: str, session: MarketSession) -> Optional[TradingPlan]:
        """Analyze stock with enhanced LLM criteria"""
        try:
            # Skip if already have position
            if symbol in self.position_manager.positions:
                logging.debug(f"⏩ {symbol}: Already in position")
                return None
                
            # Basic validation
            if not self._is_valid_stock(symbol):
                return None
                
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1d', interval='1m')
            
            if hist.empty:
                logging.debug(f"❌ {symbol}: No minute data")
                return None
                
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].sum()
            
            try:
                shares_float = stock.info.get('floatShares', 0)
                if shares_float > TRADING_FILTERS['max_float']:
                    logging.info(f"❌ {symbol}: Float too large ({shares_float:,})")
                    return None
                float_category = self._categorize_float(shares_float)
            except:
                float_category = "UNKNOWN FLOAT"
                shares_float = float('inf')
                
            daily_hist = stock.history(period='5d')
            if daily_hist.empty:
                return None
                
            avg_volume = daily_hist['Volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            if (volume_ratio > TRADING_FILTERS['min_rel_volume'] and 
                current_volume >= TRADING_FILTERS['min_volume']):
                
                logging.info(f"💡 Strong setup detected for {symbol}:")
                logging.info(f"  - Volume: {volume_ratio:.1f}x average")
                logging.info(f"  - Float Category: {float_category}")
                logging.info(f"  - Current Price: ${current_price:.2f}")
                
                # LLM analysis for qualifying stocks
                if self._should_use_llm(symbol, volume_ratio, shares_float, session):
                    pattern, risk_level = self.llm.analyze_pattern(symbol, hist)
                    if pattern:
                        logging.info(f"🤖 LLM Analysis Complete for {symbol}:")
                        logging.info(f"  - Pattern: {pattern}")
                        logging.info(f"  - Risk Level: {risk_level}")
                        
                        exit_plan = self.llm.get_exit_plan(symbol, current_price, pattern, risk_level)
                        stop_loss = exit_plan['stop_loss']
                        profit_target = exit_plan['profit_target']
                        scale_out_levels = exit_plan['scale_out_levels']
                    else:
                        logging.info(f"❌ {symbol}: LLM analysis failed")
                        return None
                else:
                    pattern = None
                    risk_level = "MEDIUM"
                    stop_loss = current_price * 0.95
                    profit_target = current_price * 1.10
                    scale_out_levels = []
                    logging.info(f"⏩ {symbol}: Using default analysis (no LLM)")
                
                position_size = self._calculate_position_size(current_price, stop_loss)
                costs = self._calculate_trading_costs(current_price, position_size)

                plan = TradingPlan(
                    symbol=symbol,
                    why=f"Volume surge ({volume_ratio:.1f}x avg), {float_category}" + 
                        (f", LLM Analysis: {pattern}" if pattern else ""),
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
                
                # Add to position manager if LLM was used
                if pattern and scale_out_levels:
                    self.position_manager.add_position(
                        symbol, plan, risk_level, scale_out_levels
                    )
                    logging.info(f"✅ {symbol}: Added to position manager")
                
                logging.info(f"📋 Trading Plan Created for {symbol}")
                return plan
                    
        except Exception as e:
            logging.error(f"❌ Error analyzing {symbol}: {str(e)}")
            return None

    def _monitor_positions(self, session: MarketSession):
        """Check active positions for exits"""
        positions = self.position_manager.get_active_positions()
        if positions:
            logging.info(f"👀 Monitoring {len(positions)} active positions")
            
        for position in positions:
            if not self.position_manager.should_check_position(position.symbol):
                continue
                
            try:
                stock = yf.Ticker(position.symbol)
                current_data = stock.history(period='1d')
                if current_data.empty:
                    continue
                    
                current_price = current_data['Close'].iloc[-1]
                current_volume = current_data['Volume'].iloc[-1]
                
                # Calculate current P&L
                pnl_pct = (current_price/position.entry_price - 1) * 100
                logging.info(f"📊 {position.symbol} Position Update:")
                logging.info(f"  - Current P&L: {pnl_pct:.1f}%")
                logging.info(f"  - Entry: ${position.entry_price:.2f}")
                logging.info(f"  - Current: ${current_price:.2f}")
                
                # Get LLM advice
                advice = self.llm.monitor_position(position, current_price, current_volume)
                position.last_llm_advice = advice
                
                # Handle recommendations
                if advice['urgency'] == 'HIGH':
                    message = (
                        f"🚨 URGENT POSITION UPDATE: {position.symbol}\n"
                        f"Action: {advice['action']}\n"
                        f"Reason: {advice['reason']}\n"
                        f"Current Price: ${current_price:.2f}\n"
                        f"Entry Price: ${position.entry_price:.2f}\n"
                        f"P&L: {pnl_pct:.1f}%"
                    )
                    
                    webhook = DiscordWebhook(
                        url=self.webhook_url,
                        content=message
                    )
                    webhook.execute()
                    logging.info(f"🚨 Urgent update sent for {position.symbol}")
                
                self.position_manager.update_position_check(position.symbol)
                
            except Exception as e:
                logging.error(f"❌ Error monitoring {position.symbol}: {str(e)}")

    def run(self):
        """Main system loop with position monitoring"""
        logging.info(f"🚀 Starting Trading Scanner with {self.llm.model}")
        
        while True:
            try:
                session = self.get_market_session()
                logging.info(f"⏰ Current session: {session.value}")
                
                if session == MarketSession.CLOSED:
                    logging.info("💤 Market closed, waiting...")
                    time.sleep(300)
                    continue
                    
                # Monitor existing positions first
                self._monitor_positions(session)
                
                # Avoid midday chop
                if session == MarketSession.MIDDAY:
                    logging.info("⏸️ Midday session - monitoring only")
                    time.sleep(self._get_scraping_interval(session))
                    continue
                
                # Update watchlist
                self.update_watchlist(session)
                
                # Process high priority symbols first
                priority_symbols = list(self.position_manager.positions.keys())
                remaining_symbols = list(self.stocks - set(priority_symbols))
                
                if priority_symbols:
                    logging.info(f"🎯 Processing {len(priority_symbols)} priority symbols")
                    
                # Monitor priority symbols more frequently
                for symbol in priority_symbols:
                    plan = self.analyze_stock(symbol, session)
                    if plan:
                        self.send_discord_alert(plan, session)
                    time.sleep(1)
                
                # Process remaining symbols in chunks
                if remaining_symbols:
                    logging.info(f"📝 Processing {len(remaining_symbols)} symbols")
                    
                for chunk in self._chunk_symbols(remaining_symbols):
                    for symbol in chunk:
                        plan = self.analyze_stock(symbol, session)
                        if plan:
                            self.send_discord_alert(plan, session)
                    time.sleep(2)
                    
                time.sleep(self._get_scraping_interval(session))
                
            except KeyboardInterrupt:
                logging.info("🛑 Shutting down scanner...")
                self.llm.executor.shutdown()
                self.position_manager.executor.shutdown()
                break
            except Exception as e:
                logging.error(f"❌ System error: {str(e)}")
                time.sleep(60)

if __name__ == "__main__":
    system = TradingSystem()
    system.run()
