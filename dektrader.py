import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

@dataclass
class Setup:
    time: datetime
    type: str  # 'Volume Spike', 'Consolidation Breakout', 'Secondary'
    price: float
    volume_ratio: float
    avg_volume: float
    risk_reward: float

@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_type: Optional[str] = None
    setup_type: str = ""

class AutomatedDeckStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.setups: List[Setup] = []
        self.trades: List[Trade] = []
        self.consolidations: List[Dict] = []
        
        # Configure logging
        logging.basicConfig(
            filename=f'{symbol}_trades.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        # Volume analysis
        df['VolumeSMA20'] = df['Volume'].rolling(window=20).mean()
        df['VolumeRatio'] = df['Volume'] / df['VolumeSMA20']
        
        # Price action
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR20'] = df['TR'].rolling(window=20).mean()
        df['PriceRange'] = df['High'] - df['Low']
        df['RangeRatio'] = df['PriceRange'] / df['ATR20']
        
        # Consolidation detection
        df['Consolidating'] = df['RangeRatio'] < 0.5
        
        return df
        
    def find_consolidations(self, df: pd.DataFrame, min_bars: int = 5):
        """Identify consolidation periods"""
        consolidation_start = None
        consol_bars = 0
        
        for i in range(len(df)):
            if df['Consolidating'].iloc[i]:
                if consolidation_start is None:
                    consolidation_start = df.index[i]
                consol_bars += 1
            else:
                if consol_bars >= min_bars:
                    consol_data = {
                        'start': consolidation_start,
                        'end': df.index[i-1],
                        'duration': consol_bars,
                        'avg_price': df['Close'].iloc[i-consol_bars:i].mean(),
                        'volume_avg': df['Volume'].iloc[i-consol_bars:i].mean()
                    }
                    self.consolidations.append(consol_data)
                consolidation_start = None
                consol_bars = 0
                
    def identify_setups(self, df: pd.DataFrame):
        """Find all potential trade setups"""
        # 1. Volume spike setups
        volume_spikes = df[df['VolumeRatio'] >= 5]
        for time, row in volume_spikes.iterrows():
            self.setups.append(Setup(
                time=time,
                type='Volume Spike',
                price=row['Close'],
                volume_ratio=row['VolumeRatio'],
                avg_volume=row['VolumeSMA20'],
                risk_reward=2.0  # 10% gain vs 5% loss
            ))
        
        # 2. Consolidation breakout setups
        for consol in self.consolidations:
            # Look for volume spike after consolidation
            post_consol = df[consol['end']:].iloc[1:5]  # Check next 5 bars
            if not post_consol.empty:
                breakout = post_consol[post_consol['VolumeRatio'] >= 3]  # Lower threshold for breakouts
                if not breakout.empty:
                    self.setups.append(Setup(
                        time=breakout.index[0],
                        type='Consolidation Breakout',
                        price=breakout['Close'].iloc[0],
                        volume_ratio=breakout['VolumeRatio'].iloc[0],
                        avg_volume=breakout['VolumeSMA20'].iloc[0],
                        risk_reward=2.0
                    ))
                    
    def simulate_trades(self, df: pd.DataFrame):
        """Simulate trades based on setups"""
        for setup in self.setups:
            # Get data after setup
            future_data = df[setup.time:]
            if len(future_data) < 2:  # Need at least 2 bars for entry/exit
                continue
                
            # Calculate entry on next bar
            entry_bar = future_data.iloc[1]
            entry_price = entry_bar['Open']
            stop_loss = entry_price * 0.95  # 5% stop
            target = entry_price * 1.10     # 10% target
            
            trade = Trade(
                entry_time=entry_bar.name,
                entry_price=entry_price,
                setup_type=setup.type
            )
            
            # Simulate trade
            for i in range(2, len(future_data)):
                bar = future_data.iloc[i]
                
                if bar['Low'] <= stop_loss:
                    trade.exit_time = bar.name
                    trade.exit_price = stop_loss
                    trade.pnl = -5.0  # 5% loss
                    trade.exit_type = 'Stop Loss'
                    break
                elif bar['High'] >= target:
                    trade.exit_time = bar.name
                    trade.exit_price = target
                    trade.pnl = 10.0  # 10% gain
                    trade.exit_type = 'Target'
                    break
            
            if trade.exit_time:  # Only add completed trades
                self.trades.append(trade)
                logging.info(f"Trade completed - {setup.type}: Entry ${entry_price:.4f}, Exit ${trade.exit_price:.4f}, P/L: {trade.pnl}%")
                
    def analyze_stock(self, date_str: str):
        """Run complete analysis for a given date"""
        print(f"\nAnalyzing {self.symbol} for {date_str}")
        
        # Get data
        start_date = datetime.strptime(date_str, '%Y-%m-%d')
        end_date = start_date + timedelta(days=1)
        
        df = yf.Ticker(self.symbol).history(
            start=start_date,
            end=end_date,
            interval='1m',
            prepost=True
        )
        
        if df.empty:
            print("No data available")
            return
            
        # Run analysis
        df = self.preprocess_data(df)
        self.find_consolidations(df)
        self.identify_setups(df)
        self.simulate_trades(df)
        
        # Print results
        print("\nTrade Summary:")
        if self.trades:
            winning_trades = len([t for t in self.trades if t.pnl > 0])
            print(f"Total Trades: {len(self.trades)}")
            print(f"Winning Trades: {winning_trades}")
            print(f"Win Rate: {(winning_trades/len(self.trades)*100):.1f}%")
            
            total_pnl = sum(t.pnl for t in self.trades if t.pnl)
            print(f"Total P/L: {total_pnl:.1f}%")
            
            print("\nDetailed Trade Report:")
            for i, trade in enumerate(self.trades, 1):
                print(f"\nTrade #{i} ({trade.setup_type}):")
                print(f"Entry: {trade.entry_time.strftime('%I:%M:%S %p')} @ ${trade.entry_price:.4f}")
                print(f"Exit: {trade.exit_time.strftime('%I:%M:%S %p')} @ ${trade.exit_price:.4f}")
                print(f"Result: {trade.exit_type} ({trade.pnl:+.1f}%)")
        else:
            print("No trades executed")
            
        print("\nConsolidation Periods:")
        for i, consol in enumerate(self.consolidations, 1):
            print(f"\nPeriod #{i}:")
            print(f"Start: {consol['start'].strftime('%I:%M:%S %p')}")
            print(f"End: {consol['end'].strftime('%I:%M:%S %p')}")
            print(f"Duration: {consol['duration']} minutes")
            print(f"Average Price: ${consol['avg_price']:.4f}")

if __name__ == "__main__":
    strategy = AutomatedDeckStrategy("PRFX")
    strategy.analyze_stock("2024-12-19")
