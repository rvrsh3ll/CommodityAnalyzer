import subprocess
import yfinance as yf
import pandas as pd
import numpy as np
import ollama
import asyncio
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StockScanner:
    def __init__(self):
        self.curl_command = """curl -s 'https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count=100&scrIds=DAY_GAINERS&formatted=true&start=0&fields=symbol,regularMarketPrice,regularMarketChangePercent,regularMarketVolume' -H 'User-Agent: Mozilla/5.0' | jq -r '.finance.result[0].quotes[].symbol' ; curl -s 'https://finance.yahoo.com/markets/stocks/trending/' -H 'User-Agent: Mozilla/5.0' | grep -oP '"symbol":"\K[A-Z]+(?=")' ; curl -s 'https://finance.yahoo.com/markets/stocks/most-active/?start=0&count=100' -H 'User-Agent: Mozilla/5.0' | grep -oP '"symbol":"\K[A-Z]+(?=")' ; curl -s 'https://finance.yahoo.com/markets/stocks/52-week-gainers/?start=0&count=50' -H 'User-Agent: Mozilla/5.0' | grep -oP '"symbol":"\K[A-Z]+(?=")'"""

    def get_symbols(self):
        """Get stock symbols from curl command"""
        try:
            result = subprocess.run(
                self.curl_command,
                shell=True,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logging.error(f"Curl command failed: {result.stderr}")
                return []

            # Split output into lines and remove duplicates
            symbols = list(set(result.stdout.strip().split('\n')))
            return symbols

        except Exception as e:
            logging.error(f"Failed to get symbols: {str(e)}")
            return []

class StockAnalyzer:
    def __init__(self):
        self.trading_filters = {
            'min_price': 2.00,
            'max_price': 20.00,
            'min_volume': 500000,
            'min_rel_volume': 5.0
        }

    def calculate_technical_indicators(self, df):
        """Calculate various technical indicators"""
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # VWAP
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()

        # ATR for volatility
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(14).mean()

        return df

    def analyze_stock(self, symbol):
        """Analyze a single stock"""
        try:
            # Get stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1d', interval='1m')

            if hist.empty:
                return None

            # Get current price and volume
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].sum()
            avg_volume = stock.info.get('averageVolume', 0)
            rel_volume = current_volume / avg_volume if avg_volume > 0 else 0

            # Apply filters
            if not self._passes_filters(current_price, current_volume, rel_volume):
                return None

            # Calculate indicators
            tech_data = self.calculate_technical_indicators(hist)

            # Format data for LLM
            data = self._format_data_for_llm(symbol, tech_data, stock.info)

            return data

        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {str(e)}")
            return None

    def _passes_filters(self, price, volume, rel_volume):
        """Check if stock passes basic filters"""
        if price < self.trading_filters['min_price']:
            return False
        if price > self.trading_filters['max_price']:
            return False
        if volume < self.trading_filters['min_volume']:
            return False
        if rel_volume < self.trading_filters['min_rel_volume']:
            return False
        return True

    def _format_data_for_llm(self, symbol, tech_data, stock_info):
        """Format stock data for LLM analysis"""
        recent = tech_data.tail(5)

        return {
            'symbol': symbol,
            'current_price': tech_data['Close'].iloc[-1],
            'rsi': tech_data['RSI'].iloc[-1],
            'vwap': tech_data['VWAP'].iloc[-1],
            'sma20': tech_data['SMA_20'].iloc[-1],
            'ema9': tech_data['EMA_9'].iloc[-1],
            'atr': tech_data['ATR'].iloc[-1],
            'volume': stock_info.get('volume', 0),
            'avg_volume': stock_info.get('averageVolume', 0),
            'market_cap': stock_info.get('marketCap', 0),
            'beta': stock_info.get('beta', 0),
            'recent_price_action': recent[['Open', 'High', 'Low', 'Close', 'Volume']].to_dict('records')
        }

class TradingAnalyst:
    def __init__(self):
        self.model = "llama3:latest"

    def generate_prompt(self, data):
        """Generate prompt for LLM analysis"""
        prompt = f"""Act as an expert day trader. Analyze this stock for a potential day trade setup.

Symbol: {data['symbol']}
Current Price: ${data['current_price']:.2f}

Technical Indicators:
- RSI: {data['rsi']:.2f}
- VWAP: ${data['vwap']:.2f}
- SMA20: ${data['sma20']:.2f}
- EMA9: ${data['ema9']:.2f}
- ATR: ${data['atr']:.2f}

Volume Analysis:
- Current Volume: {data['volume']:,}
- Average Volume: {data['avg_volume']:,}
- Relative Volume: {data['volume']/data['avg_volume']:.1f}x

Recent price action:
{data['recent_price_action']}

Based on this data, determine if this stock is worth trading right now.
If yes, provide a trading setup in EXACTLY this format:

TRADING SETUP: {data['symbol']}
Entry: $PRICE
Target: $PRICE
Stop: $PRICE
Size: # shares
Reason: [2-3 sentence explanation]

If no clear setup exists, respond with 'NO SETUP'."""

        return prompt

    async def analyze_setup(self, data):
        """Get trading analysis from LLM"""
        try:
            prompt = self.generate_prompt(data)
            response = ollama.generate(
                model=self.model,
                prompt=prompt
            )
            return response['response']
        except Exception as e:
            logging.error(f"LLM analysis failed: {str(e)}")
            return "NO SETUP"

async def main():
    scanner = StockScanner()
    analyzer = StockAnalyzer()
    trader = TradingAnalyst()

    while True:
        try:
            # Get symbols
            symbols = scanner.get_symbols()
            logging.info(f"Found {len(symbols)} symbols to analyze")

            # Analyze each symbol
            for symbol in symbols:
                # Get technical analysis data
                data = analyzer.analyze_stock(symbol)

                if data:
                    # Get trading setup from LLM
                    setup = await trader.analyze_setup(data)

                    if setup and "NO SETUP" not in setup:
                        print(f"\n{setup}")

            # Wait before next scan
            await asyncio.sleep(60)  # 1 minute delay

        except Exception as e:
            logging.error(f"Main loop error: {str(e)}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
