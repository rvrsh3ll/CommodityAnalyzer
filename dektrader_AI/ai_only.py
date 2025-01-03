import subprocess
import yfinance as yf
import pandas as pd
import numpy as np
import ollama
import asyncio
import logging
from datetime import datetime, timedelta
import pytz
import colorama
from tabulate import tabulate

# Initialize colorama for colored output
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_scanner.log'),
        logging.StreamHandler()
    ]
)

def is_market_open():
    """
    Check if current time is during US stock market trading hours.
    Market hours are 9:30 AM to 4:00 PM Eastern Time, Monday through Friday.
    """
    # Set timezone to Eastern Time
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)

    # Check if it's a weekday (Monday = 0, Friday = 4)
    is_weekday = now.weekday() < 5

    # Check if time is between 9:30 AM and 4:00 PM
    market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    is_market_hours = market_open_time <= now < market_close_time

    return is_weekday and is_market_hours

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
Confidence: [0-100%]

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

class OutputFormatter:
    @staticmethod
    def format_trading_setup(setup):
        """Nicely format the trading setup"""
        try:
            lines = setup.split("\n")
            
            # Prepare a table for better readability
            table_data = [
                ["Symbol", lines[0].split(": ")[1]],
                ["Entry", lines[2].split(": ")[1]],
                ["Target", lines[3].split(": ")[1]],
                ["Stop Loss", lines[4].split(": ")[1]],
                ["Position Size", lines[5].split(": ")[1]],
                ["Confidence", lines[7].split(": ")[1]]
            ]
            
            # Color-code based on confidence
            confidence = int(lines[7].split(": ")[1].rstrip('%'))
            if confidence > 80:
                confidence_color = colorama.Fore.GREEN
            elif confidence > 60:
                confidence_color = colorama.Fore.YELLOW
            else:
                confidence_color = colorama.Fore.RED
            
            # Create formatted output
            formatted_output = (
                f"\n{colorama.Fore.CYAN}🔍 TRADING SETUP FOUND {colorama.Fore.RESET}\n"
                f"{tabulate(table_data, headers=['Detail', 'Value'], tablefmt='fancy_grid')}\n\n"
                f"🔬 {colorama.Fore.MAGENTA}Reason:{colorama.Fore.RESET} {lines[6]}\n"
                f"📊 {colorama.Fore.CYAN}Confidence:{confidence_color} {lines[7]} {colorama.Fore.RESET}"
            )
            
            return formatted_output
        except Exception as e:
            logging.error(f"Error formatting setup: {str(e)}")
            return setup

async def main():
    scanner = StockScanner()
    analyzer = StockAnalyzer()
    trader = TradingAnalyst()
    formatter = OutputFormatter()

    logging.info(f"{colorama.Fore.GREEN}Stock Scanner Initialized{colorama.Fore.RESET}")

    while True:
        try:
            # Only scan during market hours
            if is_market_open():
                # Get symbols
                symbols = scanner.get_symbols()
                logging.info(f"{colorama.Fore.CYAN}Found {len(symbols)} symbols to analyze{colorama.Fore.RESET}")

                # Analyze each symbol asynchronously
                tasks = []
                for symbol in symbols:
                    task = asyncio.create_task(analyze_symbol(symbol, analyzer, trader, formatter))
                    tasks.append(task)

                await asyncio.gather(*tasks)
            else:
                # Market closed message with current time
                eastern = pytz.timezone('US/Eastern')
                current_time = datetime.now(eastern)
                logging.info(
                    f"{colorama.Fore.YELLOW}Market is closed. "
                    f"Current time: {current_time.strftime('%I:%M %p %Z on %A')}"
                    f"{colorama.Fore.RESET}"
                )

            # Wait before next scan
            await asyncio.sleep(60)  # 1 minute delay

        except Exception as e:
            logging.error(f"{colorama.Fore.RED}Main loop error: {str(e)}{colorama.Fore.RESET}")
            await asyncio.sleep(60)

async def analyze_symbol(symbol, analyzer, trader, formatter):
    # Get technical analysis data
    data = analyzer.analyze_stock(symbol)

    if data:
        # Get trading setup from LLM
        setup = await trader.analyze_setup(data)

        if setup and "NO SETUP" not in setup:
            try:
                confidence = float(setup.split("Confidence: ")[1].split("%")[0])
                if confidence > 70:
                    # Format and print setup
                    formatted_setup = formatter.format_trading_setup(setup)
                    print(formatted_setup)
            except Exception as e:
                logging.error(f"Error processing setup for {symbol}: {str(e)}")

if __name__ == "__main__":
    # Add dependencies notice
    print(f"{colorama.Fore.YELLOW}Note: Requires 'colorama', 'tabulate', 'pytz' libraries.{colorama.Fore.RESET}")
    print(f"{colorama.Fore.YELLOW}Install with: pip install colorama tabulate pytz{colorama.Fore.RESET}")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{colorama.Fore.RED}Stock scanner stopped by user.{colorama.Fore.RESET}")
