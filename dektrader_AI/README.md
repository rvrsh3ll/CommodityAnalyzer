# AI Day Trading Scanner

Real-time trading scanner using Llama 3 LLM for advanced pattern recognition and trading signals. Built for low-float stock momentum trading.

## Key Features

- Real-time stock scanning and analysis
- Low float focus with volume surge detection
- Pattern recognition using Llama 3
- Position monitoring with AI-driven exit signals
- Discord alerts for entries and exits
- Smart session management (pre-market, market open, etc.)
- Scale-out management for position exits

## System Requirements

- Python 3.10 or higher
- 16GB+ RAM recommended
- NVIDIA GPU recommended (4090 or similar)
- Ubuntu/Linux OS

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-trading-scanner.git
cd ai-trading-scanner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama and Llama 3:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama 3 model
ollama pull llama3:latest
```

4. Set up Discord webhook:
- Create a webhook in your Discord server
- Copy webhook URL to `config.py`

5. Run the scanner:
```bash
python trading_scanner.py
```

## Configuration

Edit `config.py` to customize:

```python
# Trading filters
TRADING_FILTERS = {
    'min_price': 2.00,     
    'max_price': 20.00,    
    'min_volume': 500000,  
    'max_float': 20000000  
}

# LLM settings
LLM_CONFIG = {
    'model': 'llama3:latest',
    'timeout': 2.0
}
```

## Discord Alerts

The scanner sends detailed alerts including:
- Entry and exit points
- Position size and risk metrics
- Pattern analysis from Llama 3
- Trading costs breakdown
- Real-time monitoring updates

## Trading Sessions

1. Pre-Market (4:00 AM - 9:30 AM)
2. Market Open (9:30 AM - 11:00 AM) - Primary trading window
3. Midday (11:00 AM - 3:00 PM) - Reduced scanning
4. Market Close (3:00 PM - 4:00 PM)
5. After Hours (4:00 PM - 8:00 PM)

## Model Performance

Llama 3 provides fast inference:
- Average response time: ~0.3-0.4 seconds
- Pattern recognition accuracy: High
- Resource usage: Moderate

## Safety Features

- Maximum position limits
- Per-trade risk controls
- Session-based trading restrictions
- Minimum price movement filters
- Alert frequency limits

## License

MIT License

## Disclaimer

This software is for educational purposes only. Always perform your own due diligence and adhere to proper risk management when trading.