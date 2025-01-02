#!/usr/bin/env python3
# Discord webhook URL for alerts
DISCORD_WEBHOOK_URL = "YOUR_WEBHOOK_URL_HERE"

# Trading parameters
TRADING_FILTERS = {
    'min_price': 2.00,      # Min $2
    'max_price': 20.00,     # $20 max price
    'min_volume': 500000,   # Minimum daily volume
    'max_float': 20000000,  # 20M float max
    'min_rel_volume': 5.0   # 500% relative volume minimum
}

# LLM parameters
LLM_FILTERS = {
    'min_volume_ratio': 7.0,    # 700% volume
    'max_float': 10_000_000,    # 10M float maximum
    'min_price_change': 0.05    # 5% minimum move
}

# LLM configuration
LLM_CONFIG = {
    'model': 'llama3:latest',  # Using Llama 3
    'timeout': 2.0,            # seconds
    'max_workers': 2
}

# Position monitoring
MONITORING_CONFIG = {
    'interval': 300,           # 5 minutes
    'max_positions': 5,        # Maximum concurrent positions
    'min_check_interval': 60   # Minimum seconds between checks
}

# Trading sessions
TRADING_SESSIONS = {
    'pre_market_start': '04:00',
    'market_open': '09:30',
    'midday_start': '11:00',
    'market_close': '16:00',
    'after_hours_end': '20:00'
}

# Alert settings
ALERT_CONFIG = {
    'max_daily_alerts': 5,
    'min_alert_interval': 900,  # 15 minutes
    'min_price_movement': 0.03  # 3%
}

# Risk management
RISK_CONFIG = {
    'max_loss_per_trade': 100,  # Maximum loss in dollars
    'max_account_risk': 0.02    # Maximum account risk per trade
}