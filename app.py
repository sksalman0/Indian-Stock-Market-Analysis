from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return pd.Series(data).rolling(window=period).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = pd.Series(data).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_momentum(data, period=10):
    """Calculate Momentum"""
    return pd.Series(data).diff(period)

def get_indian_market_hours():
    """Check if it's currently Indian market hours (9:15 AM to 3:30 PM IST on weekdays)"""
    india_tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(india_tz)
    
    if current_time.weekday() > 4:  # Weekend check
        return False
        
    market_start = current_time.replace(hour=9, minute=15, second=0)
    market_end = current_time.replace(hour=15, minute=30, second=0)
    
    return market_start <= current_time <= market_end

def format_indian_ticker(ticker, exchange):
    """Format ticker symbol for Indian stocks based on exchange"""
    suffix = '.NS' if exchange == 'NSE' else '.BO'
    return f"{ticker}{suffix}" if not ticker.endswith(suffix) else ticker

def get_stock_data(ticker, exchange, period="1d", interval="5m"):
    """Get stock data with error handling and exchange selection"""
    try:
        formatted_ticker = format_indian_ticker(ticker, exchange)
        stock = yf.Ticker(formatted_ticker)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            alt_exchange = 'BSE' if exchange == 'NSE' else 'NSE'
            alt_ticker = format_indian_ticker(ticker, alt_exchange)
            stock = yf.Ticker(alt_ticker)
            hist = stock.history(period=period, interval=interval)
            
        return stock, hist
    except Exception as e:
        return None, None

def calculate_intraday_score(ticker, exchange):
    """Calculate intraday trading score (0-100)"""
    try:
        stock, hist = get_stock_data(ticker, exchange, period="1d", interval="5m")
        if stock is None or len(hist) < 5:
            return None
            
        score = 0
        closes = hist['Close'].values
        
        # 1. Volume Analysis (25 points)
        avg_volume = hist['Volume'].mean()
        current_volume = hist['Volume'][-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        score += min(25, max(0, volume_ratio * 12.5))
        
        # 2. Price Movement (25 points)
        price_change = (closes[-1] - closes[0]) / closes[0] * 100
        score += min(25, max(0, abs(price_change) * 2.5))
        
        # 3. Volatility (25 points)
        high_low_ratio = hist['High'].max() / hist['Low'].min()
        volatility = high_low_ratio - 1
        score += min(25, max(0, volatility * 50))
        
        # 4. Technical Indicators (25 points)
        if len(closes) >= 14:
            rsi = calculate_rsi(closes)[-1]
            rsi_score = 25 - abs(rsi - 50) / 2
            score += max(0, rsi_score)
        
        return round(score, 2)
    except Exception as e:
        return None

def calculate_short_term_score(ticker, exchange):
    """Calculate short-term trading score (0-100)"""
    try:
        stock, hist = get_stock_data(ticker, exchange, period="5d", interval="1h")
        if stock is None or len(hist) < 5:
            return None
            
        score = 0
        closes = hist['Close'].values
        
        # 1. Moving Average Analysis (25 points)
        if len(closes) >= 20:
            sma20 = calculate_sma(closes, 20).iloc[-1]
            current_price = closes[-1]
            ma_ratio = current_price / sma20
            score += min(25, max(0, (ma_ratio - 0.95) * 125))
        
        # 2. RSI Analysis (25 points)
        if len(closes) >= 14:
            rsi = calculate_rsi(closes).iloc[-1]
            if rsi < 30:  # Oversold
                score += 25
            elif rsi > 70:  # Overbought
                score += 10
            else:
                score += 15
        
        # 3. Volume Trend (25 points)
        volume_ma = pd.Series(hist['Volume']).rolling(window=5).mean()
        current_volume = hist['Volume'][-1]
        vol_ratio = current_volume / volume_ma.mean() if volume_ma.mean() > 0 else 0
        score += min(25, max(0, vol_ratio * 12.5))
        
        # 4. Price Momentum (25 points)
        if len(closes) >= 10:
            momentum = calculate_momentum(closes).iloc[-1]
            score += min(25, max(0, abs(momentum) * 0.5))
        
        return round(score, 2)
    except Exception as e:
        return None

def calculate_mid_term_score(ticker, exchange):
    """Calculate mid-term trading score (0-100)"""
    try:
        stock, hist = get_stock_data(ticker, exchange, period="60d", interval="1d")
        if stock is None or len(hist) < 20:
            return None
            
        score = 0
        closes = hist['Close'].values
        
        # 1. Trend Analysis (25 points)
        if len(closes) >= 50:
            sma50 = calculate_sma(closes, 50).iloc[-1]
            current_price = closes[-1]
            trend_strength = (current_price / sma50 - 1) * 100
            score += min(25, max(0, abs(trend_strength) * 2))
        
        # 2. Financial Metrics (25 points)
        try:
            info = stock.info
            pe_ratio = info.get('forwardPE', 0)
            if 0 < pe_ratio < 30:
                score += min(25, max(0, (30 - pe_ratio)))
        except:
            score += 12.5
        
        # 3. Volatility (25 points)
        if len(closes) >= 20:
            atr = calculate_atr(hist['High'], hist['Low'], closes).iloc[-1]
            norm_volatility = atr / closes[-1]
            score += min(25, max(0, (0.1 - norm_volatility) * 250))
        
        # 4. Relative Strength (25 points)
        if len(closes) >= 20:
            returns = pd.Series(closes).pct_change()
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
            score += min(25, max(0, sharpe * 12.5))
        
        return round(score, 2)
    except Exception as e:
        return None

def calculate_long_term_score(ticker, exchange):
    """Calculate long-term investment score (0-100)"""
    try:
        stock, hist = get_stock_data(ticker, exchange, period="1y", interval="1d")
        if stock is None or len(hist) < 50:
            return None
            
        score = 0
        closes = hist['Close'].values
        
        # 1. Fundamental Analysis (25 points)
        try:
            info = stock.info
            pe_ratio = info.get('forwardPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            
            if 0 < pe_ratio < 25 and 0 < pb_ratio < 5:
                fundamental_score = (25 - pe_ratio) / 25 * 12.5 + (5 - pb_ratio) / 5 * 12.5
                score += min(25, max(0, fundamental_score))
        except:
            score += 12.5
        
        # 2. Growth Metrics (25 points)
        if len(closes) >= 200:
            yearly_return = (closes[-1] / closes[0] - 1) * 100
            score += min(25, max(0, yearly_return))
        
        # 3. Financial Health (25 points)
        try:
            info = stock.info
            debt_to_equity = info.get('debtToEquity', 0)
            current_ratio = info.get('currentRatio', 0)
            
            if debt_to_equity > 0 and current_ratio > 0:
                health_score = (1 - min(debt_to_equity, 200) / 200) * 12.5
                health_score += min(current_ratio, 2) / 2 * 12.5
                score += health_score
        except:
            score += 12.5
        
        # 4. Dividend and Stability (25 points)
        try:
            info = stock.info
            dividend_yield = info.get('dividendYield', 0)
            beta = info.get('beta', 1)
            
            if dividend_yield > 0:
                score += min(12.5, dividend_yield * 2.5)
            if 0 < beta < 2:
                score += min(12.5, (2 - beta) * 6.25)
        except:
            score += 12.5
        
        return round(score, 2)
    except Exception as e:
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form.get('ticker', '')
    timeframe = request.form.get('timeframe', 'intraday')
    exchange = request.form.get('exchange', 'NSE')
    
    if not ticker:
        return jsonify({'error': 'Please enter a valid ticker symbol'})
    
    if timeframe == 'intraday' and not get_indian_market_hours():
        return jsonify({
            'error': 'Intraday analysis is only available during Indian market hours (9:15 AM to 3:30 PM IST, Monday-Friday)'
        })
    
    score_functions = {
        'intraday': calculate_intraday_score,
        'short_term': calculate_short_term_score,
        'mid_term': calculate_mid_term_score,
        'long_term': calculate_long_term_score
    }
    
    score = score_functions[timeframe](ticker, exchange)
    
    if score is None:
        return jsonify({'error': f'Unable to calculate {timeframe} score for {ticker} on {exchange}'})
    
    return jsonify({
        'ticker': ticker,
        'exchange': exchange,
        'timeframe': timeframe,
        'score': score
    })

if __name__ == '__main__':
    app.run(debug=True)