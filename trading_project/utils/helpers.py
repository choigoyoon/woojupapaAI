# -*- coding: utf-8 -*-
from datetime import datetime
import pandas as pd

def format_number(number, decimals=8):
    """숫자 포맷팅"""
    return f"{number:.{decimals}f}"

def get_timestamp():
    """현재 시간 문자열"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def print_analysis(symbol, market_type, exchange, timeframe, analysis, signal):
    """분석 결과 출력"""
    print(f"\n{'='*50}")
    print(f"Symbol: {symbol} ({market_type})")
    print(f"Exchange: {exchange}")
    print(f"Timeframe: {timeframe}")
    print(f"Time: {get_timestamp()}")
    print(f"Price: {format_number(analysis['close'], 8)}")
    print("\nIndicators:")
    
    indicators = analysis['indicators']
    
    print(f"MACD: {format_number(indicators['MACD']['macd'])} "
          f"(Signal: {format_number(indicators['MACD']['signal'])}, "
          f"Cross: {indicators['MACD']['cross']})")
    
    print(f"RSI: {format_number(indicators['RSI']['value'])} "
          f"({indicators['RSI']['signal']})")
    
    print(f"CCI: {format_number(indicators['CCI']['value'])} "
          f"({indicators['CCI']['signal']})")
    
    print(f"Stoch RSI: K={format_number(indicators['Stoch_RSI']['k'])} "
          f"D={format_number(indicators['Stoch_RSI']['d'])} "
          f"({indicators['Stoch_RSI']['signal']})")
    
    print(f"Williams %R: {format_number(indicators['Williams_R']['value'])} "
          f"({indicators['Williams_R']['signal']})")
    
    print(f"Squeeze: {'ON' if indicators['Squeeze_Momentum']['squeeze_on'] else 'OFF'} "
          f"({indicators['Squeeze_Momentum']['signal']})")
    
    print(f"\nOverall Signal: {signal['signal']} (Strength: {format_number(signal['strength']*100, 2)}%)")
    print(f"{'='*50}\n")
