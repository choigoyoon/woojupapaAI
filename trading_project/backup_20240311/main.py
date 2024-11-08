# -*- coding: utf-8 -*-
import time
from datetime import datetime
from core.exchange import ExchangeHandler
from core.analyzer import MarketAnalyzer
from utils.helpers import print_analysis
from config.settings import TIMEFRAMES, UPDATE_INTERVAL

def main():
    print("Starting program...")
    
    exchange_handler = ExchangeHandler()
    analyzer = MarketAnalyzer()
    
    print("Getting symbols...")
    all_symbols = exchange_handler.get_all_symbols()
    
    print("\nStarting analysis...")
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n=== Analysis Time: {current_time} ===")
        
        # 현물 시장 분석
        for exchange_name, symbols in all_symbols['spot'].items():
            exchange = exchange_handler.spot_exchanges.get(exchange_name)
            if not exchange:
                continue
            
            print(f"\n--- {exchange_name} Spot Market ---")
            # 처음 5개 심볼만 분석 (테스트용)
            for symbol in symbols[:5]:
                try:
                    ohlcv = exchange_handler.fetch_ohlcv(exchange, symbol, '15m')
                    if ohlcv:
                        analysis = analyzer.analyze_market(ohlcv)
                        if analysis:
                            signal = analyzer.generate_signal(analysis)
                            print(f"\n{symbol}:")
                            print(f"Signal: {signal['signal']} (Strength: {signal['strength']:.2f})")
                            print(f"Price: {ohlcv[-1][4]:.2f} USDT")
                except Exception as e:
                    print(f"Error analyzing {symbol}: {str(e)}")
        
        print(f"\nWaiting {UPDATE_INTERVAL/60} minutes for next update...")
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    main()
