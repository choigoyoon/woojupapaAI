# -*- coding: utf-8 -*-
import ccxt
from config.api_keys import EXCHANGE_CONFIGS
from config.settings import EXCHANGES

class ExchangeHandler:
    def __init__(self):
        self.spot_exchanges = {}
        self.futures_exchanges = {}
        self.initialize_exchanges()

    def initialize_exchanges(self):
        # 현물 거래소 초기화 (API 키 없이)
        for exchange_id in EXCHANGES['spot']:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                self.spot_exchanges[exchange_id] = exchange_class()
            except Exception as e:
                print(f"Error initializing {exchange_id} spot: {str(e)}")

        # 선물 거래소 초기화 (API 키 없이)
        for exchange_id in EXCHANGES['futures']:
            try:
                if exchange_id == 'binance':
                    exchange_class = getattr(ccxt, 'binanceusdm')
                else:
                    exchange_class = getattr(ccxt, exchange_id)
                self.futures_exchanges[exchange_id] = exchange_class()
            except Exception as e:
                print(f"Error initializing {exchange_id} futures: {str(e)}")

    def get_all_symbols(self):
        all_symbols = {'spot': {}, 'futures': {}}
        
        # 현물 심볼 조회
        for name, exchange in self.spot_exchanges.items():
            try:
                markets = exchange.load_markets()
                symbols = [symbol for symbol in markets.keys() 
                          if symbol.endswith('/USDT') and 
                          not any(x in symbol for x in ['UP/', 'DOWN/', 'BULL/', 'BEAR/'])]
                all_symbols['spot'][name] = symbols
                print(f"{name} spot: {len(symbols)} symbols loaded")
            except Exception as e:
                print(f"Error loading {name} spot symbols: {str(e)}")

        # 선물 심볼 조회
        for name, exchange in self.futures_exchanges.items():
            try:
                markets = exchange.load_markets()
                symbols = [symbol for symbol in markets.keys() 
                          if symbol.endswith('/USDT:USDT') or symbol.endswith('/USDT')]
                all_symbols['futures'][name] = symbols
                print(f"{name} futures: {len(symbols)} symbols loaded")
            except Exception as e:
                print(f"Error loading {name} futures symbols: {str(e)}")

        return all_symbols

    def fetch_ohlcv(self, exchange, symbol, timeframe, limit=100):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {str(e)}")
            return None
