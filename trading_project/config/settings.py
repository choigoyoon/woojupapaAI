# -*- coding: utf-8 -*-

# 타임프레임 설정
TIMEFRAMES = {
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '4h': '4h',
    '12h': '12h',
    '1d': '1d',
    '3d': '3d',
    '1w': '1w'
}

# 지표 설정
INDICATOR_SETTINGS = {
    'MACD': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'RSI': {
        'period': 6
    },
    'CCI': {
        'period': 20
    },
    'Stoch_RSI': {
        'period': 6,
        'smooth1': 3,
        'smooth2': 3
    },
    'Williams_R': {
        'period': 6
    },
    'Squeeze_Momentum': {
        'length': 20,
        'mult': 2.0,
        'length_kc': 20,
        'mult_kc': 1.5
    }
}

# 업데이트 주기 (초)
UPDATE_INTERVAL = 900  # 15분

# 거래소 설정
EXCHANGES = {
    'spot': ['binance', 'upbit', 'bithumb', 'okx', 'bybit'],
    'futures': ['binance', 'okx', 'bybit']
}
