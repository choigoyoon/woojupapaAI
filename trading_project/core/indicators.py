# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ta
from config.settings import INDICATOR_SETTINGS

class TechnicalIndicators:
    def __init__(self):
        self.settings = INDICATOR_SETTINGS

    def calculate_all_indicators(self, df):
        """모든 기술적 지표 계산"""
        try:
            indicators = {
                'MACD': self.calculate_macd(df),
                'RSI': self.calculate_rsi(df),
                'CCI': self.calculate_cci(df),
                'Stoch_RSI': self.calculate_stoch_rsi(df),
                'Williams_R': self.calculate_williams_r(df),
                'Squeeze_Momentum': self.calculate_squeeze_momentum(df)
            }
            
            return {
                'timestamp': df['timestamp'].iloc[-1],
                'open': df['open'].iloc[-1],
                'high': df['high'].iloc[-1],
                'low': df['low'].iloc[-1],
                'close': df['close'].iloc[-1],
                'volume': df['volume'].iloc[-1],
                'indicators': indicators
            }
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return None

    def calculate_macd(self, df):
        """MACD 계산"""
        macd = ta.trend.MACD(
            close=df['close'],
            window_slow=self.settings['MACD']['slow'],
            window_fast=self.settings['MACD']['fast'],
            window_sign=self.settings['MACD']['signal']
        )
        
        macd_val = macd.macd().iloc[-1]
        signal = macd.macd_signal().iloc[-1]
        
        # 크로스 확인
        if macd_val > signal and macd.macd().iloc[-2] <= macd.macd_signal().iloc[-2]:
            cross = 'Golden'
        elif macd_val < signal and macd.macd().iloc[-2] >= macd.macd_signal().iloc[-2]:
            cross = 'Dead'
        else:
            cross = 'None'
            
        return {
            'macd': macd_val,
            'signal': signal,
            'cross': cross
        }

    def calculate_rsi(self, df):
        """RSI 계산"""
        rsi = ta.momentum.RSIIndicator(
            close=df['close'],
            window=self.settings['RSI']['period']
        ).rsi().iloc[-1]
        
        return {
            'value': rsi,
            'signal': 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'
        }

    def calculate_cci(self, df):
        """CCI 계산"""
        cci = ta.trend.CCIIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.settings['CCI']['period']
        ).cci().iloc[-1]
        
        return {
            'value': cci,
            'signal': 'Oversold' if cci < -100 else 'Overbought' if cci > 100 else 'Neutral'
        }

    def calculate_stoch_rsi(self, df):
        """Stochastic RSI 계산"""
        stoch_rsi = ta.momentum.StochRSIIndicator(
            close=df['close'],
            window=self.settings['Stoch_RSI']['period'],
            smooth1=self.settings['Stoch_RSI']['smooth1'],
            smooth2=self.settings['Stoch_RSI']['smooth2']
        )
        
        k = stoch_rsi.stochrsi_k().iloc[-1]
        d = stoch_rsi.stochrsi_d().iloc[-1]
        
        return {
            'k': k,
            'd': d,
            'signal': 'Oversold' if k < 0.2 else 'Overbought' if k > 0.8 else 'Neutral'
        }

    def calculate_williams_r(self, df):
        """Williams %R 계산"""
        wr = ta.momentum.WilliamsRIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            lbp=self.settings['Williams_R']['period']
        ).williams_r().iloc[-1]
        
        return {
            'value': wr,
            'signal': 'Oversold' if wr < -80 else 'Overbought' if wr > -20 else 'Neutral'
        }

    def calculate_squeeze_momentum(self, df):
        """스퀴즈 모멘텀 계산"""
        length = self.settings['Squeeze_Momentum']['length']
        mult = self.settings['Squeeze_Momentum']['mult']
        length_kc = self.settings['Squeeze_Momentum']['length_kc']
        mult_kc = self.settings['Squeeze_Momentum']['mult_kc']
        
        # 볼린저 밴드
        basis = df['close'].rolling(window=length).mean()
        dev = mult * df['close'].rolling(window=length).std()
        
        # 켈트너 채널
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        ma = typical_price.rolling(window=length_kc).mean()
        range_ma = (df['high'] - df['low']).rolling(window=length_kc).mean()
        
        squeeze_on = (basis + dev).iloc[-1] <= (ma + mult_kc * range_ma).iloc[-1] and \
                    (basis - dev).iloc[-1] >= (ma - mult_kc * range_ma).iloc[-1]
        
        # 모멘텀
        highest = df['high'].rolling(window=length).max()
        lowest = df['low'].rolling(window=length).min()
        m = (highest + lowest) / 2
        momentum = df['close'] - m
        
        return {
            'squeeze_on': squeeze_on,
            'momentum_change': momentum.iloc[-1] - momentum.iloc[-2],
            'signal': 'Buy' if squeeze_on and momentum.iloc[-1] > momentum.iloc[-2] else
                     'Sell' if squeeze_on and momentum.iloc[-1] < momentum.iloc[-2] else 'Neutral'
        }
