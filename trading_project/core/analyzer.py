# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from core.indicators import TechnicalIndicators

class MarketAnalyzer:
    def __init__(self):
        self.indicators = TechnicalIndicators()

    def analyze_market(self, ohlcv_data):
        """시장 데이터 분석"""
        try:
            # OHLCV 데이터를 DataFrame으로 변환
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 기술적 지표 계산
            analysis = self.indicators.calculate_all_indicators(df)
            
            return analysis
        except Exception as e:
            print(f"Error in market analysis: {str(e)}")
            return None

    def generate_signal(self, analysis):
        """매매 신호 생성"""
        if analysis is None:
            return None

        signals = {
            'Buy': 0,
            'Sell': 0,
            'Neutral': 0
        }

        # 각 지표의 신호를 집계
        for indicator, data in analysis['indicators'].items():
            if 'signal' in data:
                if data['signal'] in ['Buy', 'Oversold']:
                    signals['Buy'] += 1
                elif data['signal'] in ['Sell', 'Overbought']:
                    signals['Sell'] += 1
                else:
                    signals['Neutral'] += 1

        # MACD 크로스 추가 가중치
        if analysis['indicators']['MACD']['cross'] == 'Golden':
            signals['Buy'] += 2
        elif analysis['indicators']['MACD']['cross'] == 'Dead':
            signals['Sell'] += 2

        # 스퀴즈 모멘텀 추가 가중치
        if analysis['indicators']['Squeeze_Momentum']['squeeze_on']:
            if analysis['indicators']['Squeeze_Momentum']['signal'] == 'Buy':
                signals['Buy'] += 2
            elif analysis['indicators']['Squeeze_Momentum']['signal'] == 'Sell':
                signals['Sell'] += 2

        # 최종 신호 결정
        max_signal = max(signals.items(), key=lambda x: x[1])
        strength = max_signal[1] / sum(signals.values()) if sum(signals.values()) > 0 else 0

        return {
            'signal': max_signal[0],
            'strength': strength,
            'signals': signals
        }
