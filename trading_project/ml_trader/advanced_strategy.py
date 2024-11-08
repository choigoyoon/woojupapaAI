# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.signal import find_peaks
import ta

@dataclass
class MarketPattern:
    pattern_type: str
    confidence: float
    price_target: float
    stop_loss: float

class AdvancedStrategy:
    def __init__(self):
        self.elliott_wave_count = 0
        self.last_pattern = None
        
    def analyze_elliott_wave(self, prices: pd.Series) -> dict:
        """엘리엇 파동 분석"""
        peaks, _ = find_peaks(prices.values, distance=10)
        troughs, _ = find_peaks(-prices.values, distance=10)
        
        wave_characteristics = {
            "wave_count": len(peaks) + len(troughs),
            "current_wave": self._identify_current_wave(prices),
            "wave_strength": self._calculate_wave_strength(prices, peaks, troughs),
            "correction_depth": self._calculate_correction_depth(prices, peaks, troughs)
        }
        
        return wave_characteristics
        
    def find_divergence(self, prices: pd.Series, indicator: pd.Series) -> list:
        """다이버전스 패턴 감지"""
        divergences = []
        
        price_peaks, _ = find_peaks(prices.values, distance=5)
        price_troughs, _ = find_peaks(-prices.values, distance=5)
        ind_peaks, _ = find_peaks(indicator.values, distance=5)
        ind_troughs, _ = find_peaks(-indicator.values, distance=5)
        
        # 일반 다이버전스
        for i in range(len(price_peaks)-1):
            if len(ind_peaks) > i+1:
                if (prices.iloc[price_peaks[i]] < prices.iloc[price_peaks[i+1]] and 
                    indicator.iloc[ind_peaks[i]] > indicator.iloc[ind_peaks[i+1]]):
                    divergences.append({
                        "type": "bearish",
                        "confidence": self._calculate_divergence_strength(
                            prices.iloc[price_peaks[i:i+2]], 
                            indicator.iloc[ind_peaks[i:i+2]]
                        )
                    })
                    
        # 숨겨진 다이버전스
        for i in range(len(price_troughs)-1):
            if len(ind_troughs) > i+1:
                if (prices.iloc[price_troughs[i]] > prices.iloc[price_troughs[i+1]] and 
                    indicator.iloc[ind_troughs[i]] < indicator.iloc[ind_troughs[i+1]]):
                    divergences.append({
                        "type": "bullish",
                        "confidence": self._calculate_divergence_strength(
                            prices.iloc[price_troughs[i:i+2]], 
                            indicator.iloc[ind_troughs[i:i+2]]
                        )
                    })
                    
        return divergences
        
    def detect_patterns(self, df: pd.DataFrame) -> list:
        """특수 패턴 감지"""
        patterns = []
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # 이중바닥 패턴
        if self._is_double_bottom(low):
            patterns.append(MarketPattern(
                pattern_type="double_bottom",
                confidence=self._calculate_pattern_strength(df, "double_bottom"),
                price_target=close[-1] * 1.04,
                stop_loss=low[-1] * 0.98
            ))
            
        # 이중천장 패턴
        if self._is_double_top(high):
            patterns.append(MarketPattern(
                pattern_type="double_top",
                confidence=self._calculate_pattern_strength(df, "double_top"),
                price_target=close[-1] * 0.96,
                stop_loss=high[-1] * 1.02
            ))
            
        # 역헤드앤숄더 패턴
        if self._is_inverse_head_shoulders(low):
            patterns.append(MarketPattern(
                pattern_type="inverse_head_shoulders",
                confidence=self._calculate_pattern_strength(df, "ihs"),
                price_target=close[-1] * 1.06,
                stop_loss=low[-1] * 0.97
            ))
            
        return patterns
        
    def analyze_market_psychology(self, df: pd.DataFrame) -> dict:
        """시장 심리 분석"""
        rsi = ta.momentum.RSIIndicator(close=df['Close']).rsi().iloc[-1]
        macd = ta.trend.MACD(close=df['Close'])
        bb = ta.volatility.BollingerBands(close=df['Close'])
        
        return {
            "market_sentiment": self._calculate_sentiment(rsi),
            "momentum": self._calculate_momentum(
                macd.macd().iloc[-1], 
                macd.macd_signal().iloc[-1]
            ),
            "volatility": self._calculate_volatility_state(df['Close'], bb),
            "volume_pressure": self._calculate_volume_pressure(df)
        }
        
    def _identify_current_wave(self, prices: pd.Series) -> int:
        """현재 엘리엇 파동 위치 식별"""
        ma5 = prices.rolling(5).mean()
        ma20 = prices.rolling(20).mean()
        ma60 = prices.rolling(60).mean()
        
        if len(prices) < 60:
            return 0
            
        current_price = prices.iloc[-1]
        if current_price > ma5.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
            return 3  # 강세 파동
        elif current_price < ma5.iloc[-1] < ma20.iloc[-1]:
            return 4  # 조정 파동
        elif ma5.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
            return 5  # 최종 파동
            
        return 0
        
    def _calculate_wave_strength(self, prices: pd.Series, peaks: np.ndarray, troughs: np.ndarray) -> float:
        """파동 강도 계산"""
        if len(peaks) == 0 or len(troughs) == 0:
            return 0.0
        
        wave_heights = []
        for i in range(min(len(peaks), len(troughs))):
            height = abs(prices.iloc[peaks[i]] - prices.iloc[troughs[i]])
            wave_heights.append(height)
            
        return np.mean(wave_heights) / prices.iloc[-1] * 100
        
    def _calculate_correction_depth(self, prices: pd.Series, peaks: np.ndarray, troughs: np.ndarray) -> float:
        """조정 깊이 계산"""
        if len(peaks) < 2 or len(troughs) < 1:
            return 0.0
            
        last_peak = prices.iloc[peaks[-1]]
        last_trough = prices.iloc[troughs[-1]]
        correction = (last_peak - last_trough) / last_peak * 100
            
        return correction
        
    def _calculate_divergence_strength(self, prices: pd.Series, indicator: pd.Series) -> float:
        """다이버전스 강도 계산"""
        price_change = abs(prices.iloc[1] - prices.iloc[0]) / prices.iloc[0]
        ind_change = abs(indicator.iloc[1] - indicator.iloc[0]) / indicator.iloc[0]
        
        return min(1.0, (price_change + ind_change) / 2)
        
    def _is_double_bottom(self, low: np.ndarray) -> bool:
        """이중바닥 패턴 확인"""
        if len(low) < 20:
            return False
            
        bottoms, _ = find_peaks(-low[-20:], distance=5)
        if len(bottoms) >= 2:
            bottom1, bottom2 = low[bottoms[-2]], low[bottoms[-1]]
            return abs(bottom1 - bottom2) / bottom1 < 0.02
            
        return False
        
    def _is_double_top(self, high: np.ndarray) -> bool:
        """이중천장 패턴 확인"""
        if len(high) < 20:
            return False
            
        tops, _ = find_peaks(high[-20:], distance=5)
        if len(tops) >= 2:
            top1, top2 = high[tops[-2]], high[tops[-1]]
            return abs(top1 - top2) / top1 < 0.02
            
        return False
        
    def _is_inverse_head_shoulders(self, low: np.ndarray) -> bool:
        """역헤드앤숄더 패턴 확인"""
        if len(low) < 30:
            return False
            
        bottoms, _ = find_peaks(-low[-30:], distance=5)
        if len(bottoms) >= 3:
            shoulder1 = low[bottoms[-3]]
            head = low[bottoms[-2]]
            shoulder2 = low[bottoms[-1]]
            return head < shoulder1 and head < shoulder2 and abs(shoulder1 - shoulder2) / shoulder1 < 0.03
            
        return False
        
    def _calculate_pattern_strength(self, df: pd.DataFrame, pattern_type: str) -> float:
        """패턴 신뢰도 계산"""
        volume = df['Volume'].values
        close = df['Close'].values
        
        volume_increase = volume[-1] / volume[-5] - 1
        price_volatility = np.std(close[-5:]) / np.mean(close[-5:])
        
        weights = {
            "double_bottom": 1.2,
            "double_top": 1.1,
            "ihs": 1.3
        }
        
        base_confidence = min(1.0, (volume_increase + 1) * (1 - price_volatility))
        return base_confidence * weights.get(pattern_type, 1.0)
        
    def _calculate_sentiment(self, rsi: float) -> str:
        """RSI 기반 시장 심리 상태"""
        if rsi > 70:
            return "극도의 과매수"
        elif rsi > 60:
            return "과매수"
        elif rsi < 30:
            return "극도의 과매도"
        elif rsi < 40:
            return "과매도"
        return "중립"
        
    def _calculate_momentum(self, macd: float, signal: float) -> str:
        """MACD 기반 모멘텀 강도"""
        diff = macd - signal
        if diff > 0.5:
            return "강한 상승 모멘텀"
        elif diff > 0:
            return "약한 상승 모멘텀"
        elif diff < -0.5:
            return "강한 하락 모멘텀"
        return "약한 하락 모멘텀"
        
    def _calculate_volatility_state(self, prices: pd.Series, bb) -> str:
        """볼린저 밴드 기반 변동성 상태"""
        current_price = prices.iloc[-1]
        upper_band = bb.bollinger_hband().iloc[-1]
        lower_band = bb.bollinger_lband().iloc[-1]
        
        if current_price > upper_band:
            return "고변동성 상승"
        elif current_price < lower_band:
            return "고변동성 하락"
        return "정상 변동성"
        
    def _calculate_volume_pressure(self, df: pd.DataFrame) -> str:
        """거래량 압력 분석"""
        recent_volume = df['Volume'].tail(5).mean()
        prev_volume = df['Volume'].tail(10).head(5).mean()
        
        volume_change = (recent_volume / prev_volume - 1) * 100
        if volume_change > 20:
            return "강한 매수세"
        elif volume_change < -20:
            return "강한 매도세"
        return "중립적 거래량"
