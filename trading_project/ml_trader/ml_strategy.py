# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import ta
from typing import Dict
import joblib

class MLStrategy:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특징 데이터 준비"""
        features = pd.DataFrame(index=df.index)
        
        # 기술적 지표
        features["rsi"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()
        features["macd"] = ta.trend.MACD(close=df["close"]).macd()
        features["cci"] = ta.trend.CCIIndicator(high=df["high"], low=df["low"], close=df["close"]).cci()
        
        # 이동평균
        features["ma5"] = df["close"].rolling(5).mean()
        features["ma20"] = df["close"].rolling(20).mean()
        features["ma60"] = df["close"].rolling(60).mean()
        
        # 변동성
        features["volatility"] = df["close"].rolling(20).std()
        
        # 거래량 지표
        features["volume_ma5"] = df["volume"].rolling(5).mean()
        features["volume_ma20"] = df["volume"].rolling(20).mean()
        
        # 가격 모멘텀
        features["price_momentum"] = df["close"].pct_change(5)
        
        return features.dropna()
        
    def prepare_labels(self, df: pd.DataFrame, threshold: float = 0.02) -> np.ndarray:
        """레이블 데이터 준비"""
        future_returns = df["close"].pct_change(5).shift(-5)
        labels = np.zeros(len(df))
        labels[future_returns > threshold] = 1
        labels[future_returns < -threshold] = -1
        return labels[:-5]  # 마지막 5일 제외
        
    def train(self, df: pd.DataFrame):
        """모델 학습"""
        features = self.prepare_features(df)
        labels = self.prepare_labels(df)
        
        # 데이터 길이 맞추기
        min_len = min(len(features), len(labels))
        features = features.iloc[:min_len]
        labels = labels[:min_len]
        
        # 유효한 데이터만 선택
        valid_mask = ~np.isnan(labels)
        X = features[valid_mask]
        y = labels[valid_mask]
        
        if len(X) > 0:
            # 스케일링
            X_scaled = self.scaler.fit_transform(X)
            
            # 모델 학습
            self.model.fit(X_scaled, y)
            self.is_trained = True
        
    def predict(self, data: pd.DataFrame) -> Dict:
        """매매 신호 생성"""
        if not self.is_trained:
            return {"action": "hold"}
            
        features = self.prepare_features(data.tail(60))
        if len(features) < 1:
            return {"action": "hold"}
            
        X = features.iloc[-1:]
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        
        current_price = data["close"].iloc[-1]
        if prediction == 1 and max(proba) > 0.7:
            return {
                "action": "buy",
                "size": 1.0,
                "entry_price": current_price,
                "stop_loss": current_price * 0.98,
                "take_profit": current_price * 1.04,
                "confidence": float(max(proba))
            }
        elif prediction == -1 and max(proba) > 0.7:
            return {
                "action": "sell",
                "confidence": float(max(proba))
            }
            
        return {"action": "hold"}
