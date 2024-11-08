# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import ta
from typing import Dict, List
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import requests
from textblob import TextBlob
from bs4 import BeautifulSoup
import tweepy
import warnings
warnings.filterwarnings('ignore')

class DeepLearningModel:
    def __init__(self):
        self.price_model = self._build_lstm_model()
        self.scaler = MinMaxScaler()
        
    def _build_lstm_model(self):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def prepare_data(self, data, lookback=60):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)
        
    def train(self, data):
        X, y = self.prepare_data(data)
        self.price_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
    def predict(self, data):
        X, _ = self.prepare_data(data[-60:])
        scaled_prediction = self.price_model.predict(X.reshape(1, 60, 1))
        return self.scaler.inverse_transform(scaled_prediction)[0, 0]

class SentimentAnalyzer:
    def __init__(self):
        self.twitter_api = self._setup_twitter_api()
        
    def _setup_twitter_api(self):
        # Twitter API 설정 (실제 키로 교체 필요)
        auth = tweepy.OAuthHandler("YOUR_API_KEY", "YOUR_API_SECRET")
        auth.set_access_token("YOUR_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN_SECRET")
        return tweepy.API(auth)
        
    def analyze_news_sentiment(self, symbol="BTC"):
        # 크립토 뉴스 수집 및 분석
        url = f"https://cryptonews.com/news/bitcoin-news/"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_texts = soup.find_all('div', class_='article__title')
        
        sentiments = []
        for news in news_texts[:10]:  # 최근 10개 뉴스
            analysis = TextBlob(news.text)
            sentiments.append(analysis.sentiment.polarity)
            
        return np.mean(sentiments)
        
    def analyze_social_sentiment(self, symbol="BTC"):
        # 트위터 감성 분석
        tweets = self.twitter_api.search_tweets(q=symbol, lang="en", count=100)
        sentiments = []
        
        for tweet in tweets:
            analysis = TextBlob(tweet.text)
            sentiments.append(analysis.sentiment.polarity)
            
        return np.mean(sentiments)

class BacktestRunner:
    def __init__(self, initial_balance=10000, leverage=1):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.trades = []
        self.leverage = leverage
        self.position_size = min(0.1, 1/leverage)
        self.stop_loss_pct = min(0.005, 0.2/leverage)
        
        # 딥러닝 모델 초기화
        self.deep_learning = DeepLearningModel()
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def fetch_btc_data(self, start_date="2022-01-01"):
        """BTC 데이터 가져오기"""
        btc = yf.Ticker("BTC-USD")
        data = btc.history(start=start_date, interval="1d")
        return data
        
    def _identify_resistance_levels(self, data: pd.DataFrame, lookback=20) -> list:
        """저항선 식별"""
        highs = data['High'].rolling(window=lookback).max()
        resistance_levels = []
        
        for i in range(lookback, len(data)):
            if abs(highs.iloc[i] - highs.iloc[i-1]) / highs.iloc[i] < 0.005:
                resistance_levels.append(highs.iloc[i])
        
        return resistance_levels
        
    def _check_breakout(self, price: float, resistance_level: float, volume: float, avg_volume: float) -> bool:
        """저항 돌파 확인"""
        return price > resistance_level * 1.002 and volume > avg_volume * 1.5
        
    def _check_support_at_resistance(self, data: pd.DataFrame, resistance_level: float) -> bool:
        """이전 저항이 지지로 전환되었는지 확인"""
        recent_lows = data['Low'].tail(5)
        return all(low >= resistance_level * 0.995 for low in recent_lows)
        
    def _get_ml_signals(self, data: pd.DataFrame) -> Dict:
        """ML 신호 생성"""
        # LSTM 예측
        price_prediction = self.deep_learning.predict(data['Close'].values)
        
        # 감성 분석
        news_sentiment = self.sentiment_analyzer.analyze_news_sentiment()
        social_sentiment = self.sentiment_analyzer.analyze_social_sentiment()
        
        # 신호 종합
        price_signal = 1 if price_prediction > data['Close'].iloc[-1] else -1
        sentiment_signal = (news_sentiment + social_sentiment) / 2
        
        return {
            'price_prediction': price_prediction,
            'sentiment_score': sentiment_signal,
            'combined_score': (price_signal + sentiment_signal) / 2
        }
        
    def run_backtest(self, data: pd.DataFrame):
        """백테스트 실행"""
        print(f"백테스트 시작... (레버리지: {self.leverage}X)")
        
        # LSTM 모델 학습
        self.deep_learning.train(data['Close'].values)
        
        # 기술적 지표 계산
        data['ATR'] = ta.volatility.AverageTrueRange(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=14
        ).average_true_range()
        
        data['RSI'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
        data['BB_upper'] = ta.volatility.BollingerBands(close=data['Close']).bollinger_hband()
        data['BB_lower'] = ta.volatility.BollingerBands(close=data['Close']).bollinger_lband()
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        for i in range(60, len(data)):
            current_data = data.iloc[:i+1]
            current_price = current_data['Close'].iloc[-1]
            current_volume = current_data['Volume'].iloc[-1]
            
            # ML 신호 획득
            ml_signals = self._get_ml_signals(current_data)
            
            resistance_levels = self._identify_resistance_levels(current_data)
            
            if not self.position:  # 포지션 없을 때
                for level in resistance_levels:
                    if (self._check_breakout(current_price, level, current_volume, current_data['Volume_MA'].iloc[-1]) and
                        self._check_support_at_resistance(current_data, level) and
                        ml_signals['combined_score'] > 0.5):  # ML 신호 추가
                        
                        position_size = self.balance * self.position_size
                        size = position_size / current_price * self.leverage
                        
                        self.position = {
                            'type': 'long',
                            'entry_price': current_price,
                            'size': size,
                            'stop_loss': current_price * (1 - self.stop_loss_pct),
                            'take_profit': current_price * (1 + self.stop_loss_pct * 5),
                            'resistance_level': level
                        }
                        self.balance -= position_size
            
            else:  # 포지션 있을 때
                # ML 신호로 인한 조기 청산
                if ml_signals['combined_score'] < -0.5:
                    self._close_position(current_price, current_data.index[-1], 'ml_signal')
                    
                elif current_price <= self.position['stop_loss']:
                    self._close_position(current_price, current_data.index[-1], 'stop_loss')
                
                elif current_price >= self.position['take_profit']:
                    self._close_position(current_price, current_data.index[-1], 'take_profit')
                
                elif current_price < self.position['resistance_level'] * 0.995:
                    self._close_position(current_price, current_data.index[-1], 'trend_reversal')
        
        return self._calculate_statistics()
        
    def _close_position(self, current_price: float, timestamp, reason: str):
        """포지션 종료"""
        profit = (current_price - self.position['entry_price']) * self.position['size']
        self.balance += (self.position['size'] * current_price) / self.leverage
        
        self.trades.append({
            'entry_price': self.position['entry_price'],
            'exit_price': current_price,
            'profit': profit,
            'profit_pct': (current_price / self.position['entry_price'] - 1) * 100 * self.leverage,
            'reason': reason
        })
        
        self.position = None
        
    def _calculate_statistics(self) -> Dict:
        """백테스트 결과 통계"""
        if not self.trades:
            return {"error": "거래 내역 없음"}
            
        profits = [trade['profit'] for trade in self.trades]
        profit_pcts = [trade['profit_pct'] for trade in self.trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        
        # 청산 이유별 통계
        reasons = [trade['reason'] for trade in self.trades]
        reason_stats = {reason: reasons.count(reason) for reason in set(reasons)}
        
        # MDD 계산
        cumulative_returns = np.cumsum(profit_pcts)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max)
        mdd = np.min(drawdowns)
        
        stats = {
            "레버리지": f"{self.leverage}X",
            "총 거래 횟수": len(self.trades),
            "승률": len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
            "평균 수익률": np.mean(profit_pcts) if profit_pcts else 0,
            "최대 수익률": max(profit_pcts) if profit_pcts else 0,
            "최대 손실률": min(profit_pcts) if profit_pcts else 0,
            "손익비": abs(np.mean(winning_trades) / np.mean(losing_trades)) if losing_trades else float('inf'),
            "MDD": mdd,
            "청산 이유별 통계": reason_stats,
            "최종 잔고": self.balance,
            "총 수익률": (self.balance / self.initial_balance - 1) * 100
        }
        
        return stats

if __name__ == "__main__":
    leverages = [1, 10, 20, 30, 40, 50]
    results = []
    
    print("\n=== 레버리지별 백테스트 결과 ===")
    print("-" * 50)
    
    for leverage in leverages:
        backtest = BacktestRunner(initial_balance=10000, leverage=leverage)
        data = backtest.fetch_btc_data()
        result = backtest.run_backtest(data)
        results.append(result)
        
        print(f"\n[레버리지 {leverage}X 결과]")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        print("-" * 50)
        
        # 거래 내역 저장
        with open(f'trade_history_{leverage}x.json', 'w', encoding='utf-8') as f:
            json.dump(backtest.trades, f, default=str, indent=2)
