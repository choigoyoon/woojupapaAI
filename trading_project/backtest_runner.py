# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta
from typing import Dict, List
import json

class BacktestRunner:
    def __init__(self, initial_balance=10000, leverage=1):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.position = None
        self.trades = []
        self.trading_enabled = True
        self.position_size = 0.2
        self.stop_loss_pct = 0.015

    def fetch_btc_data(self):
        btc = yf.Ticker("BTC-USD")
        data_daily = btc.history(period="max", interval="1d")
        return self._add_wave_analysis(data_daily)

    def _add_wave_analysis(self, data):
        # 기본 지표
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
        data['EMA_200'] = ta.trend.EMAIndicator(data['Close'], window=200).ema_indicator()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['MACD_signal'] = ta.trend.MACD(data['Close']).macd_signal()
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        # 파동 분석을 위한 추가 지표
        data['High_Low_Range'] = data['High'] - data['Low']
        data['Price_Change'] = data['Close'].pct_change()
        data['Volatility'] = data['Price_Change'].rolling(window=20).std()
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
        
        # 추가 모멘텀 지표
        data['Stoch_K'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
        data['Stoch_D'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch_signal()
        
        # 추세 강도 분석
        data['DMI_plus'] = ta.trend.DMIIndicator(data['High'], data['Low'], data['Close']).adx_pos()
        data['DMI_minus'] = ta.trend.DMIIndicator(data['High'], data['Low'], data['Close']).adx_neg()
        
        return data

    def _identify_wave_position(self, data, lookback=200):
        try:
            recent_data = data.tail(lookback)
            
            # 주요 고점/저점 식별
            highs = recent_data['High'].rolling(window=20, center=True).max()
            lows = recent_data['Low'].rolling(window=20, center=True).min()
            
            # 현재 가격 위치
            current_price = recent_data['Close'].iloc[-1]
            last_swing_high = highs.iloc[-20:].max()
            last_swing_low = lows.iloc[-20:].min()
            
            # 파동 위치 추정
            price_position = (current_price - last_swing_low) / (last_swing_high - last_swing_low)
            
            # 추세 강도
            trend_strength = recent_data['ADX'].iloc[-1]
            
            # 파동 단계 추정 (1~5)
            if price_position < 0.2:
                return 1  # 1파동 시작점
            elif price_position < 0.4:
                return 2  # 2파동 조정
            elif price_position < 0.6:
                return 3  # 3파동 상승
            elif price_position < 0.8:
                return 4  # 4파동 조정
            else:
                return 5  # 5파동 정점
                
        except Exception as e:
            print(f"Wave position identification error: {e}")
            return 0

    def _check_entry_conditions(self, data):
        try:
            wave_position = self._identify_wave_position(data)
            if wave_position not in [1, 3]:  # 1파동과 3파동에서만 진입
                return False
                
            current = data.iloc[-1]
            prev = data.iloc[-2]
            
            # 기본 조건
            if not (30 <= current['RSI'] <= 70):
                return False
                
            # 추세 조건
            trend_up = (current['EMA_20'] > current['EMA_50'] > current['EMA_200'])
            
            # 모멘텀 조건
            momentum_good = (current['Stoch_K'] > current['Stoch_D'] and 
                           current['MACD'] > current['MACD_signal'])
            
            # DMI 조건
            dmi_good = current['DMI_plus'] > current['DMI_minus']
            
            # 거래량 조건 완화
            volume_good = current['Volume'] > current['Volume_MA']
            
            # ADX 조건 완화
            trend_strong = current['ADX'] > 20  # 25에서 20으로 완화
            
            return trend_up and momentum_good and dmi_good and volume_good and trend_strong
            
        except Exception as e:
            print(f"Error in entry conditions: {e}")
            return False

    def _calculate_exit_points(self, data):
        current_price = data['Close'].iloc[-1]
        atr = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range().iloc[-1]
        
        stop_loss = current_price - (atr * 2)
        take_profit = current_price + (atr * 3)
        
        return stop_loss, take_profit

    def run_backtest(self, data):
        for i in range(200, len(data)):  # 200일 이후부터 시작 (지표 계산을 위해)
            current_data = data.iloc[:i+1]
            current_price = current_data['Close'].iloc[-1]
            
            if self.position is None and self._check_entry_conditions(current_data):
                stop_loss, take_profit = self._calculate_exit_points(current_data)
                self.position = {
                    'type': 'long',
                    'entry_price': current_price,
                    'entry_time': current_data.index[-1],
                    'size': self.balance * self.position_size * self.leverage,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                
            elif self.position is not None:
                if current_price <= self.position['stop_loss'] or current_price >= self.position['take_profit']:
                    profit = (current_price - self.position['entry_price']) / self.position['entry_price']
                    self.balance += self.position['size'] * profit
                    self.trades.append({
                        'entry_time': self.position['entry_time'],
                        'exit_time': current_data.index[-1],
                        'entry_price': self.position['entry_price'],
                        'exit_price': current_price,
                        'profit': profit * 100,
                        'balance': self.balance,
                        'wave_position': self._identify_wave_position(current_data.iloc[:-1])
                    })
                    self.position = None
                    
        return self._calculate_statistics()

    def _calculate_statistics(self):
        if not self.trades:
            return {"error": "No trades"}
            
        profits = [trade['profit'] for trade in self.trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        
        wave_positions = [trade.get('wave_position', 0) for trade in self.trades]
        
        stats = {
            "Total Trades": len(self.trades),
            "Win Rate": len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
            "Average Profit": np.mean(profits) if profits else 0,
            "Max Profit": max(profits) if profits else 0,
            "Max Loss": min(profits) if profits else 0,
            "Final Balance": self.balance,
            "Total Return": (self.balance / self.initial_balance - 1) * 100,
            "Profit Factor": abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf'),
            "Average Wave Position": np.mean(wave_positions) if wave_positions else 0
        }
        
        return stats

if __name__ == "__main__":
    backtest = BacktestRunner(initial_balance=10000, leverage=1)
    data = backtest.fetch_btc_data()
    results = backtest.run_backtest(data)
    print("\nBacktest Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
