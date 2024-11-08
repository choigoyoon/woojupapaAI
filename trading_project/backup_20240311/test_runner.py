# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from ml_trader.gpt_advisor import GPTAdvisor
from ml_trader.ml_strategy import MLStrategy

def run_test():
    print("=== ML Trader with GPT-4 테스트 시작 ===\n")
    
    # 테스트용 데이터 생성
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "close": np.random.normal(100, 5, 100).cumsum(),
        "high": np.random.normal(102, 5, 100).cumsum(),
        "low": np.random.normal(98, 5, 100).cumsum(),
        "volume": np.abs(np.random.normal(1000, 200, 100))  # abs() 대신 np.abs() 사용
    }, index=dates)
    
    print("1. ML 모델 초기화 및 학습")
    ml_strategy = MLStrategy()
    ml_strategy.train(data)
    
    print("2. ML 모델 예측 실행")
    ml_signals = []
    for i in range(-5, 0):
        signal = ml_strategy.predict(data[:i])
        ml_signals.append(signal)
    
    print("3. GPT-4 분석 시작")
    advisor = GPTAdvisor()
    analysis = advisor.analyze_market(data, ml_signals)
    
    print("\n=== GPT-4 분석 결과 ===")
    print(analysis["analysis"])
    
    print("\n=== 최종 추천 ===")
    print(f"행동: {analysis['recommendation']}")
    if "ml_confidence" in analysis:
        print(f"ML 신뢰도: {analysis['ml_confidence']:.2%}")
    if "price_change" in analysis:
        print(f"가격 변화: {analysis['price_change']:.2f}%")

if __name__ == "__main__":
    run_test()
