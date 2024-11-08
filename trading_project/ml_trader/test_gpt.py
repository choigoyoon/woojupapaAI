# -*- coding: utf-8 -*-
from ml_trader.gpt_advisor import GPTAdvisor
import pandas as pd

def test_gpt():
    advisor = GPTAdvisor()
    
    # 테스트용 데이터
    test_data = pd.DataFrame({
        "close": [100, 102, 101, 103, 105],
        "volume": [1000, 1200, 900, 1100, 1300]
    })
    
    # 테스트용 ML 신호
    test_signals = [
        {"action": "buy", "confidence": 0.85}
    ]
    
    # GPT 분석 실행
    result = advisor.analyze_market(test_data, test_signals)
    
    print("=== GPT-4 분석 결과 ===")
    print(result["analysis"])
    print("\n=== 추천 행동 ===")
    print(result["recommendation"])

if __name__ == "__main__":
    test_gpt()
