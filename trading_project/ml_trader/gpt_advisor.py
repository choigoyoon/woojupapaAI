# -*- coding: utf-8 -*-
from typing import Dict, List
from openai import OpenAI  # 새로운 import 방식
import pandas as pd
from .config import Config

class GPTAdvisor:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)  # 새로운 클라이언트 초기화
        self.model = Config.GPT_MODEL
        
    def analyze_market(self, data: pd.DataFrame, ml_signals: List[Dict]) -> Dict:
        """GPT를 사용한 시장 분석"""
        try:
            # 최근 데이터 요약
            recent_data = data.tail(5)
            price_change = (recent_data["close"].iloc[-1] / recent_data["close"].iloc[0] - 1) * 100
            
            # ML 신호 요약
            ml_summary = "ML 모델 신호:\n"
            for signal in ml_signals[-5:]:
                ml_summary += f"- {signal['action'].upper()}: 신뢰도 {signal.get('confidence', 0):.2%}\n"
                
            # GPT 프롬프트 생성
            prompt = f"""
            다음은 최근 시장 데이터입니다:
            - 가격 변화: {price_change:.2f}%
            - 거래량 변화: {recent_data['volume'].pct_change().mean():.2f}%
            
            {ml_summary}
            
            위 데이터를 바탕으로:
            1. 현재 시장 상황 분석
            2. 주요 기술적 지표 해석
            3. ML 모델 신호와의 연관성 분석
            4. 매매 전략 제안
            
            위 항목들에 대해 분석해주세요.
            """
            
            # 새로운 API 호출 방식
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 전문 트레이딩 분석가입니다."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # 새로운 응답 처리 방식
            analysis = response.choices[0].message.content
            
            return {
                "analysis": analysis,
                "ml_confidence": ml_signals[-1].get('confidence', 0),
                "price_change": price_change,
                "recommendation": self._extract_recommendation(analysis)
            }
            
        except Exception as e:
            print(f"GPT 분석 중 오류: {str(e)}")
            return {
                "analysis": "분석 중 오류 발생",
                "error": str(e),
                "recommendation": "hold"
            }
            
    def _extract_recommendation(self, analysis: str) -> str:
        """GPT 분석에서 추천 사항 추출"""
        if "매수" in analysis or "buy" in analysis.lower():
            return "buy"
        elif "매도" in analysis or "sell" in analysis.lower():
            return "sell"
        return "hold"
