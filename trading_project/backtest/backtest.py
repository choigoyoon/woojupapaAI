# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

class Backtester:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.performance_metrics = {}
        
    def calculate_metrics(self):
        """백테스트 결과 계산"""
        if not self.trade_history:
            return {}
            
        profits = [trade["profit"] for trade in self.trade_history]
        win_trades = [p for p in profits if p > 0]
        loss_trades = [p for p in profits if p < 0]
        
        metrics = {
            "total_trades": len(self.trade_history),
            "winning_trades": len(win_trades),
            "losing_trades": len(loss_trades),
            "win_rate": len(win_trades) / len(self.trade_history) if self.trade_history else 0,
            "total_profit": sum(profits),
            "profit_factor": abs(sum(win_trades) / sum(loss_trades)) if sum(loss_trades) != 0 else float("inf"),
            "average_profit": np.mean(profits) if profits else 0,
            "max_drawdown": self.calculate_max_drawdown(),
            "final_balance": self.current_balance
        }
        
        self.performance_metrics = metrics
        return metrics
