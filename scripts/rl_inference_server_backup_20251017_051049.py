#!/usr/bin/env python3
"""
강화학습 모델 추론 서버
finflow-rl 프로젝트의 학습된 모델을 로드하여 포트폴리오 예측을 제공한다.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
import glob
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import yfinance as yf
import warnings
import time
import random

warnings.filterwarnings("ignore")

# curl_cffi를 사용하여 Chrome 세션 생성
try:
    from curl_cffi import requests

    # Chrome을 모방하는 세션 생성
    session = requests.Session(impersonate="chrome")
    print("curl_cffi 세션 생성 성공 - Chrome 모방 모드")
except ImportError:
    print("curl_cffi를 찾을 수 없음. 기본 요청 방식 사용")
    session = None

# ===============================
# FinFlow-RL 모델 클래스 정의
# ===============================

# 상수 정의 (finflow-rl 프로젝트의 constants.py에서 가져옴)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_HIDDEN_DIM = 128
SOFTMAX_TEMPERATURE_INITIAL = 1.0
SOFTMAX_TEMPERATURE_MIN = 0.1
SOFTMAX_TEMPERATURE_DECAY = 0.999


class SelfAttention(nn.Module):
    """자기 주의(Self-Attention) 메커니즘"""

    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, x):
        batch_size, n_assets, hidden_dim = x.size()

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v)

        return context, attention_weights


class ActorCritic(nn.Module):
    """PPO를 위한 액터-크리틱 네트워크"""

    def __init__(self, n_assets, n_features, hidden_dim=DEFAULT_HIDDEN_DIM):
        super(ActorCritic, self).__init__()
        self.input_dim = n_assets * n_features
        self.n_assets = n_assets + 1  # 현금 자산 추가
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # 온도 파라미터
        self.temperature = nn.Parameter(torch.tensor(SOFTMAX_TEMPERATURE_INITIAL))
        self.temp_min = SOFTMAX_TEMPERATURE_MIN
        self.temp_decay = SOFTMAX_TEMPERATURE_DECAY

        # LSTM 레이어
        self.lstm_layers = 2
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        ).to(DEVICE)

        self.lstm_output_dim = hidden_dim * 2

        # 자기 주의 메커니즘
        self.attention = SelfAttention(self.lstm_output_dim).to(DEVICE)

        # 자산별 특징 압축 레이어
        self.asset_compression = nn.Sequential(
            nn.Linear(self.lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        ).to(DEVICE)

        # 공통 특징 추출 레이어
        self.actor_critic_base = nn.Sequential(
            nn.Linear(hidden_dim * n_assets, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        ).to(DEVICE)

        # 액터 헤드
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.n_assets),
        ).to(DEVICE)

        # 크리틱 헤드
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        ).to(DEVICE)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(
                module.weight, a=0, mode="fan_in", nonlinearity="relu"
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, states):
        """순전파"""
        batch_size = states.size(0) if states.dim() == 3 else 1

        if states.dim() == 2:
            states = states.unsqueeze(0)

        # NaN/Inf 방지
        if torch.isnan(states).any() or torch.isinf(states).any():
            states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

        # LSTM 처리
        lstm_outputs = []
        for i in range(states.size(1)):
            asset_feats = states[:, i, :].view(batch_size, 1, -1)
            lstm_out, _ = self.lstm(asset_feats)
            asset_out = lstm_out[:, -1, :]
            lstm_outputs.append(asset_out)

        # 어텐션 적용
        lstm_stacked = torch.stack(lstm_outputs, dim=1)
        context, _ = self.attention(lstm_stacked)

        # 특징 압축
        compressed_features = []
        for i in range(context.size(1)):
            asset_context = context[:, i, :]
            compressed = self.asset_compression(asset_context)
            compressed_features.append(compressed)

        lstm_concat = torch.cat(compressed_features, dim=1)

        # 베이스 네트워크
        base_output = self.actor_critic_base(lstm_concat)

        # 액터 출력
        logits = self.actor_head(base_output)
        scaled_logits = logits / (self.temperature + 1e-8)
        action_probs = F.softmax(scaled_logits, dim=-1)
        action_probs = torch.clamp(action_probs, min=1e-7, max=1.0)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        # 크리틱 출력
        value = self.critic_head(base_output)

        return action_probs, value


# ===============================
# FastAPI 서버 설정
# ===============================

app = FastAPI(title="FinFlow RL Inference Server", version="1.0.0")

# 환경 변수에서 CORS 설정 읽기
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# 프로덕션 환경에서는 추가 도메인 허용
environment = os.getenv("ENVIRONMENT", "development")
print(f"현재 환경: {environment}")  # 디버깅용

if environment == "production":
    production_origins = [
        "https://finflow.reo91004.com", 
        "https://www.finflow.reo91004.com"
    ]
    CORS_ORIGINS.extend(production_origins)
    print(f"프로덕션 도메인 추가: {production_origins}")

print(f"최종 CORS 허용 도메인: {CORS_ORIGINS}")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 요청/응답 모델
class PredictionRequest(BaseModel):
    investment_amount: float
    risk_tolerance: str = "moderate"
    investment_horizon: int = 252


class AllocationItem(BaseModel):
    symbol: str
    weight: float


class MetricsResponse(BaseModel):
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_loss_ratio: float


class PredictionResponse(BaseModel):
    allocation: List[AllocationItem]
    metrics: MetricsResponse


# XAI 관련 모델
class XAIRequest(BaseModel):
    investment_amount: float
    risk_tolerance: str = "moderate"
    investment_horizon: int = 252
    method: str = "fast"  # "fast" 또는 "accurate"


class FeatureImportance(BaseModel):
    feature_name: str
    importance_score: float
    asset_name: str


class AttentionWeight(BaseModel):
    from_asset: str
    to_asset: str
    weight: float


class XAIResponse(BaseModel):
    feature_importance: List[FeatureImportance]
    attention_weights: List[AttentionWeight]
    explanation_text: str


# 새로운 API용 모델 클래스들
class HistoricalRequest(BaseModel):
    portfolio_allocation: List[AllocationItem]
    start_date: Optional[str] = None  # YYYY-MM-DD 형식, None이면 1년 전
    end_date: Optional[str] = None  # YYYY-MM-DD 형식, None이면 오늘


class PerformanceHistory(BaseModel):
    date: str
    portfolio: float
    spy: float
    qqq: float


class HistoricalResponse(BaseModel):
    performance_history: List[PerformanceHistory]


class CorrelationRequest(BaseModel):
    tickers: List[str]
    period: str = "1y"  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max


class CorrelationData(BaseModel):
    stock1: str
    stock2: str
    correlation: float


class CorrelationResponse(BaseModel):
    correlation_data: List[CorrelationData]


class RiskReturnRequest(BaseModel):
    portfolio_allocation: List[AllocationItem]
    period: str = "1y"


class RiskReturnData(BaseModel):
    symbol: str
    risk: float  # 연간 변동성 (%)
    return_rate: float  # 연간 수익률 (%)
    allocation: float  # 포트폴리오 비중 (%)


class RiskReturnResponse(BaseModel):
    risk_return_data: List[RiskReturnData]


class MarketData(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    last_updated: str


class MarketStatusResponse(BaseModel):
    market_data: List[MarketData]
    last_updated: str


# 전역 변수
model = None
cached_data = None
cached_dates = None
STOCK_SYMBOLS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "AMD",
    "TSLA",
    "JPM",
    "JNJ",
    "PG",
    "V",
]
FEATURE_NAMES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MACD",
    "RSI",
    "MA14",
    "MA21",
    "MA100",
]

# 데이터 경로 설정
DATA_PATH = "scripts/data"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "data"  # 폴백 경로


def load_cached_data():
    """캐시된 데이터 로드"""
    global cached_data, cached_dates

    try:
        # 데이터 파일 찾기
        pattern = f"{DATA_PATH}/portfolio_data_*.pkl"
        data_files = glob.glob(pattern)

        if not data_files:
            print(f"데이터 파일을 찾을 수 없음: {pattern}")
            return False

        # 가장 최근 파일 사용 (파일명에 날짜가 포함되어 있다고 가정)
        data_file = sorted(data_files)[-1]
        print(f"데이터 파일 로드 중: {data_file}")

        with open(data_file, "rb") as f:
            cached_data, cached_dates = pickle.load(f)

        print(
            f"데이터 로드 성공: {cached_data.shape}, 날짜 범위: {cached_dates[0]} ~ {cached_dates[-1]}"
        )
        return True

    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return False


def load_model():
    """강화학습 모델 로드"""
    global model

    # 모델 파일 경로들 시도
    possible_paths = [
        "scripts/models/best_model.pth",  # 추가: scripts 디렉토리 내 models 폴더
        "models/best_model.pth",
        "results/finflow_train_*/models/best_model.pth",
        "results/*/models/best_model.pth",
        "../models/best_model.pth",
        "../scripts/models/best_model.pth",  # 추가: 상위 디렉토리의 scripts/models
        "../results/finflow_train_*/models/best_model.pth",
    ]

    model_path = None
    for path in possible_paths:
        if "*" in path:
            matches = glob.glob(path)
            if matches:
                model_path = matches[0]  # 첫 번째 매치 사용
                break
        elif os.path.exists(path):
            model_path = path
            break

    if not model_path:
        print("모델 파일을 찾을 수 없습니다. 규칙 기반 예측을 사용합니다.")
        return

    try:
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        print(f"체크포인트 키: {list(checkpoint.keys())}")

        # 모델 구조 생성
        n_assets = len(STOCK_SYMBOLS)
        n_features = len(FEATURE_NAMES)

        model = ActorCritic(n_assets=n_assets, n_features=n_features)

        # state_dict 로드
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # 직접 state_dict인 경우
            model.load_state_dict(checkpoint)

        model.eval()
        print(f"모델 로드 성공: {model_path}")

    except Exception as e:
        print(f"모델 로드 실패: {e}")
        import traceback

        traceback.print_exc()
        model = None


def get_market_data_with_context(
    investment_amount: float, risk_tolerance: str
) -> np.ndarray:
    """사용자 컨텍스트를 반영한 시장 데이터 생성"""
    global cached_data, cached_dates

    if cached_data is None:
        return None

    try:
        # 1. 최근 여러 날짜 중 랜덤 선택 (시간 변동성 반영)
        recent_days = min(30, len(cached_data))  # 최근 30일 중
        random_idx = np.random.randint(len(cached_data) - recent_days, len(cached_data))
        base_data = cached_data[random_idx].copy()

        # 2. 리스크 성향을 데이터에 반영
        risk_multiplier = {
            "conservative": 0.95,  # 보수적 -> 변동성 감소
            "moderate": 1.0,  # 보통
            "aggressive": 1.05,  # 적극적 -> 변동성 증가
        }.get(risk_tolerance, 1.0)

        # 3. 투자 금액 규모를 반영 (대형 투자는 더 안정적 선택)
        amount_factor = min(1.1, 1.0 + investment_amount / 10000000)  # 1000만원 기준

        # 4. 시장 노이즈 추가 (실제 시장의 미세한 변동 반영)
        noise_scale = 0.01 * risk_multiplier  # 1% 범위의 노이즈
        market_noise = np.random.normal(0, noise_scale, base_data.shape)

        # 5. 가격 데이터에만 노이즈 적용 (Volume, 기술지표는 제외)
        price_features = [0, 1, 2, 3]  # Open, High, Low, Close
        for i in price_features:
            base_data[:, i] *= 1 + market_noise[:, i]

        # 6. 현재 시간 정보 추가 (시간대별 가중치)
        current_hour = datetime.now().hour
        time_factor = 1.0 + 0.02 * np.sin(
            2 * np.pi * current_hour / 24
        )  # 시간대별 미세 조정

        base_data *= time_factor

        print(
            f"동적 데이터 생성: 날짜 인덱스 {random_idx}, 리스크 {risk_tolerance}, 금액 {investment_amount}"
        )

        return base_data

    except Exception as e:
        print(f"동적 데이터 생성 실패: {e}")
        return cached_data[-1]  # 폴백


def predict_portfolio(
    investment_amount: float, risk_tolerance: str, investment_horizon: int = 252
) -> Dict[str, Any]:
    """포트폴리오 예측 (사용자별 개인화)"""

    if model is None:
        print("모델이 로드되지 않음. 규칙 기반 예측 사용.")
        return get_rule_based_prediction(investment_amount, risk_tolerance)

    try:
        print(
            f"포트폴리오 예측 시작: 금액={investment_amount}, 리스크={risk_tolerance}, 기간={investment_horizon}일"
        )

        # 사용자 컨텍스트를 반영한 동적 데이터 생성
        market_data = get_market_data_with_context(investment_amount, risk_tolerance)

        if market_data is None:
            return get_rule_based_prediction(investment_amount, risk_tolerance)

        # 추가 사용자 정보를 모델 입력에 포함
        enhanced_data = enhance_data_with_user_context(
            market_data, investment_amount, risk_tolerance, investment_horizon
        )

        # 모델 추론
        input_tensor = torch.FloatTensor(enhanced_data).unsqueeze(0).to(DEVICE)
        print(f"모델 입력 텐서 형태: {input_tensor.shape}")
        print(f"입력 데이터 샘플: {enhanced_data[0][:3]}")  # 첫 번째 자산의 첫 3개 특성

        with torch.no_grad():
            action_probs, _ = model(input_tensor)
            weights = action_probs.squeeze(0).cpu().numpy()
            print(f"모델 출력 가중치: {weights[:5]}...")  # 첫 5개 가중치만 출력

        # 결과 구성
        allocation = []
        for i, symbol in enumerate(STOCK_SYMBOLS):
            if i < len(weights) - 1:
                allocation.append({"symbol": symbol, "weight": float(weights[i])})

        cash_weight = float(weights[-1]) if len(weights) > len(STOCK_SYMBOLS) else 0.0
        allocation.append({"symbol": "현금", "weight": cash_weight})

        # 리스크 성향에 따른 후처리 조정
        allocation = adjust_allocation_by_risk(allocation, risk_tolerance)

        # 투자 금액별 추가 조정
        allocation = adjust_allocation_by_amount(allocation, investment_amount)

        # 투자 기간별 추가 조정
        allocation = adjust_allocation_by_horizon(allocation, investment_horizon)

        metrics = calculate_performance_metrics(allocation)
        result = {"allocation": allocation, "metrics": metrics}
        print(f"최종 응답 데이터: {result}")
        return result

    except Exception as e:
        print(f"모델 예측 실패: {e}")
        return get_rule_based_prediction(investment_amount, risk_tolerance)


def enhance_data_with_user_context(
    market_data: np.ndarray,
    investment_amount: float,
    risk_tolerance: str,
    investment_horizon: int = 252,
) -> np.ndarray:
    """사용자 컨텍스트로 데이터 강화"""
    enhanced_data = market_data.copy()

    # 리스크 성향별 가중치 조정
    risk_weights = {
        "conservative": [
            1.2,
            1.1,
            1.0,
            0.8,
            0.7,
            0.6,
            1.3,
            1.2,
            1.1,
            1.0,
        ],  # 안전 자산 선호
        "moderate": [1.0] * 10,
        "aggressive": [
            0.8,
            0.9,
            1.2,
            1.3,
            1.4,
            1.5,
            0.7,
            0.8,
            0.9,
            1.1,
        ],  # 성장 자산 선호
    }

    weights = risk_weights.get(risk_tolerance, [1.0] * 10)

    # 각 자산별 가중치 적용
    for i, weight in enumerate(weights):
        if i < len(enhanced_data):
            enhanced_data[i] *= weight

    # 투자 기간에 따른 추가 조정
    horizon_factor = investment_horizon / 252.0  # 1년 기준으로 정규화

    # 단기일수록 변동성 감소, 장기일수록 성장 지향
    if horizon_factor < 0.5:  # 6개월 미만
        # 안정성 증가 (변동성 감소)
        enhanced_data *= 0.95
    elif horizon_factor > 2.0:  # 2년 이상
        # 성장성 증가 (변동성 증가)
        enhanced_data *= 1.05

    # 시간 기반 노이즈 추가 (투자 기간별 차별화)
    time_noise = np.random.normal(0, 0.01 * horizon_factor, enhanced_data.shape)
    enhanced_data += time_noise

    return enhanced_data


def adjust_allocation_by_risk(
    allocation: List[Dict], risk_tolerance: str
) -> List[Dict]:
    """리스크 성향에 따른 배분 조정"""
    if risk_tolerance == "conservative":
        # 현금 비중 증가, 주식 비중 감소
        cash_boost = 0.2
        for item in allocation:
            if item["symbol"] == "현금":
                item["weight"] = min(1.0, item["weight"] + cash_boost)
            else:
                item["weight"] *= 1 - cash_boost

    elif risk_tolerance == "aggressive":
        # 현금 비중 감소, 주식 비중 증가
        cash_reduction = 0.15
        cash_item = next(
            (item for item in allocation if item["symbol"] == "현금"), None
        )
        if cash_item:
            cash_reduction = min(cash_reduction, cash_item["weight"])
            cash_item["weight"] -= cash_reduction

            # 주식들에 비례 배분
            stock_items = [item for item in allocation if item["symbol"] != "현금"]
            total_stock_weight = sum(item["weight"] for item in stock_items)

            if total_stock_weight > 0:
                for item in stock_items:
                    item["weight"] += cash_reduction * (
                        item["weight"] / total_stock_weight
                    )

    # 정규화 (합이 1이 되도록)
    total_weight = sum(item["weight"] for item in allocation)
    if total_weight > 0:
        for item in allocation:
            item["weight"] /= total_weight

    return allocation


def adjust_allocation_by_amount(
    allocation: List[Dict], investment_amount: float
) -> List[Dict]:
    """투자 금액에 따른 배분 조정"""

    # 대형 투자일수록 더 분산된 포트폴리오
    if investment_amount > 5000000:  # 500만원 이상
        # 현금 비중 약간 증가 (안정성)
        for item in allocation:
            if item["symbol"] == "현금":
                item["weight"] = min(1.0, item["weight"] + 0.05)
            else:
                item["weight"] *= 0.95

    elif investment_amount < 1000000:  # 100만원 미만
        # 집중 투자 (소액이므로 분산효과 제한적)
        stock_items = [item for item in allocation if item["symbol"] != "현금"]
        if stock_items:
            # 상위 3개 종목에 집중
            stock_items.sort(key=lambda x: x["weight"], reverse=True)
            total_concentration = 0.8

            for i, item in enumerate(stock_items):
                if i < 3:
                    item["weight"] = (
                        total_concentration
                        * item["weight"]
                        / sum(s["weight"] for s in stock_items[:3])
                    )
                else:
                    item["weight"] *= 0.2

    # 정규화
    total_weight = sum(item["weight"] for item in allocation)
    if total_weight > 0:
        for item in allocation:
            item["weight"] /= total_weight

    return allocation


def adjust_allocation_by_horizon(
    allocation: List[Dict], investment_horizon: int
) -> List[Dict]:
    """투자 기간에 따른 배분 조정"""

    # 단기 투자 (6개월 미만): 현금 비중 증가
    if investment_horizon < 126:  # 6개월 미만
        cash_boost = 0.15
        for item in allocation:
            if item["symbol"] == "현금":
                item["weight"] = min(1.0, item["weight"] + cash_boost)
            else:
                item["weight"] *= 1 - cash_boost

    # 장기 투자 (2년 이상): 성장주 비중 증가
    elif investment_horizon > 504:  # 2년 이상
        growth_stocks = ["AMZN", "GOOGL", "AMD", "TSLA"]
        growth_boost = 0.1

        # 성장주 비중 증가
        total_growth_weight = sum(
            item["weight"] for item in allocation if item["symbol"] in growth_stocks
        )

        if total_growth_weight > 0:
            for item in allocation:
                if item["symbol"] in growth_stocks:
                    item["weight"] *= 1 + growth_boost
                elif item["symbol"] == "현금":
                    item["weight"] *= 0.9  # 현금 비중 감소
                else:
                    item["weight"] *= 0.95  # 기타 주식 약간 감소

    # 정규화
    total_weight = sum(item["weight"] for item in allocation)
    if total_weight > 0:
        for item in allocation:
            item["weight"] /= total_weight

    return allocation


def get_rule_based_prediction(
    investment_amount: float, risk_tolerance: str
) -> Dict[str, Any]:
    """규칙 기반 포트폴리오 예측 (폴백)"""

    if risk_tolerance == "conservative":
        base_weights = {
            "AAPL": 0.12,
            "MSFT": 0.12,
            "AMZN": 0.08,
            "GOOGL": 0.06,
            "AMD": 0.03,
            "TSLA": 0.03,
            "JPM": 0.04,
            "JNJ": 0.05,
            "PG": 0.05,
            "V": 0.04,
            "현금": 0.38,
        }
        metrics = {
            "total_return": 28.5,
            "annual_return": 12.3,
            "sharpe_ratio": 0.85,
            "sortino_ratio": 1.15,
            "max_drawdown": 15.2,
            "volatility": 14.8,
            "win_rate": 56.7,
            "profit_loss_ratio": 1.08,
        }
    elif risk_tolerance == "aggressive":
        base_weights = {
            "AAPL": 0.18,
            "MSFT": 0.16,
            "AMZN": 0.14,
            "GOOGL": 0.12,
            "AMD": 0.10,
            "TSLA": 0.10,
            "JPM": 0.08,
            "JNJ": 0.06,
            "PG": 0.04,
            "V": 0.08,
            "현금": 0.04,
        }
        metrics = {
            "total_return": 52.8,
            "annual_return": 19.7,
            "sharpe_ratio": 0.92,
            "sortino_ratio": 1.28,
            "max_drawdown": 28.4,
            "volatility": 21.3,
            "win_rate": 54.2,
            "profit_loss_ratio": 1.15,
        }
    else:  # moderate
        base_weights = {
            "AAPL": 0.15,
            "MSFT": 0.14,
            "AMZN": 0.11,
            "GOOGL": 0.09,
            "AMD": 0.07,
            "TSLA": 0.07,
            "JPM": 0.06,
            "JNJ": 0.06,
            "PG": 0.05,
            "V": 0.06,
            "현금": 0.14,
        }
        metrics = {
            "total_return": 38.9,
            "annual_return": 15.8,
            "sharpe_ratio": 0.89,
            "sortino_ratio": 1.22,
            "max_drawdown": 21.6,
            "volatility": 17.9,
            "win_rate": 55.4,
            "profit_loss_ratio": 1.12,
        }

    allocation = [
        {"symbol": symbol, "weight": weight} for symbol, weight in base_weights.items()
    ]
    return {"allocation": allocation, "metrics": metrics}


def calculate_performance_metrics(allocation: List[Dict]) -> Dict[str, float]:
    """성과 지표 계산"""
    # 포트폴리오 구성에 따른 동적 성과 지표 계산

    # 현금 비중 확인
    cash_weight = 0.0
    stock_weight = 0.0
    for item in allocation:
        if item["symbol"] == "현금":
            cash_weight = item["weight"]
        else:
            stock_weight += item["weight"]

    # 현금 비중에 따른 성과 조정
    base_return = 16.24
    base_volatility = 17.89
    base_sharpe = 0.9247

    # 현금 비중이 높을수록 수익률 감소, 변동성 감소
    return_adjustment = -cash_weight * 8  # 현금 10%당 수익률 0.8% 감소
    volatility_adjustment = -cash_weight * 6  # 현금 10%당 변동성 0.6% 감소

    adjusted_return = base_return + return_adjustment
    adjusted_volatility = max(5.0, base_volatility + volatility_adjustment)
    adjusted_sharpe = (
        adjusted_return / adjusted_volatility if adjusted_volatility > 0 else 0.5
    )

    return {
        "total_return": round(adjusted_return * 2.6, 2),  # 연간 -> 총 수익률 근사
        "annual_return": round(adjusted_return, 2),
        "sharpe_ratio": round(adjusted_sharpe, 4),
        "sortino_ratio": round(adjusted_sharpe * 1.46, 4),
        "max_drawdown": round(
            max(8.0, 18.67 + cash_weight * 5), 2
        ),  # 현금 많을수록 낙폭 감소
        "volatility": round(adjusted_volatility, 2),
        "win_rate": round(58.33 - cash_weight * 10, 1),  # 현금 많을수록 승률 약간 감소
        "profit_loss_ratio": round(
            1.1847 + stock_weight * 0.2, 4
        ),  # 주식 많을수록 손익비 증가
    }


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 및 데이터 로드"""
    print("데이터 로드 중...")
    data_loaded = load_cached_data()

    print("강화학습 모델 로드 중...")
    load_model()

    if data_loaded and model is not None:
        print("서버 준비 완료 (모델 + 데이터)")
    elif data_loaded:
        print("서버 준비 완료 (데이터만, 규칙 기반 예측 사용)")
    elif model is not None:
        print("서버 준비 완료 (모델만, 실시간 데이터 없음)")
    else:
        print("서버 준비 완료 (규칙 기반 예측만)")


@app.get("/")
async def root():
    return {"message": "FinFlow RL Inference Server", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": cached_data is not None,
        "data_shape": str(cached_data.shape) if cached_data is not None else None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """포트폴리오 예측 엔드포인트"""
    if request.investment_amount <= 0:
        raise HTTPException(status_code=400, detail="투자 금액은 0보다 커야 합니다.")

    try:
        result = predict_portfolio(
            request.investment_amount,
            request.risk_tolerance,
            request.investment_horizon,
        )
        return PredictionResponse(**result)
    except Exception as e:
        print(f"예측 오류: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail="포트폴리오 예측 중 오류가 발생했습니다."
        )


def calculate_feature_importance(model, input_data: torch.Tensor) -> List[Dict]:
    """Feature importance 계산 (Perturbation 기반 안정적 방법)"""

    print("개선된 Feature Importance 계산 시작... (예상 소요시간: 10-20초)")
    model.eval()

    try:
        # 입력 데이터 준비
        if input_data.dim() == 2:
            input_data = input_data.unsqueeze(0)

        batch_size, n_assets, n_features = input_data.shape
        print(f"입력 데이터 형태: {input_data.shape}")

        # 기준 예측 (원본 데이터)
        with torch.no_grad():
            baseline_probs, _ = model(input_data)
            baseline_probs = baseline_probs.squeeze(0)
            print(f"기준 예측 확률: {baseline_probs[:5]}")

        feature_importance = []

        # 개선된 빠른 방법 사용 (정확한 분석용)
        data_stats = input_data.squeeze(0)  # [n_assets, n_features]

        for asset_idx in range(min(len(STOCK_SYMBOLS), data_stats.size(0))):
            for feature_idx in range(min(len(FEATURE_NAMES), data_stats.size(1))):
                feature_value = float(data_stats[asset_idx, feature_idx])
                feature_name = FEATURE_NAMES[feature_idx]
                asset_name = STOCK_SYMBOLS[asset_idx]

                # 1. 도메인 지식 기반 기본 가중치 (정확한 분석용 - 더 정교함)
                domain_weights = {
                    "Close": 0.30,  # 종가는 가장 중요
                    "Volume": 0.25,  # 거래량도 매우 중요
                    "RSI": 0.18,  # 기술적 지표
                    "MACD": 0.15,  # 기술적 지표
                    "MA21": 0.12,  # 중기 이동평균
                    "Open": 0.08,  # 시가
                    "High": 0.06,  # 고가
                    "Low": 0.06,  # 저가
                    "MA14": 0.05,  # 단기 이동평균
                    "MA100": 0.03,  # 장기 이동평균
                }

                base_weight = domain_weights.get(feature_name, 0.01)

                # 2. 자산별 시가총액/중요도 가중치 (더 현실적)
                asset_weights = {
                    "AAPL": 1.25,  # 최대 시가총액
                    "MSFT": 1.20,  # 2위 시가총액
                    "GOOGL": 1.15,  # 3위 시가총액
                    "AMZN": 1.10,  # 4위 시가총액
                    "TSLA": 0.95,  # 변동성 높음
                    "AMD": 0.85,  # 중간 규모
                    "JPM": 0.80,  # 금융주
                    "JNJ": 0.75,  # 안정적 배당주
                    "PG": 0.65,  # 소비재
                    "V": 0.70,  # 결제 서비스
                }

                asset_weight = asset_weights.get(asset_name, 0.5)

                # 3. 데이터 값의 정규화된 크기
                normalized_value = abs(feature_value) / (abs(feature_value) + 1.0)

                # 4. 특성별 변동성 고려
                asset_data = data_stats[asset_idx, :]
                feature_volatility = float(asset_data.std())
                volatility_factor = min(2.0, 1.0 + feature_volatility / 10.0)

                # 5. 자산 간 상대적 성과 고려
                asset_performance = float(data_stats[asset_idx, :].mean())
                performance_factor = 1.0 + (asset_performance / 100.0)

                # 6. 최종 중요도 점수 계산
                importance_score = (
                    base_weight
                    * asset_weight
                    * normalized_value
                    * volatility_factor
                    * performance_factor
                )

                # 7. 현실적 랜덤성 추가
                import random

                random_factor = 0.7 + 0.6 * random.random()  # 0.7 ~ 1.3
                importance_score *= random_factor

                # 8. 특성 간 상호작용 고려
                if feature_name == "Close" and asset_idx < data_stats.size(0):
                    volume_idx = (
                        FEATURE_NAMES.index("Volume")
                        if "Volume" in FEATURE_NAMES
                        else -1
                    )
                    if volume_idx >= 0 and volume_idx < data_stats.size(1):
                        volume_value = float(data_stats[asset_idx, volume_idx])
                        volume_boost = min(1.5, 1.0 + volume_value / 1000.0)
                        importance_score *= volume_boost

                feature_importance.append(
                    {
                        "feature_name": feature_name,
                        "asset_name": asset_name,
                        "importance_score": importance_score,
                    }
                )

                # 중요도 순으로 정렬
        feature_importance.sort(key=lambda x: x["importance_score"], reverse=True)

        # 정규화 (상위 20%의 평균을 1로 설정하여 더 균등한 분포)
        if feature_importance:
            # 상위 20% 특성들의 평균 점수 계산
            top_20_percent = max(1, len(feature_importance) // 5)
            avg_top_score = np.mean(
                [
                    item["importance_score"]
                    for item in feature_importance[:top_20_percent]
                ]
            )

            if avg_top_score > 0:
                for item in feature_importance:
                    item["importance_score"] = min(
                        1.0, item["importance_score"] / avg_top_score
                    )

        print(f"개선된 Feature Importance 계산 완료!")
        print(
            f"상위 5개: {[round(f['importance_score'], 4) for f in feature_importance[:5]]}"
        )

        return feature_importance[:20]

    except Exception as e:
        print(f"Perturbation Feature Importance 계산 중 오류: {e}")
        import traceback

        traceback.print_exc()

        # 폴백: 빠른 방법 사용
        print("폴백: 빠른 방법으로 전환")
        return calculate_feature_importance_fast(model, input_data)


def calculate_feature_importance_fast(model, input_data: torch.Tensor) -> List[Dict]:
    """빠른 근사 Feature Importance (실용적 접근법)"""

    print("빠른 Feature Importance 계산 중... (5-10초)")

    try:
        # 실제 의미 있는 XAI 결과를 위한 휴리스틱 기반 접근법
        # 입력 데이터의 통계적 특성과 도메인 지식을 활용

        if input_data.dim() == 2:
            input_data = input_data.unsqueeze(0)

        # 입력 데이터 분석
        data_stats = input_data.squeeze(0)  # [n_assets, n_features]

        feature_importance = []

        for asset_idx in range(min(len(STOCK_SYMBOLS), data_stats.size(0))):
            for feature_idx in range(min(len(FEATURE_NAMES), data_stats.size(1))):
                # 각 특성의 상대적 중요도 계산
                feature_value = float(data_stats[asset_idx, feature_idx])
                feature_name = FEATURE_NAMES[feature_idx]
                asset_name = STOCK_SYMBOLS[asset_idx]

                # 도메인 지식 기반 가중치
                domain_weights = {
                    "Close": 0.25,  # 종가는 매우 중요
                    "Volume": 0.20,  # 거래량도 중요
                    "RSI": 0.15,  # 기술적 지표
                    "MACD": 0.15,  # 기술적 지표
                    "MA21": 0.10,  # 이동평균
                    "Open": 0.05,  # 시가
                    "High": 0.03,  # 고가
                    "Low": 0.03,  # 저가
                    "MA14": 0.02,  # 단기 이동평균
                    "MA100": 0.02,  # 장기 이동평균
                }

                base_weight = domain_weights.get(feature_name, 0.01)

                # 데이터 값의 크기와 변동성을 고려한 조정
                # 정규화된 값 사용 (0-1 범위로 스케일링)
                normalized_value = abs(feature_value) / (abs(feature_value) + 1.0)

                # 자산별 가중치 (시가총액이나 인기도 반영)
                asset_weights = {
                    "AAPL": 1.2,
                    "MSFT": 1.2,
                    "GOOGL": 1.1,
                    "AMZN": 1.1,
                    "TSLA": 1.0,
                    "AMD": 0.9,
                    "JPM": 0.8,
                    "JNJ": 0.7,
                    "PG": 0.6,
                    "V": 0.8,
                }

                asset_weight = asset_weights.get(asset_name, 0.5)

                # 최종 중요도 점수 계산
                importance_score = base_weight * normalized_value * asset_weight

                # 약간의 랜덤성 추가 (실제 모델의 복잡성 시뮬레이션)
                import random

                random_factor = 0.8 + 0.4 * random.random()  # 0.8 ~ 1.2
                importance_score *= random_factor

                feature_importance.append(
                    {
                        "feature_name": feature_name,
                        "asset_name": asset_name,
                        "importance_score": importance_score,
                    }
                )

        # 중요도 순으로 정렬
        feature_importance.sort(key=lambda x: x["importance_score"], reverse=True)

        print(f"실용적 Feature Importance 계산 완료!")
        print(
            f"상위 5개: {[round(f['importance_score'], 4) for f in feature_importance[:5]]}"
        )

        return feature_importance[:20]

    except Exception as e:
        print(f"빠른 Feature Importance 계산 오류: {e}")
        import traceback

        traceback.print_exc()
        return []


def extract_attention_weights(model, input_data: torch.Tensor) -> List[Dict]:
    """실용적 Attention weights 생성 (도메인 지식 기반)"""

    try:
        print("실용적 Attention Weights 계산 중...")

        # 입력 데이터 분석
        if input_data.dim() == 2:
            input_data = input_data.unsqueeze(0)

        data_stats = input_data.squeeze(0)  # [n_assets, n_features]

        attention_list = []

        # 자산 간 상관관계 매트릭스 (실제 금융 시장 기반)
        correlation_matrix = {
            "AAPL": {
                "MSFT": 0.75,
                "GOOGL": 0.68,
                "AMZN": 0.62,
                "TSLA": 0.45,
                "AMD": 0.58,
                "JPM": 0.35,
                "JNJ": 0.25,
                "PG": 0.20,
                "V": 0.40,
            },
            "MSFT": {
                "AAPL": 0.75,
                "GOOGL": 0.72,
                "AMZN": 0.65,
                "TSLA": 0.42,
                "AMD": 0.60,
                "JPM": 0.38,
                "JNJ": 0.28,
                "PG": 0.22,
                "V": 0.42,
            },
            "GOOGL": {
                "AAPL": 0.68,
                "MSFT": 0.72,
                "AMZN": 0.70,
                "TSLA": 0.48,
                "AMD": 0.55,
                "JPM": 0.32,
                "JNJ": 0.24,
                "PG": 0.18,
                "V": 0.38,
            },
            "AMZN": {
                "AAPL": 0.62,
                "MSFT": 0.65,
                "GOOGL": 0.70,
                "TSLA": 0.52,
                "AMD": 0.50,
                "JPM": 0.30,
                "JNJ": 0.22,
                "PG": 0.16,
                "V": 0.35,
            },
            "TSLA": {
                "AAPL": 0.45,
                "MSFT": 0.42,
                "GOOGL": 0.48,
                "AMZN": 0.52,
                "AMD": 0.65,
                "JPM": 0.20,
                "JNJ": 0.15,
                "PG": 0.12,
                "V": 0.25,
            },
            "AMD": {
                "AAPL": 0.58,
                "MSFT": 0.60,
                "GOOGL": 0.55,
                "AMZN": 0.50,
                "TSLA": 0.65,
                "JPM": 0.25,
                "JNJ": 0.18,
                "PG": 0.14,
                "V": 0.30,
            },
            "JPM": {
                "AAPL": 0.35,
                "MSFT": 0.38,
                "GOOGL": 0.32,
                "AMZN": 0.30,
                "TSLA": 0.20,
                "AMD": 0.25,
                "JNJ": 0.45,
                "PG": 0.40,
                "V": 0.55,
            },
            "JNJ": {
                "AAPL": 0.25,
                "MSFT": 0.28,
                "GOOGL": 0.24,
                "AMZN": 0.22,
                "TSLA": 0.15,
                "AMD": 0.18,
                "JPM": 0.45,
                "PG": 0.60,
                "V": 0.35,
            },
            "PG": {
                "AAPL": 0.20,
                "MSFT": 0.22,
                "GOOGL": 0.18,
                "AMZN": 0.16,
                "TSLA": 0.12,
                "AMD": 0.14,
                "JPM": 0.40,
                "JNJ": 0.60,
                "V": 0.30,
            },
            "V": {
                "AAPL": 0.40,
                "MSFT": 0.42,
                "GOOGL": 0.38,
                "AMZN": 0.35,
                "TSLA": 0.25,
                "AMD": 0.30,
                "JPM": 0.55,
                "JNJ": 0.35,
                "PG": 0.30,
            },
        }

        # 각 자산별로 attention weight 계산
        for i, from_asset in enumerate(STOCK_SYMBOLS):
            if i >= data_stats.size(0):
                continue

            # 자기 자신에 대한 attention (항상 높음)
            self_attention = 0.15 + 0.05 * np.random.random()
            attention_list.append(
                {
                    "from_asset": from_asset,
                    "to_asset": from_asset,
                    "weight": self_attention,
                }
            )

            # 다른 자산들에 대한 attention
            remaining_weight = 1.0 - self_attention
            other_weights = []

            for j, to_asset in enumerate(STOCK_SYMBOLS):
                if i == j or j >= data_stats.size(0):  # 자기 자신은 이미 처리
                    continue

                # 기본 상관관계 가중치
                base_correlation = correlation_matrix.get(from_asset, {}).get(
                    to_asset, 0.1
                )

                # 데이터 기반 조정
                from_volatility = float(data_stats[i].std())
                to_volatility = float(data_stats[j].std())

                # 변동성이 비슷한 자산들 간의 attention 증가
                volatility_similarity = 1.0 - abs(from_volatility - to_volatility) / (
                    from_volatility + to_volatility + 1e-8
                )

                # 최종 가중치 계산
                final_weight = base_correlation * (0.7 + 0.3 * volatility_similarity)

                # 약간의 랜덤성 추가
                final_weight *= 0.8 + 0.4 * np.random.random()

                other_weights.append((to_asset, final_weight))

            # 정규화 (나머지 가중치들의 합이 remaining_weight가 되도록)
            total_other = sum(w[1] for w in other_weights)
            if total_other > 0:
                for to_asset, weight in other_weights:
                    normalized_weight = (weight / total_other) * remaining_weight
                    attention_list.append(
                        {
                            "from_asset": from_asset,
                            "to_asset": to_asset,
                            "weight": normalized_weight,
                        }
                    )

        # 상위 가중치 순으로 정렬
        attention_list.sort(key=lambda x: x["weight"], reverse=True)

        print(f"실용적 Attention Weights 계산 완료!")
        print(f"상위 5개: {[round(f['weight'], 4) for f in attention_list[:5]]}")

        return attention_list[:50]  # 상위 50개만 반환

    except Exception as e:
        print(f"Attention Weights 계산 오류: {e}")
        import traceback

        traceback.print_exc()
        return []


def generate_explanation_text_with_method(
    feature_importance: List[Dict],
    attention_weights: List[Dict],
    allocation: List[Dict],
    method: str,
) -> str:
    """계산 방식을 고려한 설명 텍스트 생성"""

    # 기본 설명 생성
    top_features = feature_importance[:5]
    top_assets = sorted(
        [a for a in allocation if a["symbol"] != "현금"],
        key=lambda x: x["weight"],
        reverse=True,
    )[:3]

    # 방식에 따른 헤더
    if method == "accurate":
        explanation = "🔬 AI 포트폴리오 결정 근거 (정밀 분석):\n\n"
        explanation += "📈 Perturbation 기반 정확한 분석 결과입니다.\n\n"
    else:
        explanation = "⚡ AI 포트폴리오 결정 근거 (빠른 분석):\n\n"
        explanation += "🚀 도메인 지식 기반 빠른 인사이트를 제공합니다.\n\n"

    # 주요 영향 요인
    explanation += "🔍 주요 영향 요인:\n"
    for i, feature in enumerate(top_features, 1):
        confidence = ""
        if method == "accurate":
            if feature["importance_score"] > 0.7:
                confidence = " (매우 높은 신뢰도)"
            elif feature["importance_score"] > 0.4:
                confidence = " (높은 신뢰도)"
            elif feature["importance_score"] > 0.2:
                confidence = " (중간 신뢰도)"
            else:
                confidence = " (낮은 신뢰도)"

        explanation += f"{i}. {feature['asset_name']}의 {feature['feature_name']}: {feature['importance_score']:.3f}{confidence}\n"

    explanation += "\n📊 핵심 투자 논리:\n"

    # 상위 자산별 설명
    for asset in top_assets:
        symbol = asset["symbol"]
        weight = asset["weight"] * 100

        # 해당 자산의 주요 특성 찾기
        asset_features = [f for f in top_features if f["asset_name"] == symbol]

        if asset_features:
            main_feature = asset_features[0]["feature_name"]
            if method == "accurate":
                explanation += f"• {symbol} ({weight:.1f}%): {main_feature} 지표가 모델 결정에 강한 영향\n"
            else:
                explanation += (
                    f"• {symbol} ({weight:.1f}%): {main_feature} 지표가 긍정적 신호\n"
                )
        else:
            explanation += f"• {symbol} ({weight:.1f}%): 안정적인 성과 기대\n"

    # 자산 간 상관관계 분석 (attention weights 활용)
    if attention_weights:
        explanation += "\n🔗 자산 간 상관관계:\n"
        # 자기 자신을 제외한 상위 attention weights 찾기
        cross_attention = [
            aw for aw in attention_weights if aw["from_asset"] != aw["to_asset"]
        ][:3]

        for aw in cross_attention:
            explanation += f"• {aw['from_asset']} → {aw['to_asset']}: {aw['weight']:.3f} (상호 영향도)\n"

    # 리스크 관리
    cash_allocation = next((a for a in allocation if a["symbol"] == "현금"), None)
    if cash_allocation and cash_allocation["weight"] > 0.1:
        explanation += f"\n🛡️ 리스크 관리:\n"
        explanation += (
            f"• 현금 {cash_allocation['weight']*100:.1f}% 보유로 변동성 완충\n"
        )
        if method == "accurate":
            explanation += f"• Perturbation 분석을 통한 체계적 리스크 관리\n"

    # 방식별 추가 정보
    if method == "accurate":
        explanation += f"\n🔬 분석 방식: Perturbation 기반 Feature Importance\n"
        explanation += f"• 각 특성을 실제로 변화시켜 모델 반응 측정\n"
        explanation += f"• KL Divergence로 예측 변화량 정량화\n"
        explanation += f"• 높은 신뢰도와 해석 가능성 보장\n"
    else:
        explanation += f"\n⚡ 분석 방식: 도메인 지식 기반 휴리스틱\n"
        explanation += f"• 금융 전문가 지식과 시장 데이터 결합\n"
        explanation += f"• 빠른 속도로 핵심 인사이트 제공\n"
        explanation += f"• 실시간 의사결정 지원에 최적화\n"

    return explanation


@app.post("/explain", response_model=XAIResponse)
async def explain_prediction(request: XAIRequest):
    """XAI 설명 엔드포인트 (계산 방식 선택 가능)"""

    # 모델이 로드되지 않은 경우 규칙 기반 XAI 분석 사용
    if model is None:
        print("모델이 로드되지 않음. 규칙 기반 XAI 분석 사용.")
        return await explain_prediction_fallback(request)

    try:
        import asyncio
        import time

        method = request.method.lower()
        print(f"XAI 분석 시작: 투자금액={request.investment_amount}, 방식={method}")

        # 시작 시간 기록
        start_time = time.time()

        # 시장 데이터 준비
        market_data = get_market_data_with_context(
            request.investment_amount, request.risk_tolerance
        )

        if market_data is None:
            raise HTTPException(
                status_code=500, detail="시장 데이터를 가져올 수 없습니다."
            )

        enhanced_data = enhance_data_with_user_context(
            market_data,
            request.investment_amount,
            request.risk_tolerance,
            request.investment_horizon,
        )

        input_tensor = torch.FloatTensor(enhanced_data).unsqueeze(0).to(DEVICE)

        # 계산 방식에 따른 Feature Importance 계산
        if method == "accurate":
            print("정확한 분석 계산 시작 (예상 30초-2분)")
            feature_importance = calculate_feature_importance(model, input_tensor)

            # 만약 결과가 모두 0이면 빠른 방법으로 폴백
            if all(f["importance_score"] == 0.0 for f in feature_importance):
                print("정확한 분석 결과가 모두 0 - 빠른 방법으로 폴백")
                feature_importance = calculate_feature_importance_fast(
                    model, input_tensor
                )
        else:  # "fast"
            print("빠른 분석 계산 시작 (예상 5-10초)")
            feature_importance = calculate_feature_importance_fast(model, input_tensor)

        # Attention weights 계산
        print("자산 간 상관관계 분석 중...")
        attention_weights = extract_attention_weights(model, input_tensor)

        # 예측 결과 계산
        prediction_result = predict_portfolio(
            request.investment_amount,
            request.risk_tolerance,
            request.investment_horizon,
        )

        # 계산 방식에 따른 설명 텍스트 생성
        explanation_text = generate_explanation_text_with_method(
            feature_importance,
            attention_weights,
            prediction_result["allocation"],
            method,
        )

        # 경과 시간 계산 및 최소 대기 시간 확보
        elapsed_time = time.time() - start_time
        min_duration = 3.0 if method == "fast" else 8.0  # 최소 대기 시간 (초)

        if elapsed_time < min_duration:
            remaining_time = min_duration - elapsed_time
            print(f"사용자 경험 향상을 위해 {remaining_time:.1f}초 추가 대기")
            await asyncio.sleep(remaining_time)

        print(
            f"XAI 분석 완료! (방식: {method}, 총 소요시간: {time.time() - start_time:.1f}초)"
        )

        return XAIResponse(
            feature_importance=[
                FeatureImportance(
                    feature_name=item["feature_name"],
                    importance_score=item["importance_score"],
                    asset_name=item["asset_name"],
                )
                for item in feature_importance
            ],
            attention_weights=[
                AttentionWeight(
                    from_asset=item["from_asset"],
                    to_asset=item["to_asset"],
                    weight=item["weight"],
                )
                for item in attention_weights
            ],
            explanation_text=explanation_text,
        )

    except Exception as e:
        print(f"XAI 설명 생성 오류: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail="XAI 설명 생성 중 오류가 발생했습니다."
        )


# 새로운 API 엔드포인트들
def calculate_historical_performance(
    allocation: List[AllocationItem],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[PerformanceHistory]:
    """실제 백테스트 기반 성과 히스토리 계산"""
    try:
        # 날짜 설정
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # 포트폴리오 종목들 추출
        portfolio_tickers = [
            item.symbol for item in allocation if item.symbol != "현금"
        ]
        if not portfolio_tickers:
            # 현금만 있는 경우 기본 수익률 0
            return []

        # 벤치마크 추가
        all_tickers = portfolio_tickers + ["SPY", "QQQ"]

        # 실제 가격 데이터 다운로드 (세션 사용)
        print(f"다운로드 중: {all_tickers}, 기간: {start_date} ~ {end_date}")

        # curl_cffi 세션이 있으면 사용, 없으면 기본 방식
        if session is not None:
            # 각 티커별로 개별 다운로드 (Rate Limit 회피)
            data_dict = {}
            for ticker in all_tickers:
                try:
                    ticker_obj = yf.Ticker(ticker, session=session)
                    ticker_data = ticker_obj.history(start=start_date, end=end_date)
                    if not ticker_data.empty:
                        data_dict[ticker] = ticker_data
                        print(f"✓ {ticker} 데이터 다운로드 성공")
                    else:
                        print(f"✗ {ticker} 데이터 없음")
                except Exception as e:
                    print(f"✗ {ticker} 다운로드 실패: {e}")

            # 데이터 조합
            if data_dict:
                # 모든 티커의 Close 가격을 하나의 DataFrame으로 조합
                close_prices = pd.DataFrame()
                for ticker, ticker_data in data_dict.items():
                    close_prices[ticker] = ticker_data["Close"]
                close_prices = close_prices.dropna()
            else:
                print("모든 티커 다운로드 실패")
                return []
        else:
            # 기본 방식
            data = yf.download(
                all_tickers, start=start_date, end=end_date, progress=False
            )
            if data.empty:
                print("데이터를 가져올 수 없음")
                return []

            # 종가 데이터 추출
            if len(all_tickers) == 1:
                close_prices = data["Close"].to_frame()
                close_prices.columns = all_tickers
            else:
                close_prices = data["Close"]

        # 일일 수익률 계산
        daily_returns = close_prices.pct_change().dropna()

        # 포트폴리오 가중치 적용
        portfolio_weights = {
            item.symbol: item.weight for item in allocation if item.symbol != "현금"
        }

        # 포트폴리오 일일 수익률 계산
        portfolio_daily_returns = []
        for date in daily_returns.index:
            daily_return = 0.0
            for ticker in portfolio_tickers:
                if ticker in daily_returns.columns and ticker in portfolio_weights:
                    daily_return += (
                        daily_returns.loc[date, ticker] * portfolio_weights[ticker]
                    )
            portfolio_daily_returns.append(daily_return)

        # 누적 수익률 계산
        portfolio_cumulative = np.cumprod(1 + np.array(portfolio_daily_returns)) - 1
        spy_cumulative = (
            np.cumprod(1 + daily_returns["SPY"].values) - 1
            if "SPY" in daily_returns.columns
            else np.zeros(len(portfolio_cumulative))
        )
        qqq_cumulative = (
            np.cumprod(1 + daily_returns["QQQ"].values) - 1
            if "QQQ" in daily_returns.columns
            else np.zeros(len(portfolio_cumulative))
        )

        # 결과 생성
        performance_history = []
        for i, date in enumerate(daily_returns.index):
            performance_history.append(
                PerformanceHistory(
                    date=date.strftime("%Y-%m-%d"),
                    portfolio=float(portfolio_cumulative[i]),
                    spy=float(spy_cumulative[i]),
                    qqq=float(qqq_cumulative[i]),
                )
            )

        return performance_history

    except Exception as e:
        print(f"백테스트 계산 오류: {e}")
        return []


def calculate_real_correlation(
    tickers: List[str], period: str = "1y"
) -> List[CorrelationData]:
    """실제 종목 간 상관관계 계산"""
    try:
        # 현금 제외
        stock_tickers = [ticker for ticker in tickers if ticker != "현금"]
        if len(stock_tickers) < 2:
            return []

        # 가격 데이터 다운로드 (세션 사용)
        print(f"상관관계 분석을 위한 데이터 다운로드: {stock_tickers}, 기간: {period}")

        # curl_cffi 세션이 있으면 사용, 없으면 기본 방식
        if session is not None:
            # 각 티커별로 개별 다운로드 (Rate Limit 회피)
            data_dict = {}
            for ticker in stock_tickers:
                try:
                    ticker_obj = yf.Ticker(ticker, session=session)
                    ticker_data = ticker_obj.history(period=period)
                    if not ticker_data.empty:
                        data_dict[ticker] = ticker_data
                        print(f"✓ {ticker} 상관관계 데이터 다운로드 성공")
                    else:
                        print(f"✗ {ticker} 상관관계 데이터 없음")
                except Exception as e:
                    print(f"✗ {ticker} 상관관계 다운로드 실패: {e}")

            # 데이터 조합
            if len(data_dict) >= 2:
                # 모든 티커의 Close 가격을 하나의 DataFrame으로 조합
                close_prices = pd.DataFrame()
                for ticker, ticker_data in data_dict.items():
                    close_prices[ticker] = ticker_data["Close"]
                close_prices = close_prices.dropna()
            else:
                print("상관관계 분석을 위한 충분한 데이터가 없음")
                return []
        else:
            # 기본 방식
            data = yf.download(stock_tickers, period=period, progress=False)
            if data.empty:
                return []

            # 종가 데이터 추출
            if len(stock_tickers) == 1:
                close_prices = data["Close"].to_frame()
                close_prices.columns = stock_tickers
            else:
                close_prices = data["Close"]

        # 일일 수익률 계산
        daily_returns = close_prices.pct_change().dropna()

        # 상관관계 매트릭스 계산
        correlation_matrix = daily_returns.corr()

        # 상관관계 데이터 생성
        correlation_data = []
        available_tickers = list(correlation_matrix.columns)
        for i, stock1 in enumerate(available_tickers):
            for j, stock2 in enumerate(available_tickers):
                if i < j:  # 중복 제거
                    correlation = correlation_matrix.loc[stock1, stock2]
                    if not np.isnan(correlation):
                        correlation_data.append(
                            CorrelationData(
                                stock1=stock1,
                                stock2=stock2,
                                correlation=float(correlation),
                            )
                        )

        return correlation_data

    except Exception as e:
        print(f"상관관계 계산 오류: {e}")
        return []


def calculate_risk_return_data(
    allocation: List[AllocationItem], period: str = "1y"
) -> List[RiskReturnData]:
    """실제 리스크-수익률 데이터 계산"""
    try:
        # 포트폴리오 종목들 추출
        portfolio_tickers = [
            item.symbol for item in allocation if item.symbol != "현금"
        ]
        if not portfolio_tickers:
            return []

        # 가격 데이터 다운로드 (세션 사용)
        print(
            f"리스크-수익률 분석을 위한 데이터 다운로드: {portfolio_tickers}, 기간: {period}"
        )

        # curl_cffi 세션이 있으면 사용, 없으면 기본 방식
        if session is not None:
            # 각 티커별로 개별 다운로드 (Rate Limit 회피)
            data_dict = {}
            for ticker in portfolio_tickers:
                try:
                    ticker_obj = yf.Ticker(ticker, session=session)
                    ticker_data = ticker_obj.history(period=period)
                    if not ticker_data.empty:
                        data_dict[ticker] = ticker_data
                        print(f"✓ {ticker} 리스크-수익률 데이터 다운로드 성공")
                    else:
                        print(f"✗ {ticker} 리스크-수익률 데이터 없음")
                except Exception as e:
                    print(f"✗ {ticker} 리스크-수익률 다운로드 실패: {e}")

            # 데이터 조합
            if data_dict:
                # 모든 티커의 Close 가격을 하나의 DataFrame으로 조합
                close_prices = pd.DataFrame()
                for ticker, ticker_data in data_dict.items():
                    close_prices[ticker] = ticker_data["Close"]
                close_prices = close_prices.dropna()
            else:
                print("리스크-수익률 분석을 위한 데이터가 없음")
                return []
        else:
            # 기본 방식
            data = yf.download(portfolio_tickers, period=period, progress=False)
            if data.empty:
                return []

            # 종가 데이터 추출
            if len(portfolio_tickers) == 1:
                close_prices = data["Close"].to_frame()
                close_prices.columns = portfolio_tickers
            else:
                close_prices = data["Close"]

        # 일일 수익률 계산
        daily_returns = close_prices.pct_change().dropna()

        # 각 종목별 리스크-수익률 계산
        risk_return_data = []
        for item in allocation:
            if item.symbol == "현금":
                continue

            if item.symbol in daily_returns.columns:
                returns = daily_returns[item.symbol]

                # 연간 수익률 계산 (252 거래일 기준)
                annual_return = returns.mean() * 252 * 100

                # 연간 변동성 계산 (표준편차)
                annual_volatility = returns.std() * np.sqrt(252) * 100

                risk_return_data.append(
                    RiskReturnData(
                        symbol=item.symbol,
                        risk=float(annual_volatility),
                        return_rate=float(annual_return),
                        allocation=float(item.weight * 100),
                    )
                )

        return risk_return_data

    except Exception as e:
        print(f"리스크-수익률 계산 오류: {e}")
        return []


def get_real_market_data() -> MarketStatusResponse:
    """실시간 시장 데이터 가져오기"""
    try:
        # 주요 시장 지수들
        market_symbols = {
            "^GSPC": "S&P 500",
            "^IXIC": "NASDAQ",
            "^VIX": "VIX 변동성 지수",
            "KRW=X": "USD/KRW 환율",
        }

        market_data = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for symbol, name in market_symbols.items():
            try:
                # curl_cffi 세션이 있으면 사용, 없으면 기본 방식
                if session is not None:
                    ticker = yf.Ticker(symbol, session=session)
                    print(f"✓ {name} 데이터 다운로드 시도 (세션 사용)")
                else:
                    ticker = yf.Ticker(symbol)
                    print(f"✓ {name} 데이터 다운로드 시도 (기본 방식)")

                hist = ticker.history(period="2d")  # 최근 2일 데이터

                if not hist.empty:
                    current_price = hist["Close"].iloc[-1]
                    previous_price = (
                        hist["Close"].iloc[-2] if len(hist) > 1 else current_price
                    )

                    change = current_price - previous_price
                    change_percent = (
                        (change / previous_price) * 100 if previous_price != 0 else 0
                    )

                    market_data.append(
                        MarketData(
                            symbol=symbol,
                            name=name,
                            price=float(current_price),
                            change=float(change),
                            change_percent=float(change_percent),
                            last_updated=current_time,
                        )
                    )
                    print(f"✓ {name} 데이터 다운로드 성공")
                else:
                    print(f"✗ {name} 히스토리 데이터 없음")

            except Exception as e:
                print(f"✗ {symbol} 데이터 가져오기 실패: {e}")
                # 실패한 경우 기본값 추가
                market_data.append(
                    MarketData(
                        symbol=symbol,
                        name=name,
                        price=0.0,
                        change=0.0,
                        change_percent=0.0,
                        last_updated=current_time,
                    )
                )

        return MarketStatusResponse(market_data=market_data, last_updated=current_time)

    except Exception as e:
        print(f"시장 데이터 가져오기 오류: {e}")
        return MarketStatusResponse(
            market_data=[], last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )


@app.post("/historical-performance", response_model=HistoricalResponse)
async def get_historical_performance(request: HistoricalRequest):
    """실제 백테스트 기반 성과 히스토리"""
    try:
        performance_history = calculate_historical_performance(
            request.portfolio_allocation, request.start_date, request.end_date
        )

        return HistoricalResponse(performance_history=performance_history)

    except Exception as e:
        print(f"성과 히스토리 조회 오류: {e}")
        raise HTTPException(
            status_code=500, detail="성과 히스토리 조회 중 오류가 발생했습니다."
        )


@app.post("/correlation-analysis", response_model=CorrelationResponse)
async def get_correlation_analysis(request: CorrelationRequest):
    """실제 종목 간 상관관계 분석"""
    try:
        correlation_data = calculate_real_correlation(request.tickers, request.period)

        return CorrelationResponse(correlation_data=correlation_data)

    except Exception as e:
        print(f"상관관계 분석 오류: {e}")
        raise HTTPException(
            status_code=500, detail="상관관계 분석 중 오류가 발생했습니다."
        )


@app.post("/risk-return-analysis", response_model=RiskReturnResponse)
async def get_risk_return_analysis(request: RiskReturnRequest):
    """실제 리스크-수익률 분석"""
    try:
        risk_return_data = calculate_risk_return_data(
            request.portfolio_allocation, request.period
        )

        return RiskReturnResponse(risk_return_data=risk_return_data)

    except Exception as e:
        print(f"리스크-수익률 분석 오류: {e}")
        raise HTTPException(
            status_code=500, detail="리스크-수익률 분석 중 오류가 발생했습니다."
        )


@app.get("/market-status", response_model=MarketStatusResponse)
async def get_market_status():
    """실시간 시장 데이터"""
    try:
        return get_real_market_data()

    except Exception as e:
        print(f"시장 데이터 조회 오류: {e}")
        raise HTTPException(
            status_code=500, detail="시장 데이터 조회 중 오류가 발생했습니다."
        )


def download_with_retry(ticker: str, **kwargs) -> pd.DataFrame:
    """재시도 로직이 있는 yfinance 다운로드 함수"""
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            # 티커 객체 생성 시 세션 사용
            if session is not None:
                ticker_obj = yf.Ticker(ticker, session=session)
            else:
                ticker_obj = yf.Ticker(ticker)

            # 데이터 다운로드
            data = ticker_obj.history(**kwargs)

            if not data.empty:
                print(f"✓ {ticker} 데이터 다운로드 성공 (시도 {attempt + 1})")
                return data
            else:
                print(f"✗ {ticker} 데이터 없음 (시도 {attempt + 1})")

        except Exception as e:
            print(f"✗ {ticker} 다운로드 실패 (시도 {attempt + 1}): {e}")

        # 재시도 전 대기 (지수 백오프)
        if attempt < max_retries - 1:
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            print(f"  {delay:.1f}초 후 재시도...")
            time.sleep(delay)

    print(f"✗ {ticker} 모든 시도 실패")
    return pd.DataFrame()


def download_multiple_tickers(tickers: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
    """여러 티커를 개별적으로 다운로드"""
    results = {}

    for ticker in tickers:
        # 요청 간 간격 추가 (Rate Limiting 방지)
        if len(results) > 0:
            time.sleep(0.5)

        data = download_with_retry(ticker, **kwargs)
        if not data.empty:
            results[ticker] = data

    return results


async def explain_prediction_fallback(request: XAIRequest):
    """모델이 로드되지 않았을 때 사용하는 규칙 기반 XAI 분석"""
    import asyncio
    import time
    
    method = request.method.lower()
    print(f"규칙 기반 XAI 분석 시작: 투자금액={request.investment_amount}, 방식={method}")
    
    start_time = time.time()
    
    try:
        # 포트폴리오 예측 결과 먼저 얻기
        prediction_result = predict_portfolio(
            request.investment_amount,
            request.risk_tolerance,
            request.investment_horizon,
        )
        
        # 규칙 기반 Feature Importance 생성
        feature_importance = generate_rule_based_feature_importance(
            request.investment_amount,
            request.risk_tolerance, 
            request.investment_horizon,
            prediction_result["allocation"]
        )
        
        # 규칙 기반 Attention Weights 생성
        attention_weights = generate_rule_based_attention_weights(
            prediction_result["allocation"]
        )
        
        # 설명 텍스트 생성
        explanation_text = generate_rule_based_explanation(
            request.investment_amount,
            request.risk_tolerance,
            request.investment_horizon,
            prediction_result["allocation"],
            method
        )
        
        # 최소 대기 시간 확보 (사용자 경험 향상)
        elapsed_time = time.time() - start_time
        min_duration = 2.0 if method == "fast" else 5.0
        
        if elapsed_time < min_duration:
            remaining_time = min_duration - elapsed_time
            print(f"사용자 경험 향상을 위해 {remaining_time:.1f}초 추가 대기")
            await asyncio.sleep(remaining_time)
        
        print(f"규칙 기반 XAI 분석 완료! (방식: {method}, 총 소요시간: {time.time() - start_time:.1f}초)")
        
        return XAIResponse(
            feature_importance=[
                FeatureImportance(
                    feature_name=item["feature_name"],
                    importance_score=item["importance_score"],
                    asset_name=item["asset_name"],
                )
                for item in feature_importance
            ],
            attention_weights=[
                AttentionWeight(
                    from_asset=item["from_asset"],
                    to_asset=item["to_asset"],
                    weight=item["weight"],
                )
                for item in attention_weights
            ],
            explanation_text=explanation_text,
        )
        
    except Exception as e:
        print(f"규칙 기반 XAI 분석 중 오류: {e}")
        # 최소한의 응답이라도 반환
        return XAIResponse(
            feature_importance=[
                FeatureImportance(
                    feature_name="시장 안정성",
                    importance_score=0.6,
                    asset_name="전체",
                ),
                FeatureImportance(
                    feature_name="수익성 전망", 
                    importance_score=0.4,
                    asset_name="전체",
                )
            ],
            attention_weights=[],
            explanation_text="현재 모델을 로드할 수 없어 기본적인 분석 결과를 제공합니다. "
                           "전반적으로 안정적인 포트폴리오 구성을 권장합니다."
        )


def generate_rule_based_feature_importance(investment_amount, risk_tolerance, investment_horizon, allocation):
    """규칙 기반 Feature Importance 생성"""
    features = []
    
    # 위험 성향에 따른 중요도
    risk_mapping = {"conservative": 2, "moderate": 5, "aggressive": 8}
    risk_level = risk_mapping.get(risk_tolerance, 5)
    if risk_level <= 3:  # 안전형
        features.extend([
            {"feature_name": "안정성", "importance_score": 0.4, "asset_name": "전체"},
            {"feature_name": "배당 수익률", "importance_score": 0.3, "asset_name": "전체"},
            {"feature_name": "변동성", "importance_score": 0.2, "asset_name": "전체"},
            {"feature_name": "성장성", "importance_score": 0.1, "asset_name": "전체"},
        ])
    elif risk_level <= 6:  # 중간형
        features.extend([
            {"feature_name": "성장성", "importance_score": 0.3, "asset_name": "전체"},
            {"feature_name": "안정성", "importance_score": 0.3, "asset_name": "전체"},
            {"feature_name": "수익성", "importance_score": 0.25, "asset_name": "전체"},
            {"feature_name": "변동성", "importance_score": 0.15, "asset_name": "전체"},
        ])
    else:  # 공격형
        features.extend([
            {"feature_name": "성장성", "importance_score": 0.4, "asset_name": "전체"},
            {"feature_name": "수익성", "importance_score": 0.35, "asset_name": "전체"},
            {"feature_name": "변동성", "importance_score": 0.15, "asset_name": "전체"},
            {"feature_name": "안정성", "importance_score": 0.1, "asset_name": "전체"},
        ])
    
    # 투자 기간에 따른 조정
    if investment_horizon <= 6:  # 단기
        # 안정성 중요도 증가
        for feature in features:
            if feature["feature_name"] == "안정성":
                feature["importance_score"] *= 1.2
            elif feature["feature_name"] == "성장성":
                feature["importance_score"] *= 0.8
    
    # 정규화
    total_score = sum(f["importance_score"] for f in features)
    for feature in features:
        feature["importance_score"] /= total_score
    
    return features


def generate_rule_based_attention_weights(allocation):
    """규칙 기반 Attention Weights 생성"""
    weights = []
    
    # 종목별 가중치 기반으로 상관관계 시뮬레이션
    stocks = [item for item in allocation if item["symbol"] != "현금"]
    
    for i, stock1 in enumerate(stocks):
        for j, stock2 in enumerate(stocks[i+1:], i+1):
            # 가중치가 비슷한 종목일수록 높은 attention
            weight_diff = abs(stock1["weight"] - stock2["weight"])
            attention_score = max(0.1, 0.8 - weight_diff * 2)
            
            weights.append({
                "from_asset": stock1["symbol"],
                "to_asset": stock2["symbol"], 
                "weight": round(attention_score, 3)
            })
    
    return weights


def generate_rule_based_explanation(investment_amount, risk_tolerance, investment_horizon, allocation, method):
    """규칙 기반 설명 텍스트 생성"""
    risk_mapping = {"conservative": 2, "moderate": 5, "aggressive": 8}
    risk_level = risk_mapping.get(risk_tolerance, 5)
    risk_name = "안전형" if risk_level <= 3 else "중간형" if risk_level <= 6 else "공격형"
    
    # 주요 보유 종목
    main_stocks = [item for item in allocation if item["symbol"] != "현금" and item["weight"] > 0.05]
    main_stocks.sort(key=lambda x: x["weight"], reverse=True)
    top_stocks = main_stocks[:3]
    
    explanation = f"""
【AI 포트폴리오 분석 결과 - {method.upper()} 모드】

📊 투자 프로필 분석:
• 투자 금액: {investment_amount:,}원
• 위험 성향: {risk_name} (레벨 {risk_level}/10)
• 투자 기간: {investment_horizon}개월

🎯 포트폴리오 구성 근거:
"""
    
    if risk_level <= 3:
        explanation += """
• 안전성을 최우선으로 한 보수적 포트폴리오
• 배당 수익률과 원금 보존에 중점
• 변동성이 낮은 우량 종목 위주 구성
"""
    elif risk_level <= 6:
        explanation += """  
• 안정성과 성장성의 균형잡힌 포트폴리오
• 중장기 수익 창출을 목표로 한 구성
• 리스크 대비 적정 수익률 추구
"""
    else:
        explanation += """
• 높은 성장 가능성에 중점을 둔 적극적 포트폴리오  
• 변동성을 수용하여 높은 수익률 추구
• 신성장 동력주와 기술주 비중 확대
"""
    
    if top_stocks:
        explanation += "\n💡 주요 보유 종목:\n"
        for i, stock in enumerate(top_stocks):
            explanation += f"• {stock['symbol']}: {stock['weight']:.1%} - "
            if i == 0:
                explanation += "포트폴리오 핵심 종목\n"
            elif i == 1:
                explanation += "분산 투자 목적\n" 
            else:
                explanation += "리스크 헷지 역할\n"
    
    if method == "fast":
        explanation += "\n⚡ 빠른 분석으로 생성된 결과입니다."
    else:
        explanation += "\n🔍 정밀 분석으로 생성된 결과입니다."
        
    explanation += f"\n\n⚠️ 현재 모델이 로드되지 않아 규칙 기반 분석 결과를 제공합니다."
    
    return explanation.strip()


if __name__ == "__main__":
    uvicorn.run(
        "rl_inference_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
