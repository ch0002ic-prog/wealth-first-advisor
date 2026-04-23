from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any
from urllib import request

import pandas as pd


@dataclass(frozen=True)
class TargetOrder:
    symbol: str
    current_weight: float
    target_weight: float
    delta_weight: float
    current_notional: float
    target_notional: float
    delta_notional: float


@dataclass(frozen=True)
class ExecutionPlan:
    equity: float
    turnover: float
    orders: list[TargetOrder]
    cash_symbol: str = "CASH"
    current_cash_weight: float = 0.0
    target_cash_weight: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "equity": self.equity,
            "turnover": self.turnover,
            "cash_symbol": self.cash_symbol,
            "current_cash_weight": self.current_cash_weight,
            "target_cash_weight": self.target_cash_weight,
            "orders": [asdict(order) for order in self.orders],
        }


def _normalize_weights(weights: pd.Series) -> pd.Series:
    normalized = weights.fillna(0.0).astype(float)
    total = float(normalized.sum())
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value.")
    return normalized / total


def build_execution_plan(
    target_weights: pd.Series,
    current_weights: pd.Series | None = None,
    equity: float = 1.0,
    min_trade_weight: float = 0.0,
    cash_symbol: str = "CASH",
) -> ExecutionPlan:
    normalized_target = _normalize_weights(target_weights)
    if current_weights is None:
        normalized_current = pd.Series(0.0, index=normalized_target.index)
    else:
        normalized_current = _normalize_weights(current_weights)

    all_symbols = normalized_target.index.union(normalized_current.index)
    normalized_target = normalized_target.reindex(all_symbols).fillna(0.0)
    normalized_current = normalized_current.reindex(all_symbols).fillna(0.0)

    delta_weights = normalized_target - normalized_current
    tradable_delta_weights = delta_weights.drop(labels=[cash_symbol], errors="ignore")
    turnover = float(tradable_delta_weights.abs().sum())
    orders: list[TargetOrder] = []

    for symbol, target_weight in normalized_target.items():
        if symbol == cash_symbol:
            continue

        current_weight = float(normalized_current.get(symbol, 0.0))
        delta_weight = float(delta_weights[symbol])
        if abs(delta_weight) < min_trade_weight:
            continue

        orders.append(
            TargetOrder(
                symbol=str(symbol),
                current_weight=current_weight,
                target_weight=float(target_weight),
                delta_weight=delta_weight,
                current_notional=current_weight * equity,
                target_notional=float(target_weight) * equity,
                delta_notional=delta_weight * equity,
            )
        )

    orders.sort(key=lambda order: abs(order.delta_weight), reverse=True)
    return ExecutionPlan(
        equity=equity,
        turnover=turnover,
        orders=orders,
        cash_symbol=cash_symbol,
        current_cash_weight=float(normalized_current.get(cash_symbol, 0.0)),
        target_cash_weight=float(normalized_target.get(cash_symbol, 0.0)),
    )


class ExchangeAdapter(ABC):
    @abstractmethod
    def submit_allocation(self, plan: ExecutionPlan) -> dict[str, Any]:
        raise NotImplementedError


class PaperExchangeAdapter(ExchangeAdapter):
    def submit_allocation(self, plan: ExecutionPlan) -> dict[str, Any]:
        return {
            "adapter": "paper",
            "status": "accepted",
            **plan.to_payload(),
        }


class WebhookExchangeAdapter(ExchangeAdapter):
    def __init__(self, url: str, token: str | None = None, timeout: float = 10.0) -> None:
        self.url = url
        self.token = token
        self.timeout = timeout

    def submit_allocation(self, plan: ExecutionPlan) -> dict[str, Any]:
        payload = json.dumps(plan.to_payload()).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        http_request = request.Request(self.url, data=payload, headers=headers, method="POST")
        with request.urlopen(http_request, timeout=self.timeout) as response:
            response_body = response.read().decode("utf-8")

        return {
            "adapter": "webhook",
            "status": "submitted",
            "response": response_body,
        }