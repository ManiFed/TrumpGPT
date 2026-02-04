#!/usr/bin/env python3
"""Manifold Markets spread-closing bot.

Scans Manifold markets for wide bid/ask spreads and places paired limit orders
using a Kelly-criterion sizing strategy.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

MANIFOLD_BASE_URL = os.getenv("MANIFOLD_BASE_URL", "https://api.manifold.markets/v0")
MANIFOLD_API_KEY = os.getenv("MANIFOLD_API_KEY")
STATE_PATH = os.getenv("STATE_PATH", ".manifold_spread_state.json")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
MARKET_BATCH_LIMIT = int(os.getenv("MARKET_BATCH_LIMIT", "200"))
MAX_MARKETS_PER_LOOP = int(os.getenv("MAX_MARKETS_PER_LOOP", "100"))
MIN_SPREAD = float(os.getenv("MIN_SPREAD", "0.02"))
SPREAD_TICK = float(os.getenv("SPREAD_TICK", "0.002"))
BANKROLL = float(os.getenv("BANKROLL", "0"))
MIN_TRADE = float(os.getenv("MIN_TRADE", "1"))
MAX_TRADE = float(os.getenv("MAX_TRADE", "50"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "600"))
DRY_RUN = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class MarketQuote:
    contract_id: str
    question: str
    yes_bid: float
    yes_ask: float
    volume: float


@dataclass
class BotState:
    last_order_times: Dict[str, float]


class ManifoldAPI:
    def __init__(self, api_key: Optional[str]) -> None:
        if not api_key:
            raise RuntimeError("MANIFOLD_API_KEY is required")
        self.api_key = api_key

    def _request_json(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Key {self.api_key}")
        req.add_header("Accept", "application/json")
        if payload is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else None

    def fetch_markets(self, limit: int, before: Optional[int] = None) -> List[Dict[str, Any]]:
        query_params = {"limit": str(limit)}
        if before is not None:
            query_params["before"] = str(before)
        query_string = "&".join(f"{key}={value}" for key, value in query_params.items())
        url = f"{MANIFOLD_BASE_URL}/markets?{query_string}"
        response = self._request_json("GET", url)
        if not isinstance(response, list):
            raise RuntimeError("Manifold markets response missing list payload")
        return response

    def fetch_bets(self, contract_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        query_string = f"contractId={contract_id}&limit={limit}"
        url = f"{MANIFOLD_BASE_URL}/bets?{query_string}"
        response = self._request_json("GET", url)
        if not isinstance(response, list):
            raise RuntimeError("Manifold bets response missing list payload")
        return response

    def fetch_me(self) -> Dict[str, Any]:
        url = f"{MANIFOLD_BASE_URL}/me"
        response = self._request_json("GET", url)
        if not isinstance(response, dict):
            raise RuntimeError("Manifold /me response missing payload")
        return response

    def place_bet(
        self, contract_id: str, outcome: str, amount: float, limit_prob: float
    ) -> Dict[str, Any]:
        payload = {
            "contractId": contract_id,
            "outcome": outcome,
            "amount": amount,
            "limitProb": limit_prob,
        }
        url = f"{MANIFOLD_BASE_URL}/bet"
        response = self._request_json("POST", url, payload=payload)
        if not isinstance(response, dict):
            raise RuntimeError("Manifold bet response missing payload")
        return response


def load_state(path: str) -> BotState:
    if not os.path.exists(path):
        return BotState(last_order_times={})
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        logging.warning("State file unreadable, starting fresh: %s", path)
        return BotState(last_order_times={})
    last_order_times = data.get("last_order_times", {})
    if not isinstance(last_order_times, dict):
        last_order_times = {}
    return BotState(last_order_times=last_order_times)


def save_state(path: str, state: BotState) -> None:
    payload = {"last_order_times": state.last_order_times}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def is_limit_order_open(bet: Dict[str, Any]) -> bool:
    if bet.get("isCancelled"):
        return False
    if bet.get("isFilled") is True:
        return False
    return bet.get("limitProb") is not None


def extract_best_quotes(bets: Iterable[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    yes_bids: List[float] = []
    yes_asks: List[float] = []
    for bet in bets:
        if not is_limit_order_open(bet):
            continue
        outcome = bet.get("outcome")
        limit_prob = bet.get("limitProb")
        if outcome not in {"YES", "NO"}:
            continue
        if not isinstance(limit_prob, (float, int)):
            continue
        yes_price = float(limit_prob) if outcome == "YES" else 1.0 - float(limit_prob)
        if outcome == "YES":
            yes_bids.append(yes_price)
        else:
            yes_asks.append(yes_price)
    if not yes_bids or not yes_asks:
        return None
    return max(yes_bids), min(yes_asks)


def estimate_bankroll(api: ManifoldAPI) -> float:
    if BANKROLL > 0:
        return BANKROLL
    profile = api.fetch_me()
    balance = profile.get("balance")
    if isinstance(balance, (float, int)):
        return float(balance)
    raise RuntimeError("Unable to determine bankroll; set BANKROLL env var")


def kelly_fraction(true_prob: float, price_prob: float) -> float:
    if price_prob <= 0 or price_prob >= 1:
        return 0.0
    edge = true_prob - price_prob
    if edge <= 0:
        return 0.0
    return min(edge / (1 - price_prob), 1.0)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def choose_markets(api: ManifoldAPI) -> List[MarketQuote]:
    markets: List[MarketQuote] = []
    before: Optional[int] = None
    while len(markets) < MAX_MARKETS_PER_LOOP:
        batch = api.fetch_markets(MARKET_BATCH_LIMIT, before=before)
        if not batch:
            break
        for market in batch:
            if market.get("isResolved") or market.get("isClosed"):
                continue
            if market.get("outcomeType") != "BINARY":
                continue
            contract_id = market.get("id")
            question = market.get("question", "")
            volume = float(market.get("volume", 0.0))
            if not contract_id:
                continue
            bets = api.fetch_bets(contract_id)
            quotes = extract_best_quotes(bets)
            if not quotes:
                continue
            yes_bid, yes_ask = quotes
            if yes_ask <= yes_bid:
                continue
            spread = yes_ask - yes_bid
            if spread < MIN_SPREAD:
                continue
            markets.append(
                MarketQuote(
                    contract_id=contract_id,
                    question=question,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    volume=volume,
                )
            )
            if len(markets) >= MAX_MARKETS_PER_LOOP:
                break
        before = batch[-1].get("createdTime")
        if before is None:
            break
    markets.sort(key=lambda quote: (quote.yes_ask - quote.yes_bid, quote.volume), reverse=True)
    return markets


def maybe_place_orders(api: ManifoldAPI, quote: MarketQuote, bankroll: float, state: BotState) -> None:
    last_order_time = state.last_order_times.get(quote.contract_id, 0)
    if time.time() - last_order_time < COOLDOWN_SECONDS:
        return
    mid = (quote.yes_bid + quote.yes_ask) / 2
    yes_price = clamp(mid + SPREAD_TICK, 0.01, 0.99)
    no_price = clamp(1 - mid + SPREAD_TICK, 0.01, 0.99)

    yes_fraction = kelly_fraction(mid, yes_price)
    no_fraction = kelly_fraction(1 - mid, no_price)

    yes_amount = clamp(bankroll * yes_fraction, MIN_TRADE, MAX_TRADE)
    no_amount = clamp(bankroll * no_fraction, MIN_TRADE, MAX_TRADE)

    if yes_fraction == 0 and no_fraction == 0:
        return

    logging.info(
        "Placing orders on %s (spread %.3f) YES @ %.3f ($%.2f) NO @ %.3f ($%.2f)",
        quote.contract_id,
        quote.yes_ask - quote.yes_bid,
        yes_price,
        yes_amount,
        no_price,
        no_amount,
    )

    if DRY_RUN:
        state.last_order_times[quote.contract_id] = time.time()
        return

    if yes_fraction > 0:
        api.place_bet(quote.contract_id, "YES", yes_amount, yes_price)
    if no_fraction > 0:
        api.place_bet(quote.contract_id, "NO", no_amount, 1 - no_price)
    state.last_order_times[quote.contract_id] = time.time()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    api = ManifoldAPI(MANIFOLD_API_KEY)
    state = load_state(STATE_PATH)
    bankroll = estimate_bankroll(api)
    logging.info("Starting spread bot with bankroll %.2f", bankroll)

    while True:
        try:
            markets = choose_markets(api)
            logging.info("Found %d markets with spreads", len(markets))
            for quote in markets:
                maybe_place_orders(api, quote, bankroll, state)
            save_state(STATE_PATH, state)
        except urllib.error.HTTPError as exc:
            logging.error("HTTP error %s: %s", exc.code, exc.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - keep running
            logging.exception("Unexpected error in loop: %s", exc)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
