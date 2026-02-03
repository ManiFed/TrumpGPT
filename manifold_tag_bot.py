#!/usr/bin/env python3
"""Manifold Markets tag-reply bot.

Continuously polls Manifold comments, looks for @TrumpGPT mentions, and replies
using OpenRouter's chat completion API.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Set

MANIFOLD_BASE_URL = os.getenv("MANIFOLD_BASE_URL", "https://api.manifold.markets/v0")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MANIFOLD_API_KEY = os.getenv("MANIFOLD_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MENTION_TAG = os.getenv("MENTION_TAG")
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
STATE_PATH = os.getenv("STATE_PATH", ".manifold_bot_state.json")
COMMENT_LIMIT = int(os.getenv("COMMENT_LIMIT", "50"))


def _request_json(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Any:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else None


def load_state(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        logging.warning("State file unreadable, starting fresh: %s", path)
        return set()
    return set(data.get("processed_comment_ids", []))


def save_state(path: str, processed_ids: Iterable[str]) -> None:
    payload = {"processed_comment_ids": list(processed_ids)}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def fetch_recent_comments(limit: int) -> List[Dict[str, Any]]:
    query_params = {"limit": str(limit)}
    if MANIFOLD_CONTRACT_ID:
        query_params["contractId"] = MANIFOLD_CONTRACT_ID
    elif MANIFOLD_USER_ID:
        query_params["userId"] = MANIFOLD_USER_ID
    query = urllib.parse.urlencode(query_params)
    url = f"{MANIFOLD_BASE_URL}/comments?{query}"
    return _request_json("GET", url) or []


def build_openrouter_reply(comment_text: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant responding to a Manifold Markets comment. "
                    "Answer the user's question directly and concisely."
                ),
            },
            {
                "role": "user",
                "content": (
                    "The following Manifold comment mentioned @TrumpGPT. "
                    "Reply to the question in that comment.\n\n"
                    f"Comment: {comment_text}"
                ),
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Accept": "application/json",
    }
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    response = _request_json("POST", url, headers=headers, payload=payload)
    choices = response.get("choices") if isinstance(response, dict) else None
    if not choices:
        raise RuntimeError("OpenRouter response missing choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        raise RuntimeError("OpenRouter response missing content")
    return content.strip()


def post_reply(comment: Dict[str, Any], reply_text: str) -> None:
    if not MANIFOLD_API_KEY:
        raise RuntimeError("MANIFOLD_API_KEY is required")
    contract_id = comment.get("contractId")
    comment_id = comment.get("id")
    if not contract_id or not comment_id:
        raise RuntimeError("Comment missing contractId or id")
    payload = {
        "contractId": contract_id,
        "content": reply_text,
        "replyToCommentId": comment_id,
    }
    headers = {
        "Authorization": f"Key {MANIFOLD_API_KEY}",
        "Accept": "application/json",
    }
    url = f"{MANIFOLD_BASE_URL}/comment"
    _request_json("POST", url, headers=headers, payload=payload)


def should_reply(comment: Dict[str, Any]) -> bool:
    text = comment.get("text") or ""
    if MENTION_TAG not in text:
        return False
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    processed_ids = load_state(STATE_PATH)
    logging.info("Loaded %d processed comment IDs", len(processed_ids))

    while True:
        try:
            comments = fetch_recent_comments(COMMENT_LIMIT)
            comments_sorted = sorted(
                comments, key=lambda c: c.get("createdTime", 0), reverse=True
            )
            for comment in comments_sorted:
                comment_id = comment.get("id")
                if not comment_id or comment_id in processed_ids:
                    continue
                if not should_reply(comment):
                    processed_ids.add(comment_id)
                    continue
                comment_text = comment.get("text", "")
                logging.info("Replying to comment %s", comment_id)
                reply = build_openrouter_reply(comment_text)
                post_reply(comment, reply)
                processed_ids.add(comment_id)
                save_state(STATE_PATH, processed_ids)
            time.sleep(POLL_INTERVAL_SECONDS)
        except urllib.error.HTTPError as exc:
            logging.error("HTTP error %s: %s", exc.code, exc.read().decode("utf-8"))
            time.sleep(POLL_INTERVAL_SECONDS)
        except Exception as exc:  # noqa: BLE001 - top-level loop
            logging.exception("Unexpected error: %s", exc)
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
