"""Resumed one-shot turns: don't repeat an unchanged leading instruction block already verbatim in context.

An orchestrator that resumes one session per wake (Paperclip's Hermes adapter) prepends the agent's full managed
instructions to every wake message. On a resumed session the identical block is usually still in context, so each
wake appends another copy (MER-646: 147 of 264 wakes in a day, 1.7M characters).

Opt-in per process by the orchestrator, through two environment variables naming the exact block it prepended:

* ``HERMES_RESUME_UNCHANGED_PREFIX_SHA256``: sha256 (hex) of the block's exact characters;
* ``HERMES_RESUME_UNCHANGED_PREFIX_CHARS``: the block's length in characters.

Both are consumed on first read, so only the first turn of the process can use them. The block is replaced by a
short reference only when ALL of these hold; otherwise the message is sent unchanged:

* the new message starts with exactly that block (length and sha256 match);
* the turn resumes a session (non-empty history) that shows no compaction: no summary handoff anywhere in the
  history and the compressor's head protection has not decayed;
* the session's first user message is inside the compressor's protected head and starts with the byte-identical
  block (same length, same sha256).

The first user message is then pinned in the protected head for the rest of this agent's life
(``ContextCompressor._pinned_head_rows``), so a compaction inside this turn cannot summarise the copy the reference
points at. A later wake whose session was compacted fails the "no compaction" test and is sent the full block again.
Nothing earlier in the session is edited; the persisted message is exactly what the model receives.

Where it decides (revision 2, Vasco's review of MER-646): ``build_turn_context`` calls :func:`apply_to_turn` on the
turn's FINAL canonical rows, after a rotated session has been recovered/adopted and after turn-start compaction, and
before the user row is stamped or persisted. Deciding on the caller's pre-admission history was unsafe: an
uncontended lease admission keeps the already-loaded parent history, and turn construction can then adopt a
compression child whose head no longer holds the block. Any turn-start compaction, an already persisted user row,
CLI-staged input or separately persisted text keeps the full block.

Wire, not canonical history (revision 3, Vasco's re-review): the request builder canonicalizes the replayed prefix
(``replay_cleanup.canonicalize_replay_history``) and may redact a whole historical user message, e.g. an expired
dangerous-confirmation phrase anywhere in the first wake. So:

* the decision also projects the history through that same cleanup, with the turn's frozen expiry clock, and the
  replayed bytes (``api_content`` sidecar, else content); the first user message must still open with the block
  there (``carrier_not_on_wire`` keeps the full block otherwise);
* :func:`enforce_on_wire` runs on EVERY assembled request of the turn, after all send-time transforms: when the
  request does not carry the block verbatim before the reference, it puts the full block back in place of the
  reference in that request (fail closed) and records ``restored``;
* :func:`report_turn` logs, after the turn, what the requests actually carried (the live acceptance signal):
  ``wire verified N/N`` or a ``RESTORED`` warning. :func:`verify_final_transcript` is only a canonical-history
  diagnostic now.

The cleanup itself is untouched: a redacted carrier stays redacted, and the wake gets its full block instead.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_SHA256 = "HERMES_RESUME_UNCHANGED_PREFIX_SHA256"
ENV_CHARS = "HERMES_RESUME_UNCHANGED_PREFIX_CHARS"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
# Below this a reference saves too little to be worth any risk.
MIN_CHARS = 512


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reference_text(chars: int, sha256: str) -> str:
    return (
        f"[Agent instructions unchanged: this wake's {chars}-character instruction block is byte-identical "
        f"(sha256 {sha256}) to the block that opens this session's first message. That copy stays in force and is "
        f"kept verbatim in context for this turn, so it is not repeated here.]"
    )


def _consume_env() -> Optional[Tuple[str, int]]:
    # A gateway-run finite turn reads the opt-in its client forwarded (gateway.client_env_passthrough);
    # an in-process one-shot reads its own environment. Either way it is consumed once.
    sha = raw_chars = None
    try:
        from gateway.session_context import take_client_env_value
    except Exception:  # health: allow BLE001 -- install without the gateway package
        take_client_env_value = None
    if take_client_env_value is not None:
        sha, raw_chars = take_client_env_value(ENV_SHA256), take_client_env_value(ENV_CHARS)
    if sha is None and raw_chars is None:
        sha = os.environ.pop(ENV_SHA256, None)
        raw_chars = os.environ.pop(ENV_CHARS, None)
    if not sha or not raw_chars:
        return None
    sha = sha.strip().lower()
    try:
        chars = int(raw_chars.strip())
    except ValueError:
        return None
    if not _SHA_RE.match(sha) or chars < MIN_CHARS:
        return None
    return sha, chars


def decide(
    user_message: Any,
    conversation_history: Optional[List[Dict[str, Any]]],
    *,
    sha256: str,
    chars: int,
    protect_first_n: int,
    effective_protect_first_n: int,
    is_summary,
) -> Tuple[Optional[str], Optional[int], str]:
    """Pure decision. Returns ``(new_message, first_user_index, reason)``; ``new_message`` is None to keep."""
    if not isinstance(user_message, str) or len(user_message) < chars:
        return None, None, "message_shorter_than_block"
    if _sha(user_message[:chars]) != sha256:
        return None, None, "message_block_mismatch"
    history = [m for m in (conversation_history or []) if isinstance(m, dict)]
    if not history:
        return None, None, "fresh_session"
    if any(is_summary(m) for m in history):
        return None, None, "compacted_session"
    if protect_first_n <= 0 or effective_protect_first_n != protect_first_n:
        return None, None, "head_protection_decayed"
    rows = [m for m in history if m.get("role") != "system"]
    first = next((i for i, m in enumerate(rows) if m.get("role") == "user"), None)
    if first is None:
        return None, None, "no_first_user_message"
    if first >= protect_first_n:
        return None, None, "first_user_message_unprotected"
    content = rows[first].get("content")
    if not isinstance(content, str) or len(content) < chars or _sha(content[:chars]) != sha256:
        return None, None, "first_user_block_differs"
    # The bytes the model saw for that message (api_content sidecar) must open with the same block too.
    sent = rows[first].get("api_content")
    if isinstance(sent, str) and sent and (len(sent) < chars or _sha(sent[:chars]) != sha256):
        return None, None, "first_user_block_differs"
    return reference_text(chars, sha256) + user_message[chars:], first, "deduplicated"


def wire_prefix_carries_block(agent: Any, history: Optional[List[Dict[str, Any]]], sha256: str, chars: int) -> bool:
    """Will the replayed prefix of this turn's requests still open its first user message with the block?

    Projects ``history`` exactly as ``build_api_messages`` does for the rows before the current turn: the shared
    replay cleanup with the turn's frozen expiry clock, then the replayed bytes (``api_content`` sidecar, else
    content). No clock (a caller that skipped the turn prologue) is not provable, so it is False.
    """
    from agent.replay_cleanup import canonicalize_replay_history

    now = getattr(agent, "_current_turn_timestamp", None)
    if isinstance(now, bool) or not isinstance(now, (int, float)):
        return False
    projected = canonicalize_replay_history([m for m in (history or []) if isinstance(m, dict)], now=float(now))
    first = next((m for m in projected if m.get("role") == "user"), None)
    if first is None:
        return False
    sidecar = first.get("api_content")
    wire = sidecar if isinstance(sidecar, str) and sidecar else first.get("content")
    return isinstance(wire, str) and len(wire) >= chars and _sha(wire[:chars]) == sha256


def _message_texts(message: Dict[str, Any]) -> List[Tuple[Optional[int], str]]:
    """``(part index or None, text)`` for a wire message's string content or its text parts."""
    content = message.get("content")
    if isinstance(content, str):
        return [(None, content)]
    if isinstance(content, list):
        return [(j, p["text"]) for j, p in enumerate(content)
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)]
    return []


def _set_text(message: Dict[str, Any], part: Optional[int], text: str) -> None:
    if part is None:
        message["content"] = text
    else:
        message["content"][part] = {**message["content"][part], "text": text}


def _wire_outcome(api_messages: List[Dict[str, Any]], line: str, block: str) -> Tuple[str, Optional[Tuple[int, Optional[int], int]]]:
    """Classify one assembled request. Returns ``(outcome, location of the reference to restore)``.

    ``verified``: the reference is there and a user message earlier in the request (or earlier in the same text)
    holds the block verbatim. ``missing``: the reference is there without the block before it.
    ``no_reference``: this request does not carry the reference (nothing was omitted from it).
    """
    seen_block = False
    last = None
    for i, message in enumerate(api_messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        for part, text in _message_texts(message):
            at = text.rfind(line)
            if at >= 0:
                last = (i, part, at, seen_block or block in text[:at])
            if block in text:
                seen_block = True
    if last is None:
        return "no_reference", None
    i, part, at, carried = last
    return ("verified", None) if carried else ("missing", (i, part, at))


def enforce_on_wire(agent: Any, api_messages: Any) -> Any:
    """Fail closed on the assembled request: the reference never reaches the provider without the block before it.

    Runs on every request of a turn that sent a reference, after all send-time transforms. When the block is not
    verbatim earlier in the request, the reference in that request is replaced by the full block (the wake's own
    bytes), so the model receives every clause. Records the outcome per request for :func:`report_turn`.
    """
    ref = getattr(agent, "_resume_prefix_reference", None)
    if not ref or not isinstance(api_messages, list):
        return api_messages
    sha256, chars, _sent, block = ref
    line = reference_text(chars, sha256)
    outcomes = getattr(agent, "_resume_prefix_wire", None)
    if not isinstance(outcomes, list):
        outcomes = []
        agent._resume_prefix_wire = outcomes
    try:
        outcome, where = _wire_outcome(api_messages, line, block)
        if outcome == "missing" and where is not None:
            i, part, at = where
            text = dict(_message_texts(api_messages[i]))[part]
            _set_text(api_messages[i], part, text[:at] + block + text[at + len(line):])
            logger.warning("resume-prefix-dedupe: RESTORED full block on the wire: the request did not carry the "
                           "referenced block (sha256=%s)", sha256[:12])
            outcome = "restored"
        outcomes.append(outcome)
    except Exception:
        # Never send the reference unchecked: put the block back wherever the reference appears.
        logger.warning("resume-prefix-dedupe: wire check error; restoring the full block", exc_info=True)
        for message in api_messages:
            if isinstance(message, dict):
                for part, text in _message_texts(message):
                    if line in text:
                        _set_text(message, part, text.replace(line, block))
        outcomes.append("restored")
    return api_messages


def report_turn(agent: Any) -> Optional[str]:
    """After a turn that sent a reference: log what its requests actually carried (the live acceptance signal).

    ``verified`` when every request carried the block verbatim before the reference; ``restored`` (a warning) when
    any request had the block put back; ``unchecked`` (a warning) when no assembled request was checked.
    Returns the summary, or None when this turn sent no reference.
    """
    ref = getattr(agent, "_resume_prefix_reference", None)
    if not ref:
        return None
    sha256 = ref[0]
    outcomes = [o for o in (getattr(agent, "_resume_prefix_wire", None) or []) if isinstance(o, str)]
    referenced = [o for o in outcomes if o != "no_reference"]
    restored = sum(1 for o in referenced if o == "restored")
    if not referenced:
        logger.warning("resume-prefix-dedupe: wire unchecked: no request carried the reference (sha256=%s, "
                       "requests=%d)", sha256[:12], len(outcomes))
        return "unchecked"
    if restored:
        logger.warning("resume-prefix-dedupe: RESTORED full block on the wire in %d/%d requests (sha256=%s)",
                       restored, len(referenced), sha256[:12])
        return "restored"
    logger.info("resume-prefix-dedupe: wire verified block before the reference in %d/%d requests (sha256=%s)",
                len(referenced), len(referenced), sha256[:12])
    return "verified"


def apply_to_turn(
    agent: Any,
    messages: List[Dict[str, Any]],
    current_turn_user_idx: int,
    *,
    persist_user_message: Any = None,
    staged_cli_input: bool = False,
    compressed: bool = False,
    moa_active: bool = False,
) -> Optional[str]:
    """Decide on the turn's final canonical rows; rewrite this turn's user row in place when it qualifies.

    ``messages[:current_turn_user_idx]`` is the history the model will actually receive (post recovery/adoption and
    post turn-start compaction). Returns the new user message, or None when the full message stays. Never raises.
    The opt-in is consumed on every call, so a kept turn never leaves it for a later one.
    """
    opt_in = _consume_env()
    if opt_in is None:
        return None
    sha256, chars = opt_in
    try:
        reason = None
        if compressed:
            reason = "compacted_this_turn"
        elif staged_cli_input:
            reason = "staged_cli_input"
        elif moa_active or getattr(agent, "api_mode", None) == "codex_app_server":
            # These send through a request path the wire check does not cover.
            reason = "unguarded_request_path"
        elif not isinstance(current_turn_user_idx, int) or not (0 <= current_turn_user_idx < len(messages)):
            reason = "no_current_turn_row"
        else:
            row = messages[current_turn_user_idx]
            if not isinstance(row, dict) or row.get("role") != "user":
                reason = "no_current_turn_row"
            elif "_row_id" in row or "api_content" in row:
                reason = "already_persisted"
        if reason is not None:
            logger.info("resume-prefix-dedupe: kept (%s), sha256=%s chars=%d", reason, sha256[:12], chars)
            return None
        row = messages[current_turn_user_idx]
        content = row.get("content")
        # The gateway always hands its persist text through (the clean message, timestamps stripped);
        # when it is the very text being sent there is no separate persisted text to keep. The reference
        # then replaces both, so the transcript holds exactly what the model received.
        same_persist = isinstance(persist_user_message, str) and persist_user_message == content
        new_message = _dedupe(agent, sha256, chars, content, messages[:current_turn_user_idx],
                              None if same_persist else persist_user_message)
        if new_message is None:
            return None
        row["content"] = new_message
        if same_persist:
            agent._persist_user_message_override = new_message
        agent._resume_prefix_reference = (sha256, chars, new_message, str(content)[:chars])
        agent._resume_prefix_wire = []
        return new_message
    except Exception:
        logger.warning("resume-prefix-dedupe: kept (error)", exc_info=True)
        return None


def verify_final_transcript(agent: Any, messages: Any) -> Optional[bool]:
    """Diagnostic: after a turn that sent a reference, is the block still verbatim in the canonical transcript?

    Canonical history is not what the provider received (the request builder may redact a replayed row), so this
    is NOT the acceptance signal; :func:`report_turn` is. Returns True/False; None when no reference was sent.
    """
    ref = getattr(agent, "_resume_prefix_reference", None)
    if not ref:
        return None
    sha256, chars, sent = ref[:3]
    try:
        rows = [m for m in (messages or []) if isinstance(m, dict) and m.get("role") != "system"]
        idx = next((i for i in range(len(rows) - 1, -1, -1)
                    if rows[i].get("role") == "user" and rows[i].get("content") == sent), None)
        prior = rows[:idx] if idx is not None else []
        ok = idx is not None and any(
            m.get("role") == "user" and isinstance(m.get("content"), str) and len(m["content"]) >= chars
            and _sha(m["content"][:chars]) == sha256
            and not (isinstance(m.get("api_content"), str) and m["api_content"]
                     and (len(m["api_content"]) < chars or _sha(m["api_content"][:chars]) != sha256))
            for m in prior
        )
        if ok:
            logger.info("resume-prefix-dedupe: canonical transcript retains the block (sha256=%s)", sha256[:12])
        else:
            logger.warning("resume-prefix-dedupe: canonical transcript lacks the referenced block (sha256=%s)", sha256[:12])
        return ok
    except Exception:
        logger.warning("resume-prefix-dedupe: verification error", exc_info=True)
        return False


def apply_resume_prefix_dedupe(
    agent: Any,
    user_message: Any,
    conversation_history: Optional[List[Dict[str, Any]]],
    persist_user_message: Any = None,
) -> Any:
    """Decision core on an explicit history (tests and replay). Never raises; any doubt returns ``user_message``."""
    opt_in = _consume_env()
    if opt_in is None:
        return user_message
    sha256, chars = opt_in
    try:
        out = _dedupe(agent, sha256, chars, user_message, conversation_history, persist_user_message)
        return user_message if out is None else out
    except Exception:
        logger.warning("resume-prefix-dedupe: kept (error)", exc_info=True)
        return user_message


def _dedupe(agent, sha256, chars, user_message, conversation_history, persist_user_message) -> Optional[str]:
    """Shared core. Returns the referenced message, or None to keep. May raise (callers catch)."""
    if persist_user_message is not None:
        # The caller stores different text from what it sends; keep both exactly as given.
        logger.info("resume-prefix-dedupe: kept (separate persisted text)")
        return None
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        logger.info("resume-prefix-dedupe: kept (no compressor)")
        return None
    from agent.context_compressor import ContextCompressor

    probe = [{"role": "system", "content": ""}] + [m for m in (conversation_history or []) if isinstance(m, dict)]
    new_message, first, reason = decide(
        user_message,
        conversation_history,
        sha256=sha256,
        chars=chars,
        protect_first_n=int(getattr(compressor, "protect_first_n", 0) or 0),
        effective_protect_first_n=int(compressor._effective_protect_first_n(probe)),
        is_summary=ContextCompressor._is_context_summary_message,
    )
    if new_message is None or first is None:
        logger.info("resume-prefix-dedupe: kept (%s), sha256=%s chars=%d", reason, sha256[:12], chars)
        return None
    if not wire_prefix_carries_block(agent, conversation_history, sha256, chars):
        # The replay cleanup the request builder applies would leave no verbatim carrier on the wire.
        logger.info("resume-prefix-dedupe: kept (carrier_not_on_wire), sha256=%s chars=%d", sha256[:12], chars)
        return None
    compressor._pinned_head_rows = max(int(getattr(compressor, "_pinned_head_rows", 0) or 0), first + 1)
    logger.info(
        "resume-prefix-dedupe: omitted %d repeated chars (sha256=%s); first user message pinned (rows=%d)",
        chars, sha256[:12], compressor._pinned_head_rows,
    )
    return new_message
