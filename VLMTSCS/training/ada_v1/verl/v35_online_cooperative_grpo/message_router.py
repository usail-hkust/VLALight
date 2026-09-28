"""Message protocol, movement routing, and receiver-context rendering.

These functions are intentionally independent of SUMO and verl workers so the
online rollout can use one tested protocol in training and deployment.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any


SIGNAL_MOVEMENTS = {
    "ETWT": ("ET", "WT"),
    "NTST": ("NT", "ST"),
    "ELWL": ("EL", "WL"),
    "NLSL": ("NL", "SL"),
}


class MessageProtocolError(ValueError):
    """Raised when a sender response does not satisfy the message protocol."""


@dataclass(frozen=True)
class RoutedMessage:
    sender_id: str
    receiver_id: str
    movement: str
    receiver_entry_direction: str
    content: str


def _single_tag(text: str, name: str) -> str:
    values = re.findall(rf"<{name}>\s*(.*?)\s*</{name}>", text, re.I | re.S)
    if len(values) != 1:
        raise MessageProtocolError(f"Expected exactly one <{name}> tag, found {len(values)}.")
    return values[0].strip()


def parse_sender_message(response: str, selected_signal: str | None = None) -> dict[str, str]:
    """Parse one sender ``<message>`` block into movement -> free-text content.

    ``selected_signal`` is normally derived from the response's ``<signal>`` by
    the rollout parser. Supplying it makes the cross-check explicit in tests.
    """
    signal = _single_tag(response, "signal")
    if selected_signal is not None and signal != selected_signal:
        raise MessageProtocolError(
            f"<signal> is {signal!r}, but selected_signal is {selected_signal!r}."
        )
    if signal not in SIGNAL_MOVEMENTS:
        raise MessageProtocolError(f"Unsupported signal: {signal!r}.")
    body = _single_tag(response, "message")
    tags = list(re.finditer(r'<to\s+movement="([A-Z]{2})"\s*>(.*?)</to>', body, re.S))
    residue = re.sub(r'<to\s+movement="[A-Z]{2}"\s*>.*?</to>', "", body, flags=re.S).strip()
    if residue:
        raise MessageProtocolError("<message> may contain only <to movement=\"..\"> elements.")
    expected = SIGNAL_MOVEMENTS[signal]
    movements = tuple(match.group(1) for match in tags)
    if len(tags) != 2 or set(movements) != set(expected) or len(set(movements)) != 2:
        raise MessageProtocolError(
            f"Signal {signal} requires exactly one <to> for {expected}; found {movements}."
        )
    return {match.group(1): html.unescape(match.group(2).strip()) for match in tags}


def route_messages(
    sender_id: str, parsed_message: dict[str, str], route_table: dict[str, Any]
) -> list[RoutedMessage]:
    """Route non-empty, non-boundary sender movement messages to receivers."""
    try:
        movement_routes = route_table["routes"][sender_id]["movements"]
    except KeyError as exc:
        raise KeyError(f"No route-table entry for sender {sender_id!r}.") from exc
    routed: list[RoutedMessage] = []
    for movement, content in parsed_message.items():
        if movement not in movement_routes:
            raise KeyError(f"No route for {sender_id} movement {movement}.")
        route = movement_routes[movement]
        if not content or route["is_boundary"]:
            continue
        receiver_id = route.get("receiver_id")
        entry = route.get("receiver_entry_direction")
        if not receiver_id or entry not in {"E", "W", "N", "S"}:
            raise ValueError(f"Invalid non-boundary route for {sender_id} {movement}.")
        routed.append(RoutedMessage(sender_id, receiver_id, movement, entry, content))
    return routed


def render_receiver_message_context(routed_messages: list[RoutedMessage]) -> dict[str, str]:
    """Render messages grouped by receiver for injection into its current prompt."""
    grouped: dict[str, list[RoutedMessage]] = {}
    for item in routed_messages:
        grouped.setdefault(item.receiver_id, []).append(item)
    rendered: dict[str, str] = {}
    for receiver_id, messages in grouped.items():
        lines = ["Current-cycle routed coordination messages:"]
        for item in sorted(messages, key=lambda value: (value.sender_id, value.movement)):
            lines.extend([
                f"- Source intersection: {item.sender_id}",
                f"  Source movement: {item.movement}",
                f"  This traffic enters your {item.receiver_entry_direction} approach.",
                f"  Sender message: {item.content}",
            ])
        rendered[receiver_id] = "\n".join(lines)
    return rendered
