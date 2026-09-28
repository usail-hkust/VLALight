"""Build phase-3 sender/receiver prompts from the offline-GRPO prompt baseline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


BASE_LAYOUT = """Critical format warning:

Your response must use exactly one of these valid layouts:

Fast mode:
<perception>...</perception>
<mode>fast</mode>
<signal>...</signal>

Slow mode:
<perception>...</perception>
<mode>slow</mode>
<reasoning>...</reasoning>
<signal>...</signal>

Do not output any text before the first tag or after the last tag.
Do not omit, rename, or reorder any tag required by the selected mode.
Do not use markdown code fences.
"""

SENDER_LAYOUT = """Critical format warning:

Your response must use exactly one of these valid layouts:

Fast mode:
<perception>...</perception>
<mode>fast</mode>
<signal>...</signal>
<message>...</message>

Slow mode:
<perception>...</perception>
<mode>slow</mode>
<reasoning>...</reasoning>
<signal>...</signal>
<message>...</message>

Do not output any text before the first tag or after the last tag.
Do not omit, rename, or reorder any tag required by the selected mode.
Do not use markdown code fences.
"""

SENDER_MESSAGE_RULES = """

Online cooperation sender requirement:
- After <signal>, output exactly one <message> block.
- The <message> block must contain exactly two <to movement="...">...</to> elements.
- The two movement values must be the two movements released by the selected signal:
  ETWT -> ET and WT; NTST -> NT and ST; ELWL -> EL and WL; NLSL -> NL and SL.
- Each <to> text is free-form and may be empty when no coordination is needed.
- Do not include destination intersection IDs or entry-direction claims; the system routes each movement.

Example:
<message>
  <to movement="NT">...</to>
  <to movement="ST">...</to>
</message>
"""

RECEIVER_SUFFIX = """

Online cooperation receiver context:
- The following system-routed messages are for the current control cycle only.
- Use them as supplementary upstream-arrival evidence, together with the visual inputs.
- Do not output a <message> tag. Keep exactly the baseline response format.

{message_context}
"""


def _user_message(messages: list[dict[str, Any]]) -> dict[str, Any]:
    users = [message for message in messages if message.get("role") == "user"]
    if len(users) != 1:
        raise ValueError(f"Expected exactly one user message, found {len(users)}.")
    return users[0]


def build_sender_prompt(base_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the offline-GRPO prompt verbatim except its output-layout block."""
    messages = deepcopy(base_messages)
    user = _user_message(messages)
    content = user["content"]
    if content.count(BASE_LAYOUT) != 1:
        raise ValueError("Offline GRPO baseline layout block is missing or duplicated.")
    user["content"] = content.replace(BASE_LAYOUT, SENDER_LAYOUT, 1) + SENDER_MESSAGE_RULES
    return messages


def build_receiver_prompt(
    base_messages: list[dict[str, Any]], message_context: str | None
) -> list[dict[str, Any]]:
    """Keep the baseline response protocol and inject current-cycle router output."""
    messages = deepcopy(base_messages)
    if message_context:
        _user_message(messages)["content"] += RECEIVER_SUFFIX.format(message_context=message_context)
    return messages
