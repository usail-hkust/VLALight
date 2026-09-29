import json
from pathlib import Path

import pytest

from v35_online_cooperative_grpo.message_router import (
    MessageProtocolError,
    parse_sender_message,
    render_receiver_message_context,
    route_messages,
)


def sender_response(signal: str, first: str, second: str) -> str:
    movements = {
        "ETWT": ("ET", "WT"), "NTST": ("NT", "ST"),
        "ELWL": ("EL", "WL"), "NLSL": ("NL", "SL"),
    }[signal]
    return (
        f"<signal>{signal}</signal><message>"
        f'<to movement="{movements[0]}">{first}</to>'
        f'<to movement="{movements[1]}">{second}</to></message>'
    )


def jinan_routes() -> dict:
    root = Path(__file__).parents[2]
    return json.loads((root / "v35_online_cooperative_grpo/artifacts/movement_routes_jinan.json").read_text())


def test_parse_signal_constrained_two_movement_message():
    parsed = parse_sender_message(sender_response("NLSL", "eastbound left flow", ""), "NLSL")
    assert parsed == {"NL": "eastbound left flow", "SL": ""}


def test_parser_rejects_wrong_signal_movement_pair():
    response = '<signal>NTST</signal><message><to movement="ET">x</to><to movement="ST">y</to></message>'
    with pytest.raises(MessageProtocolError, match="requires exactly one"):
        parse_sender_message(response)


def test_router_uses_topology_and_skips_boundary_messages():
    routes = jinan_routes()
    parsed = parse_sender_message(sender_response("NLSL", "to east", "boundary text"))
    routed = route_messages("intersection_1_1", parsed, routes)
    assert len(routed) == 1
    assert routed[0].receiver_id == "intersection_2_1"
    assert routed[0].movement == "NL"
    assert routed[0].receiver_entry_direction == "W"


def test_receiver_context_contains_system_owned_direction_metadata():
    routed = route_messages(
        "intersection_1_1",
        parse_sender_message(sender_response("ETWT", "", "westbound traffic approaching")),
        jinan_routes(),
    )
    context = render_receiver_message_context(routed)["intersection_2_1"]
    assert "Source movement: WT" in context
    assert "enters your W approach" in context
    assert "westbound traffic approaching" in context
