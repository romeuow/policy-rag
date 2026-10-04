from policy_rag.api.sse import encode_event, parse_sse


def test_encode_event_produces_single_frame_with_json_payload():
    frame = encode_event("token", {"text": "olá\nmundo"})
    assert frame == 'event: token\ndata: {"text": "olá\\nmundo"}\n\n'


def test_parse_sse_roundtrip_and_tolerance():
    stream = (
        ": keep-alive comment\n\n"
        + encode_event("token", {"text": "a"})
        + encode_event("citations", [{"index": 1}])
        + "event: raw\ndata: line1\ndata: line2\n\n"
        + "data: plain\r\n\r\n"
        + "event: nodata\n\n"
    )
    events = parse_sse(stream)
    assert [e.event for e in events] == ["token", "citations", "raw", "message"]
    assert events[0].data == {"text": "a"}
    assert events[1].data == [{"index": 1}]
    assert events[2].data == "line1\nline2"
    assert events[3].data == "plain"
    assert parse_sse("") == []
