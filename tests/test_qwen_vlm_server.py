from modal_world.qwen_vlm_server import _normalize_messages


def test_normalize_plain_text_message_to_multimodal_content():
    result = _normalize_messages([{"role": "system", "content": "You are helpful."}])
    assert result == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are helpful."}],
        }
    ]
