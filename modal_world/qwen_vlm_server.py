from __future__ import annotations

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _decode_data_image(url: str):
    from PIL import Image

    if not url.startswith("data:image/"):
        return url
    try:
        _, encoded = url.split(",", 1)
    except ValueError as exc:
        raise ValueError("invalid image data URI") from exc
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, list):
            normalized.append(
                {
                    "role": message.get("role", "user"),
                    "content": [{"type": "text", "text": str(content)}],
                }
            )
            continue
        items = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if not isinstance(url, str):
                    raise ValueError("image_url content must contain a string URL")
                items.append({"type": "image", "image": _decode_data_image(url)})
            elif item_type == "image":
                image = item.get("image")
                if isinstance(image, str) and image.startswith("data:image/"):
                    image = _decode_data_image(image)
                items.append({"type": "image", "image": image})
            elif item_type == "text":
                items.append({"type": "text", "text": str(item.get("text", ""))})
        normalized.append({"role": message.get("role", "user"), "content": items})
    return normalized


class Qwen3VLEngine:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct") -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.model_id = model_id
        self._generation_lock = threading.Lock()
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        ).eval()
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - started

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages is required")
        normalized = _normalize_messages(messages)
        inputs = self.processor.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        seed = int(body.get("seed", 1024))
        torch.manual_seed(seed)
        max_tokens = min(int(body.get("max_tokens", 1024)), 2048)
        temperature = float(body.get("temperature", 0.0))
        generate_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "do_sample": False}
        if temperature > 0:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": max(temperature, 1e-5),
                    "top_p": float(body.get("top_p", 1.0)),
                }
            )
        with self._generation_lock, torch.inference_mode():
            generated = self.model.generate(**inputs, **generate_kwargs)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return {
            "id": f"chatcmpl-local-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }


def start_openai_server(engine: Qwen3VLEngine, host: str = "127.0.0.1", port: int = 8000):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *args):
            return

        def _json(self, code: int, payload: dict[str, Any]):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/v1/models":
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": engine.model_id, "object": "model", "owned_by": "modal-world"}
                        ],
                    },
                )
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                self._json(200, engine.chat(body))
            except Exception as exc:  # noqa: BLE001 - HTTP boundary must serialize model failures
                self._json(500, {"error": {"message": str(exc), "type": type(exc).__name__}})

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="qwen3-vl-openai")
    thread.start()
    return server, thread
