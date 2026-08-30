from __future__ import annotations

import base64
import io
import json
import threading
import time
from dataclasses import dataclass, field
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


@dataclass
class _PendingChat:
    body: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: Exception | None = None


class Qwen3VLEngine:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        *,
        max_batch_size: int = 3,
        batch_window_s: float = 0.03,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.model_id = model_id
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_window_s = max(0.0, float(batch_window_s))
        self._generation_lock = threading.Lock()
        self._batch_condition = threading.Condition()
        self._pending_chats: list[_PendingChat] = []
        self._batch_leader = False
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        ).eval()
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - started

    @staticmethod
    def _generation_signature(body: dict[str, Any]) -> tuple[int, float, float, int]:
        return (
            min(int(body.get("max_tokens", 1024)), 2048),
            float(body.get("temperature", 0.0)),
            float(body.get("top_p", 1.0)),
            int(body.get("seed", 1024)),
        )

    def _chat_batch(self, bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        torch = self.torch
        if not bodies:
            return []
        conversations = []
        for body in bodies:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages is required")
            conversations.append(_normalize_messages(messages))

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        max_tokens, temperature, top_p, seed = self._generation_signature(bodies[0])
        torch.manual_seed(seed)
        generate_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "do_sample": False}
        if temperature > 0:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": max(temperature, 1e-5),
                    "top_p": top_p,
                }
            )
        with self._generation_lock, torch.inference_mode():
            generated = self.model.generate(**inputs, **generate_kwargs)

        prompt_width = inputs.input_ids.shape[1]
        trimmed = generated[:, prompt_width:]
        texts = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        now = int(time.time())
        return [
            {
                "id": f"chatcmpl-local-{time.time_ns()}-{index}",
                "object": "chat.completion",
                "created": now,
                "model": body.get("model") or self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text.strip()},
                        "finish_reason": "stop",
                    }
                ],
            }
            for index, (body, text) in enumerate(zip(bodies, texts, strict=True))
        ]

    def _take_compatible_batch(self) -> list[_PendingChat]:
        with self._batch_condition:
            if not self._pending_chats:
                return []
            signature = self._generation_signature(self._pending_chats[0].body)
            selected: list[_PendingChat] = []
            remaining: list[_PendingChat] = []
            for request in self._pending_chats:
                if (
                    len(selected) < self.max_batch_size
                    and self._generation_signature(request.body) == signature
                ):
                    selected.append(request)
                else:
                    remaining.append(request)
            self._pending_chats = remaining
            return selected

    def _drain_microbatches(self) -> None:
        if self.batch_window_s:
            time.sleep(self.batch_window_s)
        while True:
            batch = self._take_compatible_batch()
            if not batch:
                with self._batch_condition:
                    if self._pending_chats:
                        continue
                    self._batch_leader = False
                return
            try:
                results = self._chat_batch([request.body for request in batch])
            except Exception as exc:  # noqa: BLE001 - batch boundary must wake all waiters
                for request in batch:
                    request.error = exc
                    request.event.set()
            else:
                for request, result in zip(batch, results, strict=True):
                    request.result = result
                    request.event.set()

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        request = _PendingChat(body=body)
        with self._batch_condition:
            self._pending_chats.append(request)
            leader = not self._batch_leader
            if leader:
                self._batch_leader = True

        if leader:
            self._drain_microbatches()
        else:
            request.event.wait()

        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("Qwen3-VL microbatch completed without a result")
        return request.result


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
