import threading
import time
from types import MethodType

from modal_world.qwen_vlm_server import Qwen3VLEngine


def _fake_body(index: int) -> dict:
    return {
        "model": "fake",
        "messages": [{"role": "user", "content": f"caption {index}"}],
        "max_tokens": 128,
        "temperature": 0.1,
        "top_p": 1.0,
        "seed": 1024,
    }


def test_concurrent_chat_requests_are_microbatched_by_three():
    engine = object.__new__(Qwen3VLEngine)
    engine.model_id = "fake"
    engine.max_batch_size = 3
    engine.batch_window_s = 0.05
    engine._generation_lock = threading.Lock()
    engine._batch_condition = threading.Condition()
    engine._pending_chats = []
    engine._batch_leader = False
    batch_sizes: list[int] = []
    call_lock = threading.Lock()

    def fake_chat_batch(self, bodies):
        with call_lock:
            batch_sizes.append(len(bodies))
        time.sleep(0.01)
        return [{"body": body} for body in bodies]

    engine._chat_batch = MethodType(fake_chat_batch, engine)
    barrier = threading.Barrier(6)
    results = [None] * 6

    def worker(index: int):
        barrier.wait()
        results[index] = engine.chat(_fake_body(index))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(batch_sizes) == [3, 3]
    assert all(result is not None for result in results)


def test_different_generation_settings_do_not_share_a_batch():
    engine = object.__new__(Qwen3VLEngine)
    engine.model_id = "fake"
    engine.max_batch_size = 3
    engine.batch_window_s = 0.05
    engine._generation_lock = threading.Lock()
    engine._batch_condition = threading.Condition()
    engine._pending_chats = []
    engine._batch_leader = False
    signatures = []

    def fake_chat_batch(self, bodies):
        signatures.append([self._generation_signature(body) for body in bodies])
        return [{"body": body} for body in bodies]

    engine._chat_batch = MethodType(fake_chat_batch, engine)
    bodies = [_fake_body(0), _fake_body(1)]
    bodies[1]["temperature"] = 0.0
    barrier = threading.Barrier(2)

    def worker(body):
        barrier.wait()
        engine.chat(body)

    threads = [threading.Thread(target=worker, args=(body,)) for body in bodies]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert [len(batch) for batch in signatures] == [1, 1]
