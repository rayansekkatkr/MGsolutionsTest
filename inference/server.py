"""HTTP front for the inference engine.

GET  /health  200 once the models are loaded and warmed up
POST /infer   body = JPEG bytes, header X-Request-Id -> {"request_id", "keypoints_2d", "keypoints_3d"}

Requests are grouped: while the GPU works on one batch, the next ones accumulate and leave together
as soon as it is free. Under light load a request is processed alone, with at most BATCH_WAIT_MS of
added delay. Admission control (how many requests may be in the system) is the proxy's job; the
limit here is a second line of defence.
"""
import asyncio
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from aiohttp import web

PORT = int(os.environ.get("PORT", "8080"))
PRECISION = os.environ.get("PRECISION", "fp32")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "8"))
BATCH_WAIT_MS = float(os.environ.get("BATCH_WAIT_MS", "3"))
MAX_PENDING = int(os.environ.get("MAX_PENDING", "64"))
MAX_AGE_S = float(os.environ.get("MAX_AGE_S", "8"))  # the client gives up after 10 s
DECODE_THREADS = int(os.environ.get("DECODE_THREADS", "3"))
WORKERS = int(os.environ.get("WORKERS", "1"))  # batch threads sharing the models
OPTIONS = {name: os.environ.get(name.upper(), "0") == "1" for name in ("head_fp16", "channels_last", "cudnn_benchmark")}

cv2.setNumThreads(1)  # decoding runs in its own pool; no nested threading


class State:
    ready = False
    pending = 0
    engine = None
    loop = None
    jobs: "queue.Queue" = queue.Queue()
    decoders = ThreadPoolExecutor(max_workers=DECODE_THREADS, thread_name_prefix="decode")


def decode(body: bytes):
    """Same decoder and colour order as cv2.imread: BGR, 8 bits."""
    return cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)


def _resolve(future, result, error):
    if future.cancelled():
        return
    if error is None:
        future.set_result(result)
    else:
        future.set_exception(error)


def batch_worker():
    """Takes whatever is waiting, up to MAX_BATCH, and runs it through both models."""
    wait = BATCH_WAIT_MS / 1000
    while True:
        batch = [State.jobs.get()]
        deadline = time.monotonic() + wait
        while len(batch) < MAX_BATCH:
            try:
                batch.append(State.jobs.get(timeout=max(0.0, deadline - time.monotonic())))
            except queue.Empty:
                break
        now = time.monotonic()
        fresh = []
        for job in batch:
            image, future, arrived = job
            if future.cancelled() or now - arrived > MAX_AGE_S:
                State.loop.call_soon_threadsafe(_resolve, future, None, TimeoutError("request expired in queue"))
            else:
                fresh.append(job)
        if not fresh:
            continue
        try:
            results = State.engine.infer([job[0] for job in fresh])
        except Exception as error:  # one bad batch must not stop the worker
            print(f"inference failed for a batch of {len(fresh)}: {error!r}", flush=True)
            for _, future, _ in fresh:
                State.loop.call_soon_threadsafe(_resolve, future, None, error)
            continue
        for (_, future, _), result in zip(fresh, results):
            State.loop.call_soon_threadsafe(_resolve, future, result, None)


def load_models():
    from engine import Engine

    started = time.monotonic()
    State.engine = Engine(precision=PRECISION, **OPTIONS)
    # Warm up with the batch sizes used in service so the first real requests are not slower.
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    for size in sorted({1, MAX_BATCH}):
        State.engine.infer([image] * size)
    for n in range(WORKERS):
        threading.Thread(target=batch_worker, name=f"batch-{n}", daemon=True).start()
    State.ready = True
    print(f"ready: device={State.engine.device} precision={'fp16' if State.engine.fp16 else 'fp32'} "
          f"max_batch={MAX_BATCH} workers={WORKERS} options={[k for k, v in OPTIONS.items() if v]} "
          f"in {time.monotonic() - started:.1f} s", flush=True)


def reply(status: int, payload: dict) -> web.Response:
    return web.Response(status=status, body=json.dumps(payload, separators=(",", ":")).encode(),
                        content_type="application/json")


async def health(request: web.Request) -> web.Response:
    return reply(200, {"status": "ok"}) if State.ready else reply(503, {"status": "loading"})


async def infer(request: web.Request) -> web.Response:
    request_id = request.headers.get("X-Request-Id", "")
    if not State.ready:
        return reply(503, {"request_id": request_id, "error": "loading"})
    if State.pending >= MAX_PENDING:
        return reply(503, {"request_id": request_id, "error": "busy"})
    State.pending += 1
    try:
        arrived = time.monotonic()
        body = await request.read()
        image = await State.loop.run_in_executor(State.decoders, decode, body)
        if image is None:
            return reply(400, {"request_id": request_id, "error": "body is not a decodable image"})
        future = State.loop.create_future()
        State.jobs.put((image, future, arrived))
        try:
            keypoints_2d, keypoints_3d = await future
        except TimeoutError:
            return reply(503, {"request_id": request_id, "error": "expired"})
        except Exception:
            return reply(500, {"request_id": request_id, "error": "inference failed"})
        return reply(200, {"request_id": request_id,
                           "keypoints_2d": np.round(keypoints_2d, 2).tolist(),
                           "keypoints_3d": np.round(keypoints_3d, 5).tolist()})
    finally:
        State.pending -= 1


async def on_startup(app: web.Application) -> None:
    State.loop = asyncio.get_running_loop()
    threading.Thread(target=load_models, name="load", daemon=True).start()


def main() -> None:
    app = web.Application(client_max_size=32 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_post("/infer", infer)
    app.on_startup.append(on_startup)
    print(f"listening on 0.0.0.0:{PORT} (loading models)", flush=True)
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None, print=None)


if __name__ == "__main__":
    main()
