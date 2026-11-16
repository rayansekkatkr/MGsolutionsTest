"""HTTP front for the inference engine.

GET  /health  200 once the models are loaded and warmed up
POST /infer   body = JPEG bytes, header X-Request-Id -> {"request_id", "keypoints_2d", "keypoints_3d"}

Requests are decoded as they arrive and wait in a small pen. Each time the GPU is free the batch
worker takes up to MAX_BATCH of them, newest first, and runs them through both models together.
Under light load a request is processed alone and at once. When more arrives than the GPU can
take, the requests that would wait longest are answered 503 immediately (see admission.Pen), so
the answers that do go out stay fast. The proxy in front caps how many requests reach this process at all.

Every few seconds the process also reports what it sees of the load (admission.FleetSignal) as
CloudWatch metrics and as a log line; the scaling alarms of the fleet are driven by them.
"""
import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from aiohttp import web

from admission import FleetSignal, Pen

PORT = int(os.environ.get("PORT", "8080"))
PRECISION = os.environ.get("PRECISION", "fp32")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "3"))
MAX_WAITING = int(os.environ.get("MAX_WAITING", "8"))      # decoded requests waiting for the GPU
MAX_AGE_S = float(os.environ.get("MAX_AGE_S", "2"))        # a request that waited longer is answered 503
MAX_PENDING = int(os.environ.get("MAX_PENDING", "64"))     # requests inside this process, all states
DECODE_THREADS = int(os.environ.get("DECODE_THREADS", "3"))
OPTIONS = {name: os.environ.get(name.upper(), "0") == "1" for name in ("head_fp16", "channels_last", "cudnn_benchmark")}
METRIC_NAMESPACE = os.environ.get("METRIC_NAMESPACE", "")  # empty: no CloudWatch metrics
IDLE_RATE = float(os.environ.get("IDLE_RATE", "3"))        # requests/s under which this task counts as idle
SIGNAL_PERIOD_S = float(os.environ.get("SIGNAL_PERIOD_S", "5"))

cv2.setNumThreads(1)  # decoding runs in its own pool; no nested threading


class Busy(Exception):
    pass


class Job:
    __slots__ = ("image", "future")

    def __init__(self, image, future):
        self.image, self.future = image, future


class State:
    ready = False
    pending = 0
    decoding = 0
    engine = None
    loop = None
    pen = Pen(MAX_WAITING, max_age_s=MAX_AGE_S)
    signal = FleetSignal(idle_rate=IDLE_RATE)
    decoders = ThreadPoolExecutor(max_workers=DECODE_THREADS, thread_name_prefix="decode")
    served = turned_away = batches = 0
    batch_seconds = 0.0


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
    """Takes the newest waiting requests, up to MAX_BATCH, and runs them through both models."""
    while True:
        taken, turned_away = State.pen.take(MAX_BATCH)
        for job in turned_away:
            State.loop.call_soon_threadsafe(_resolve, job.future, None, Busy())
        jobs = [job for job in taken if not job.future.cancelled()]  # the client may have gone away
        if not jobs:
            continue
        started = time.monotonic()
        try:
            results = State.engine.infer([job.image for job in jobs])
        except Exception as error:  # one bad batch must not stop the worker
            print(f"inference failed for a batch of {len(jobs)}: {error!r}", flush=True)
            for job in jobs:
                State.loop.call_soon_threadsafe(_resolve, job.future, None, error)
            continue
        State.batches += 1
        State.batch_seconds += time.monotonic() - started
        for job, result in zip(jobs, results):
            State.loop.call_soon_threadsafe(_resolve, job.future, result, None)


def signal_worker():
    """Reports the fleet signals: CloudWatch metrics when a namespace is given, and log lines."""
    client = None
    if METRIC_NAMESPACE:
        try:
            import boto3
            from botocore.config import Config
            client = boto3.client("cloudwatch", config=Config(connect_timeout=2, read_timeout=2,
                                                              retries={"total_max_attempts": 1}))
        except Exception as error:
            print(f"fleet metrics disabled: {error!r}", flush=True)
    previous, failures, last_summary = None, 0, time.monotonic()
    counted = (0, 0, 0, 0.0)
    while True:
        time.sleep(SIGNAL_PERIOD_S)
        state = State.signal.state()
        flags = (state["saturated"], state["idle"])
        if flags != previous:
            print(f"fleet signal: saturated={flags[0]} idle={flags[1]} armed={state['armed']} "
                  f"arrivals={state['arrivals_per_s']:.1f}/s", flush=True)
            previous = flags
        if state["idle"]:
            print("fleet-idle 1", flush=True)  # counted by a log metric filter: second path to the alarm
        if client is not None:
            try:
                client.put_metric_data(Namespace=METRIC_NAMESPACE, MetricData=[
                    {"MetricName": "FleetIdle", "Value": state["idle"], "StorageResolution": 1},
                    {"MetricName": "FleetSaturated", "Value": state["saturated"], "StorageResolution": 1}])
                failures = 0
            except Exception as error:
                failures += 1
                if failures in (1, 12) or failures % 120 == 0:
                    print(f"fleet metrics not delivered ({failures} in a row): {error!r}", flush=True)
        if time.monotonic() - last_summary >= 30:
            now = (State.served, State.turned_away, State.batches, State.batch_seconds)
            served, turned_away, batches, seconds = (a - b for a, b in zip(now, counted))
            if served or turned_away:
                print(f"last 30 s: served {served}, turned away {turned_away}, "
                      f"{served / max(batches, 1):.1f} photos per batch, {1000 * seconds / max(batches, 1):.0f} ms per batch",
                      flush=True)
            counted, last_summary = now, time.monotonic()


def load_models():
    started = time.monotonic()
    try:
        from engine import Engine

        State.engine = Engine(precision=PRECISION, **OPTIONS)
        # Warm up with the batch sizes used in service so the first real requests are not slower.
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        for size in range(1, MAX_BATCH + 1):
            State.engine.infer([image] * size)
    except Exception as error:
        # A process that can never become ready must not stay up answering 503: exit, so that the
        # container is restarted and the failure is visible.
        print(f"could not load the models: {error!r}", flush=True)
        os._exit(1)
    threading.Thread(target=batch_worker, name="batch", daemon=True).start()
    threading.Thread(target=signal_worker, name="signal", daemon=True).start()
    State.ready = True
    print(f"ready: device={State.engine.device} precision={'fp16' if State.engine.fp16 else 'fp32'} "
          f"max_batch={MAX_BATCH} max_waiting={MAX_WAITING} options={[k for k, v in OPTIONS.items() if v]} "
          f"in {time.monotonic() - started:.1f} s", flush=True)


def reply(status: int, payload: dict) -> web.Response:
    return web.Response(status=status, body=json.dumps(payload, separators=(",", ":")).encode(),
                        content_type="application/json")


def turn_away(request_id: str, reason: str) -> web.Response:
    State.turned_away += 1
    State.signal.turned_away()
    return reply(503, {"request_id": request_id, "error": reason})


async def health(request: web.Request) -> web.Response:
    return reply(200, {"status": "ok"}) if State.ready else reply(503, {"status": "loading"})


async def infer(request: web.Request) -> web.Response:
    request_id = request.headers.get("X-Request-Id", "")
    if not State.ready:
        return reply(503, {"request_id": request_id, "error": "loading"})
    State.signal.arrival()
    if State.pending >= MAX_PENDING or State.decoding >= 2 * DECODE_THREADS:
        return turn_away(request_id, "busy")
    State.pending += 1
    try:
        body = await request.read()
        State.decoding += 1
        try:
            image = await State.loop.run_in_executor(State.decoders, decode, body)
        finally:
            State.decoding -= 1
        if image is None:
            return reply(400, {"request_id": request_id, "error": "body is not a decodable image"})
        future = State.loop.create_future()
        pushed_out = State.pen.put(Job(image, future))
        if pushed_out is not None:
            _resolve(pushed_out.future, None, Busy())
        try:
            keypoints_2d, keypoints_3d = await future
        except Busy:
            return turn_away(request_id, "busy")
        except Exception:
            return reply(500, {"request_id": request_id, "error": "inference failed"})
        State.served += 1
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
