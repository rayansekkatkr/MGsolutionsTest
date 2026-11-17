"""HTTP front for the inference engine.

GET  /health  200 once the models are loaded and warmed up
POST /infer   body = JPEG bytes, header X-Request-Id -> {"request_id", "keypoints_2d", "keypoints_3d"}

Requests wait in a small pen. Batch workers take up to MAX_BATCH of them, newest first, decode
them and run them through both models together. With two workers, one prepares its batch on the
CPU while the other's is on the GPU, so the GPU does not wait.
Under light load a request is processed alone and at once. When more arrives than the GPU can
take, the requests that would wait longest are answered 503 immediately (see admission.Pen), so
the answers that do go out stay fast. The proxy in front caps how many requests reach this process at all.

Every few seconds the process also reports what it sees of the load (admission.FleetSignal) as
CloudWatch metrics and as a log line; the scaling alarms of the fleet are driven by them.

On SIGTERM (the host or the task is being stopped) the process does not stop at once: /health
answers 503 from then on, so that the load balancer takes the task out, and /infer keeps
answering for DRAIN_S seconds, so that the requests still routed here meanwhile are not lost.
"""
import asyncio
import json
import os
import signal
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
MAX_WAITING = int(os.environ.get("MAX_WAITING", "8"))      # requests waiting for a batch worker
WORKERS = int(os.environ.get("WORKERS", "2"))              # batch workers; at most the engine's lanes
MAX_AGE_S = float(os.environ.get("MAX_AGE_S", "2"))        # a request that waited longer is answered 503
MAX_PENDING = int(os.environ.get("MAX_PENDING", "64"))     # requests inside this process, all states
DECODE_THREADS = int(os.environ.get("DECODE_THREADS", "3"))
OPTIONS = {name: os.environ.get(name.upper(), "0") == "1" for name in ("head_fp16", "channels_last", "cudnn_benchmark")}
TRT_DIR = os.environ.get("TRT_DIR", "")                    # directory with det.engine and pose.engine; empty: PyTorch only
METRIC_NAMESPACE = os.environ.get("METRIC_NAMESPACE", "")  # empty: no CloudWatch metrics
IDLE_RATE = float(os.environ.get("IDLE_RATE", "3"))        # requests/s under which this task counts as idle
SIGNAL_PERIOD_S = float(os.environ.get("SIGNAL_PERIOD_S", "5"))
DRAIN_S = float(os.environ.get("DRAIN_S", "20"))           # seconds /infer keeps answering after SIGTERM

cv2.setNumThreads(1)  # decoding runs in its own pool; no nested threading


class Busy(Exception):
    pass


class NotAnImage(Exception):
    pass


class Job:
    __slots__ = ("body", "future")

    def __init__(self, body, future):
        self.body, self.future = body, future


class State:
    ready = False
    stopping = False
    pending = 0
    workers = 1
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


def batch_worker(lane: int):
    """Takes the newest waiting requests, up to MAX_BATCH, decodes them and runs them through both models."""
    while True:
        taken, turned_away = State.pen.take(MAX_BATCH)
        for job in turned_away:
            State.loop.call_soon_threadsafe(_resolve, job.future, None, Busy())
        taken = [job for job in taken if not job.future.cancelled()]  # the client may have gone away
        # Decoded here, not on arrival: only requests that are going to be answered cost a decode.
        jobs, images = [], []
        for job, image in zip(taken, State.decoders.map(decode, [job.body for job in taken])):
            if image is None:
                State.loop.call_soon_threadsafe(_resolve, job.future, None, NotAnImage())
            else:
                jobs.append(job)
                images.append(image)
        if not jobs:
            continue
        started = time.monotonic()
        try:
            results = State.engine.infer(images, lane)
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

        State.engine = Engine(precision=PRECISION, trt_dir=TRT_DIR, **OPTIONS)
        # Warm up with the batch sizes used in service, several times over: the first passes of
        # each size are slower, and the first requests after start should not pay for that.
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        State.workers = max(1, min(WORKERS, State.engine.lanes))
        for lane in range(State.workers):
            for _ in range(5 if State.engine.device.startswith("cuda") else 1):
                for size in range(1, MAX_BATCH + 1):
                    State.engine.infer([image] * size, lane)
    except Exception as error:
        # A process that can never become ready must not stay up answering 503: exit, so that the
        # container is restarted and the failure is visible.
        print(f"could not load the models: {error!r}", flush=True)
        os._exit(1)
    for lane in range(State.workers):
        threading.Thread(target=batch_worker, args=(lane,), name=f"batch-{lane}", daemon=True).start()
    threading.Thread(target=signal_worker, name="signal", daemon=True).start()
    State.ready = True
    runs = lambda net: "tensorrt" if net is not None else "pytorch"
    print(f"ready: device={State.engine.device} precision={'fp16' if State.engine.fp16 else 'fp32'} "
          f"detector={runs(State.engine.trt_det)} pose={runs(State.engine.trt_pose)} "
          f"workers={State.workers} max_batch={MAX_BATCH} max_waiting={MAX_WAITING} options={[k for k, v in OPTIONS.items() if v]} "
          f"in {time.monotonic() - started:.1f} s", flush=True)


def reply(status: int, payload: dict) -> web.Response:
    return web.Response(status=status, body=json.dumps(payload, separators=(",", ":")).encode(),
                        content_type="application/json")


def turn_away(request_id: str, reason: str) -> web.Response:
    State.turned_away += 1
    State.signal.turned_away()
    return reply(503, {"request_id": request_id, "error": reason})


async def health(request: web.Request) -> web.Response:
    if State.stopping:
        return reply(503, {"status": "stopping"})
    return reply(200, {"status": "ok"}) if State.ready else reply(503, {"status": "loading"})


def stop() -> None:
    """SIGTERM or SIGINT. A second signal, or one received before the models are ready, stops at once."""
    if State.stopping or not State.ready or DRAIN_S <= 0:
        print("stopped", flush=True)
        os._exit(0)
    State.stopping = True
    print(f"stopping: /health answers 503 from now on, /infer keeps answering for {DRAIN_S:.0f} s", flush=True)
    State.loop.call_later(DRAIN_S, stop)


async def infer(request: web.Request) -> web.Response:
    request_id = request.headers.get("X-Request-Id", "")
    if not State.ready:
        return reply(503, {"request_id": request_id, "error": "loading"})
    State.signal.arrival()
    # The body is always read before answering: a proxy handles an answer that arrives while it
    # is still forwarding the request less predictably than a plain request/answer exchange.
    State.pending += 1
    try:
        body = await request.read()
        if State.pending > MAX_PENDING:
            return turn_away(request_id, "busy")
        future = State.loop.create_future()
        pushed_out = State.pen.put(Job(body, future))
        if pushed_out is not None:
            _resolve(pushed_out.future, None, Busy())
        try:
            keypoints_2d, keypoints_3d = await future
        except Busy:
            return turn_away(request_id, "busy")
        except NotAnImage:
            return reply(400, {"request_id": request_id, "error": "body is not a decodable image"})
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
    for number in (signal.SIGTERM, signal.SIGINT):
        State.loop.add_signal_handler(number, stop)
    threading.Thread(target=load_models, name="load", daemon=True).start()


def main() -> None:
    app = web.Application(client_max_size=32 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_post("/infer", infer)
    app.on_startup.append(on_startup)
    print(f"listening on 0.0.0.0:{PORT} (loading models)", flush=True)
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None, print=None, handle_signals=False)


if __name__ == "__main__":
    main()
