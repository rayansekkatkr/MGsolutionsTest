"""Admission and fleet signalling. No model code here, so both can be tested on their own.

Pen          decoded requests waiting for the GPU, served newest first
FleetSignal  what one task concludes about the fleet from its own traffic: saturated, or idle after a peak
"""
import threading
import time
from collections import deque


class Pen:
    """Requests that are decoded and waiting for the GPU.

    The batch worker takes the newest ones. While demand stays below capacity this changes nothing:
    a batch takes everything that waits, and a request left behind by a full batch goes with the
    next one. When demand exceeds capacity it is what keeps answers fast:

    - the pen has a fixed size; once it is full each arrival pushes the oldest waiting request out;
    - while that is happening (an overflow in the last `overflow_memory_s`), requests left behind
      by a batch are turned away too instead of waiting for the next batch;
    - a request that waited longer than `max_age_s` is turned away in any case.

    Requests pushed out or turned away are answered 503 at once. An admitted request therefore
    waits for about one batch, however large the excess. A first-come-first-served queue of the
    same size would make every answer wait for the whole queue instead.
    """

    def __init__(self, capacity: int, max_age_s: float = 2.0, overflow_memory_s: float = 1.0, clock=time.monotonic):
        self.capacity = capacity
        self.max_age_s = max_age_s
        self.overflow_memory_s = overflow_memory_s
        self._clock = clock
        self._jobs = deque()  # (arrival time, job), oldest first
        self._last_overflow = float("-inf")
        self._changed = threading.Condition()

    def __len__(self) -> int:
        return len(self._jobs)

    def put(self, job):
        """Adds a job. Returns the job pushed out to make room, if any."""
        with self._changed:
            pushed_out = None
            if len(self._jobs) >= self.capacity:
                pushed_out = self._jobs.popleft()[1]
                self._last_overflow = self._clock()
            self._jobs.append((self._clock(), job))
            self._changed.notify()
        return pushed_out

    def take(self, limit: int, timeout=None):
        """Blocks until something waits, then returns (newest jobs up to `limit`, jobs turned away).

        Returns ([], []) when `timeout` elapses with nothing waiting.
        """
        with self._changed:
            if not self._jobs and not self._changed.wait_for(lambda: self._jobs, timeout):
                return [], []
            now = self._clock()
            turned_away = []
            while self._jobs and now - self._jobs[0][0] > self.max_age_s:
                turned_away.append(self._jobs.popleft()[1])
            taken = [self._jobs.pop()[1] for _ in range(min(limit, len(self._jobs)))]
            if now - self._last_overflow <= self.overflow_memory_s:
                turned_away += [job for _, job in self._jobs]
                self._jobs.clear()
        return taken, turned_away


class FleetSignal:
    """Two yes/no signals derived from the traffic one task sees, in one-second buckets.

    saturated  the task turned requests away in most of the recent seconds: demand exceeds capacity
    idle       the task has been through a period of saturation, and for the whole recent window
               it received less than `idle_rate` requests per second and turned none away

    `idle` is deliberately impossible before a first peak: a fleet that has just started is not
    shrunk on the grounds that nobody has called it yet.
    """

    def __init__(self, idle_rate: float, window_s: int = 10, arm_after_s: int = 60, clock=time.monotonic):
        self.idle_rate = idle_rate
        self.window_s = window_s
        self.arm_after_s = arm_after_s
        self._clock = clock
        self._started = int(clock())
        self._buckets = deque()  # [second, arrivals, turned away], oldest first
        self._saturated_seconds = 0
        self._lock = threading.Lock()

    def _bucket(self):
        second = int(self._clock())
        if not self._buckets or self._buckets[-1][0] != second:
            self._buckets.append([second, 0, 0])
            while self._buckets[0][0] <= second - self.window_s - 1:
                self._buckets.popleft()
        return self._buckets[-1]

    def arrival(self) -> None:
        with self._lock:
            self._bucket()[1] += 1

    def turned_away(self) -> None:
        with self._lock:
            bucket = self._bucket()
            if bucket[2] == 0:
                self._saturated_seconds += 1
            bucket[2] += 1

    def state(self) -> dict:
        with self._lock:
            now = int(self._clock())
            # The current second is still filling up: look at the `window_s` complete seconds before it.
            recent = [b for b in self._buckets if now - self.window_s <= b[0] < now]
            arrivals = sum(b[1] for b in recent)
            seconds_turning_away = sum(1 for b in recent if b[2])
            armed = self._saturated_seconds >= self.arm_after_s
            observed = now - self._started >= self.window_s
            saturated = observed and seconds_turning_away > self.window_s // 2
            idle = armed and observed and seconds_turning_away == 0 and arrivals < self.idle_rate * self.window_s
            return {"saturated": int(saturated), "idle": int(idle), "armed": armed,
                    "arrivals_per_s": arrivals / self.window_s}
