"""Checks of the admission pen and of the fleet signal. No models needed: python3 test_admission.py"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inference"))
from admission import FleetSignal, Pen  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_pen_below_capacity_loses_nothing():
    clock = Clock()
    pen = Pen(capacity=8, clock=clock)
    served = []
    for wave in range(60):                                     # waves of 1 to 5 arrivals, batches of 3
        for n in range(wave % 5 + 1):
            assert pen.put((wave, n)) is None
        clock.now += 0.08
        taken, turned_away = pen.take(3)
        served += taken
        assert not turned_away                                 # those left behind go with the next batch
    while len(pen):
        served += pen.take(3)[0]
    assert len(served) == sum(w % 5 + 1 for w in range(60))


def test_pen_under_overload_serves_only_fresh_requests():
    clock = Clock()
    pen = Pen(capacity=8, clock=clock)
    pushed_out, turned_away, served = [], [], []
    arrival = {}
    for n in range(300):                                       # 10 arrivals per batch time, batches of 3
        arrival[n] = clock.now
        out = pen.put(n)
        if out is not None:
            pushed_out.append(out)
        clock.now += 0.008
        if n % 10 == 9:
            taken, away = pen.take(3)
            assert taken == [n, n - 1, n - 2]                  # the three newest
            served += [(job, clock.now - arrival[job]) for job in taken]
            turned_away += away
    assert len(served) == 90 and max(wait for _, wait in served) < 0.03
    assert len(pushed_out) + len(turned_away) + len(pen) == 300 - 90
    assert len(pen) == 0                                       # nobody is left to wait for a later batch


def test_pen_serves_newest_and_pushes_out_oldest():
    pen = Pen(capacity=4, overflow_memory_s=0)
    assert [pen.put(n) for n in range(4)] == [None] * 4       # room for four: nobody is pushed out
    assert pen.put(4) == 0 and pen.put(5) == 1                # full: each arrival pushes the oldest out
    assert pen.take(3) == ([5, 4, 3], [])                     # newest first


def test_pen_turns_away_requests_that_waited_too_long():
    clock = Clock()
    pen = Pen(capacity=8, max_age_s=2, clock=clock)
    pen.put("old")
    clock.now += 3
    pen.put("new")
    assert pen.take(2) == (["new"], ["old"])


def test_pen_take_blocks_until_a_job_arrives():
    pen = Pen(capacity=2)
    assert pen.take(2, timeout=0.05) == ([], [])
    result = []
    worker = threading.Thread(target=lambda: result.append(pen.take(2, timeout=5)))
    worker.start()
    pen.put("job")
    worker.join(2)
    assert result == [(["job"], [])]


def run(signal, clock, seconds, arrivals_per_s, turned_away_per_s=0):
    for _ in range(seconds):
        for _ in range(arrivals_per_s):
            signal.arrival()
        for _ in range(turned_away_per_s):
            signal.turned_away()
        clock.now += 1


def test_signal_through_the_load_profile():
    clock = Clock()
    signal = FleetSignal(idle_rate=3, window_s=10, arm_after_s=60, clock=clock)
    assert signal.state()["idle"] == 0                         # just started, nothing observed yet
    run(signal, clock, 300, 0)                                 # quiet start: never idle before a first peak
    assert signal.state() == {"saturated": 0, "idle": 0, "armed": False, "arrivals_per_s": 0.0}
    run(signal, clock, 300, 2)                                 # light load
    assert signal.state()["idle"] == 0 and signal.state()["saturated"] == 0
    run(signal, clock, 300, 21, 0)                             # busy but within capacity
    run(signal, clock, 20, 21, 1)                              # a short burst over capacity does not arm
    assert not signal.state()["armed"]
    run(signal, clock, 30, 21, 0)
    assert signal.state()["saturated"] == 0 and signal.state()["idle"] == 0
    run(signal, clock, 600, 100, 60)                           # sustained overload
    state = signal.state()
    assert state["saturated"] == 1 and state["armed"] and state["idle"] == 0
    run(signal, clock, 5, 0)                                   # load just stopped: window not yet clean
    assert signal.state()["idle"] == 0
    run(signal, clock, 6, 0)
    assert signal.state() == {"saturated": 0, "idle": 1, "armed": True, "arrivals_per_s": 0.0}
    run(signal, clock, 60, 2)                                  # one task serving the whole quiet period stays idle
    assert signal.state()["idle"] == 1
    run(signal, clock, 12, 100, 60)                            # load is back
    state = signal.state()
    assert state["idle"] == 0 and state["saturated"] == 1


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
