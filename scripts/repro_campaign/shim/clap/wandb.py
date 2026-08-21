"""REPRO campaign: offline stub for `wandb`.

`third_party/CLAP` calls `wandb.login()`, `wandb.init(...)` and `wandb.log(...)`
unconditionally from the server and the client.  The campaign runs offline and
records its own logs, so this module is put ahead of the real package on
PYTHONPATH.  It has no side effects beyond a one-line print at init.
"""


def login(*a, **k):
    return True


class _Run:
    def log(self, *a, **k):
        return None

    def finish(self, *a, **k):
        return None


_run = _Run()


def init(*a, **k):
    print("[wandb-stub] init (offline, no-op)")
    return _run


def log(*a, **k):
    return None


def finish(*a, **k):
    return None


def Image(*a, **k):
    return None


def Table(*a, **k):
    return None


class run:  # some code paths poke wandb.run
    id = "offline"
