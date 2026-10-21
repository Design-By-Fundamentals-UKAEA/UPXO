"""First function run in every worker process of a WorkerPool; imports nothing heavy itself.

A spawned worker unpickles its start-up message before it runs anything. When that message
names a stage module as the initializer, the worker imports that module (numpy, scipy, ...)
while the parent is still writing the rest of the message into the pipe, and the parent waits
for it. Naming this module instead keeps the message cheap to read: the stage module is
imported afterwards, by ``initialize``, once the parent has been released.
"""
import importlib


def initialize(module, qualname, args):
    """Import ``module``, find ``qualname`` in it and call it with ``args``."""
    target = importlib.import_module(module)
    for part in qualname.split('.'):
        target = getattr(target, part)
    target(*args)
