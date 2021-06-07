from contextlib import contextmanager


class Config:
    enable_backprop = True
    train = True


@contextmanager
def using_config(name, value):
    old_val = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_val)


def no_grad():
    return using_config("enable_backprop", False)


def test_mode():
    return using_config("train", False)
