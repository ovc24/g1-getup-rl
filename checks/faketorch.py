"""A tiny numpy stand-in for the subset of torch used by getup/*.py (checks/getup_check.py only)."""
import builtins
import math as _math
from collections import namedtuple

import numpy as np

_rng = np.random.default_rng(0)


class _DType:
    def __init__(self, np_dtype, name):
        self.np = np.dtype(np_dtype)
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"


bool = _DType(np.bool_, "bool")  # noqa: A001
float32 = _DType(np.float32, "float32")
float = float32  # noqa: A001
float64 = _DType(np.float64, "float64")
int64 = _DType(np.int64, "int64")
long = int64
int32 = _DType(np.int32, "int32")
int = int32  # noqa: A001
_DT = {np.dtype(np.bool_): bool, np.dtype(np.float32): float32, np.dtype(np.float64): float64,
       np.dtype(np.int64): int64, np.dtype(np.int32): int32}


def _np_dtype(d):
    if d is None:
        return None
    if isinstance(d, _DType):
        return d.np
    return np.dtype(d)


def _arr(x):
    if isinstance(x, Tensor):
        return x.a
    return x


def _idx(i):
    if isinstance(i, tuple):
        return tuple(_idx(j) for j in i)
    if isinstance(i, Tensor):
        return i.a
    return i


def _wrap(a):
    a = np.asarray(a)
    if a.dtype == np.float64:
        a = a.astype(np.float32)
    return Tensor(a)


MaxRet = namedtuple("MaxRet", "values indices")


class Tensor:
    __array_priority__ = 1000

    def __init__(self, a):
        self.a = np.asarray(a)

    # -- info
    @property
    def shape(self):
        return tuple(self.a.shape)

    @property
    def dtype(self):
        return _DT.get(self.a.dtype, self.a.dtype)

    @property
    def device(self):
        return "cpu"

    def dim(self):
        return self.a.ndim

    def numel(self):
        return self.a.size

    def size(self, d=None):
        return self.a.shape if d is None else self.a.shape[d]

    def __len__(self):
        return len(self.a)

    def __repr__(self):
        return f"faketensor({self.a!r})"

    # -- conversion
    def __bool__(self):
        if self.a.size != 1:
            raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
        return builtins.bool(self.a.reshape(-1)[0])

    def __int__(self):
        return builtins.int(self.a.reshape(-1)[0]) if self.a.size == 1 else (_ for _ in ()).throw(
            RuntimeError("int() of a multi-element tensor"))

    def __float__(self):
        if self.a.size != 1:
            raise RuntimeError("float() of a multi-element tensor")
        return builtins.float(self.a.reshape(-1)[0])

    def __index__(self):
        return self.__int__()

    def item(self):
        return self.a.reshape(-1)[0].item()

    def tolist(self):
        return self.a.tolist()

    def numpy(self):
        return self.a

    def cpu(self):
        return self

    def to(self, *args, **kw):
        dt = kw.get("dtype")
        for a in args:
            if isinstance(a, _DType):
                dt = a
        if dt is None:
            return self
        return Tensor(self.a.astype(_np_dtype(dt)))

    def float(self):
        return Tensor(self.a.astype(np.float32))

    def long(self):
        return Tensor(self.a.astype(np.int64))

    def bool(self):
        return Tensor(self.a.astype(np.bool_))

    def clone(self):
        return Tensor(np.array(self.a, copy=True))

    def detach(self):
        return self

    # -- indexing
    def __getitem__(self, i):
        return Tensor(self.a[_idx(i)])

    def __setitem__(self, i, v):
        self.a[_idx(i)] = _arr(v)

    def __iter__(self):
        for row in self.a:
            yield Tensor(row)

    # -- arithmetic
    def _bin(self, o, f):
        return _wrap_keep(f(self.a, _arr(o)), self, o)

    def __add__(self, o): return self._bin(o, np.add)
    def __radd__(self, o): return self._bin(o, lambda a, b: np.add(b, a))
    def __sub__(self, o): return self._bin(o, np.subtract)
    def __rsub__(self, o): return self._bin(o, lambda a, b: np.subtract(b, a))
    def __mul__(self, o): return self._bin(o, np.multiply)
    def __rmul__(self, o): return self._bin(o, lambda a, b: np.multiply(b, a))
    def __truediv__(self, o): return _wrap(np.true_divide(self.a, _arr(o)))
    def __rtruediv__(self, o): return _wrap(np.true_divide(_arr(o), self.a))
    def __mod__(self, o): return self._bin(o, np.mod)
    def __matmul__(self, o): return self._bin(o, np.matmul)
    def __rmatmul__(self, o): return self._bin(o, lambda a, b: np.matmul(b, a))
    def __pow__(self, o): return self._bin(o, np.power)
    def __neg__(self): return Tensor(-self.a)
    def __abs__(self): return Tensor(np.abs(self.a))
    def __lt__(self, o): return Tensor(self.a < _arr(o))
    def __le__(self, o): return Tensor(self.a <= _arr(o))
    def __gt__(self, o): return Tensor(self.a > _arr(o))
    def __ge__(self, o): return Tensor(self.a >= _arr(o))
    def __eq__(self, o): return Tensor(self.a == _arr(o))
    def __ne__(self, o): return Tensor(self.a != _arr(o))
    def __and__(self, o): return Tensor(self.a & _arr(o))
    def __or__(self, o): return Tensor(self.a | _arr(o))
    def __xor__(self, o): return Tensor(self.a ^ _arr(o))
    def __invert__(self):
        if self.a.dtype != np.bool_:
            raise TypeError("~ only on bool tensors here")
        return Tensor(~self.a)

    def __iadd__(self, o):
        self.a += _arr(o)
        return self

    def __isub__(self, o):
        self.a -= _arr(o)
        return self

    __hash__ = object.__hash__

    # -- elementwise
    def square(self): return _wrap_keep(np.square(self.a), self)
    def sqrt(self): return _wrap_keep(np.sqrt(self.a), self)
    def exp(self): return _wrap_keep(np.exp(self.a), self)
    def pow(self, e): return _wrap_keep(np.power(self.a, _arr(e)), self)

    # -- reductions / shape ops
    def abs(self): return Tensor(np.abs(self.a))
    def sum(self, dim=None, keepdim=False): return _wrap_keep(self.a.sum(axis=dim, keepdims=keepdim), self)
    def mean(self, dim=None, keepdim=False): return _wrap(self.a.mean(axis=dim, keepdims=keepdim))
    def any(self, dim=None): return Tensor(self.a.any(axis=dim))
    def all(self, dim=None): return Tensor(self.a.all(axis=dim))

    def max(self, dim=None):
        if dim is None:
            return Tensor(self.a.max())
        return MaxRet(Tensor(self.a.max(axis=dim)), Tensor(self.a.argmax(axis=dim)))

    def amax(self, dim=None):
        return Tensor(self.a.max(axis=dim))

    def amin(self, dim=None):
        return Tensor(self.a.min(axis=dim))

    def transpose(self, d0, d1):
        return Tensor(np.swapaxes(self.a, d0, d1))

    def clamp(self, min=None, max=None):  # noqa: A002
        a = self.a
        if min is not None:
            a = np.maximum(a, _arr(min))
        if max is not None:
            a = np.minimum(a, _arr(max))
        return Tensor(np.asarray(a).astype(self.a.dtype, copy=False))

    def clamp_(self, min=None, max=None):  # noqa: A002
        self.a[...] = self.clamp(min, max).a
        return self

    def norm(self, dim=None, keepdim=False):
        return _wrap(np.linalg.norm(self.a, axis=dim, keepdims=keepdim))

    def unsqueeze(self, d): return Tensor(np.expand_dims(self.a, d))

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(self.a.shape[i - (len(shape) - self.a.ndim)] if s == -1 else s
                      for i, s in enumerate(shape))
        return Tensor(np.broadcast_to(self.a, shape))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(self.a.reshape(shape))

    view = reshape

    def flatten(self, start_dim=0):
        if start_dim == 0:
            return Tensor(self.a.reshape(-1))
        return Tensor(self.a.reshape(self.a.shape[:start_dim] + (-1,)))

    def nonzero(self, as_tuple=False):
        return Tensor(np.argwhere(self.a))

    def unbind(self, d=0):
        return tuple(Tensor(x) for x in np.moveaxis(self.a, d, 0))

    def uniform_(self, lo=0.0, hi=1.0):
        if lo > hi:
            raise RuntimeError(f"uniform_ expects from <= to, got {lo} > {hi}")
        self.a[...] = _rng.uniform(lo, hi, self.a.shape)
        return self

    def fill_(self, v):
        self.a[...] = _arr(v)
        return self

    def copy_(self, src):
        self.a[...] = _arr(src)
        return self


def _wrap_keep(res, *ops):
    res = np.asarray(res)
    dts = [o.a.dtype for o in ops if isinstance(o, Tensor)]
    if res.dtype == np.float64 and all(d != np.float64 for d in dts):
        res = res.astype(np.float32)
    return Tensor(res)


# -- creation ----------------------------------------------------------------------------
def _shape(args):
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(args[0])
    return tuple(args)


def zeros(*shape, device=None, dtype=None):
    return Tensor(np.zeros(_shape(shape), dtype=_np_dtype(dtype) or np.float32))


def ones(*shape, device=None, dtype=None):
    return Tensor(np.ones(_shape(shape), dtype=_np_dtype(dtype) or np.float32))


def empty(*shape, device=None, dtype=None):
    return Tensor(np.zeros(_shape(shape), dtype=_np_dtype(dtype) or np.float32))


def full(shape, value, device=None, dtype=None):
    dt = _np_dtype(dtype)
    if dt is None:
        dt = np.bool_ if isinstance(value, builtins.bool) else (
            np.int64 if isinstance(value, builtins.int) else np.float32)
    return Tensor(np.full(tuple(shape), value, dtype=dt))


def zeros_like(t, dtype=None):
    return Tensor(np.zeros_like(t.a, dtype=_np_dtype(dtype)))


def ones_like(t, dtype=None):
    return Tensor(np.ones_like(t.a, dtype=_np_dtype(dtype)))


def full_like(t, v, dtype=None):
    return Tensor(np.full_like(t.a, v, dtype=_np_dtype(dtype)))


def tensor(data, device=None, dtype=None):
    a = np.array(_arr(data))
    dt = _np_dtype(dtype)
    if dt is not None:
        a = a.astype(dt)
    elif a.dtype == np.float64:
        a = a.astype(np.float32)
    return Tensor(a)


def arange(*args, device=None, dtype=None):
    return Tensor(np.arange(*args, dtype=_np_dtype(dtype) or np.int64))


def eye(n, device=None, dtype=None):
    return Tensor(np.eye(n, dtype=_np_dtype(dtype) or np.float32))


def rand(*shape, device=None):
    return Tensor(_rng.random(_shape(shape)).astype(np.float32))


def randn(*shape, device=None):
    return Tensor(_rng.standard_normal(_shape(shape)).astype(np.float32))


# -- ops ---------------------------------------------------------------------------------
def _operand(x, ref_dtype):
    if isinstance(x, Tensor):
        return x.a
    if isinstance(x, builtins.bool):
        return np.bool_(x)
    if isinstance(x, builtins.float):
        return np.float32(x) if ref_dtype in (None, np.float32) or np.issubdtype(ref_dtype, np.integer) \
            else np.asarray(x, dtype=ref_dtype)
    if isinstance(x, builtins.int):
        return np.int64(x) if ref_dtype is None or np.issubdtype(ref_dtype, np.integer) else \
            np.asarray(x, dtype=ref_dtype)
    return x


def where(cond, a, b):
    ref = None
    for x in (a, b):
        if isinstance(x, Tensor):
            ref = x.a.dtype
    return Tensor(np.where(_arr(cond), _operand(a, ref), _operand(b, ref)))


def stack(seq, dim=0):
    return Tensor(np.stack([_arr(s) for s in seq], axis=dim))


def cat(seq, dim=0):
    return Tensor(np.concatenate([_arr(s) for s in seq], axis=dim))


def maximum(a, b): return Tensor(np.maximum(_arr(a), _arr(b)))
def minimum(a, b): return Tensor(np.minimum(_arr(a), _arr(b)))
def atan2(y, x): return _wrap_keep(np.arctan2(_arr(y), _arr(x)), y, x)
def cos(x): return _wrap_keep(np.cos(_arr(x)), x)
def sin(x): return _wrap_keep(np.sin(_arr(x)), x)
def acos(x): return _wrap_keep(np.arccos(_arr(x)), x)
def sqrt(x): return _wrap_keep(np.sqrt(_arr(x)), x)
def exp(x): return _wrap_keep(np.exp(_arr(x)), x)
def log1p(x): return _wrap_keep(np.log1p(_arr(x)), x)
def floor(x): return Tensor(np.floor(_arr(x)))
def sign(x): return Tensor(np.sign(_arr(x)))
def abs(x): return Tensor(np.abs(_arr(x)))  # noqa: A001
def remainder(a, b): return _wrap_keep(np.mod(_arr(a), _arr(b)), a)
def rad2deg(x): return _wrap_keep(np.rad2deg(_arr(x)), x)
def isnan(x): return Tensor(np.isnan(_arr(x)))
def isfinite(x): return Tensor(np.isfinite(_arr(x)))


def multinomial(probs, k, replacement=False):
    p = np.asarray(_arr(probs), dtype=np.float64)
    return Tensor(_rng.choice(len(p), size=k, replace=replacement, p=p / p.sum()).astype(np.int64))
def broadcast_to(t, shape): return Tensor(np.broadcast_to(_arr(t), tuple(shape)))
def einsum(eq, *ops): return _wrap(np.einsum(eq, *[_arr(o) for o in ops]))
def count_nonzero(x): return Tensor(np.count_nonzero(_arr(x)))
def mean(x): return _wrap(np.mean(_arr(x)))
def quantile(x, q): return Tensor(np.quantile(_arr(x), q))


class _Linalg:
    @staticmethod
    def norm(x, dim=None, keepdim=False):
        return _wrap(np.linalg.norm(_arr(x), axis=dim, keepdims=keepdim))


linalg = _Linalg()
pi = _math.pi


class no_grad:  # noqa: N801  (no autograd here: a do-nothing context manager)
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Parameter(Tensor):
    """torch.nn.Parameter stand-in: a Tensor whose .data reads and replaces the values."""

    def __init__(self, data):
        super().__init__(_arr(data))

    @property
    def data(self):
        return Tensor(self.a)

    @data.setter
    def data(self, value):
        self.a = np.asarray(_arr(value))


class _NN:
    Parameter = _Parameter


nn = _NN()
