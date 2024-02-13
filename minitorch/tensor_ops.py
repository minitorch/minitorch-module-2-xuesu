from typing import List
import numpy
import numpy.typing as numpy_typing
from .tensor_data import (
    IndexingError,
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)


def zeros_list(l: int) -> List[int]:
    return [0] * l


def big_pos2small_pos(
    big_pos: int,
    big_shape: numpy_typing.NDArray[numpy.int32],
    big_strides: numpy_typing.NDArray[numpy.int32],
    small_shape: numpy_typing.NDArray[numpy.int32],
    small_strides: numpy_typing.NDArray[numpy.int32],
):
    need_broadcast = False
    if not numpy.array_equal(small_shape, big_shape):
        if list(shape_broadcast(big_shape, small_shape)) != big_shape.tolist():
            raise IndexingError()
        need_broadcast = True

    big_dimension = len(big_shape)
    small_dimension = len(small_shape)
    big_index = zeros_list(big_dimension)
    to_index(big_pos, big_shape, big_index)
    if need_broadcast:
        small_index = zeros_list(small_dimension)
        broadcast_index(big_index, big_shape, small_shape, small_index)
    else:
        small_index = big_index
    small_pos = index_to_position(small_index, small_strides)
    assert big_pos == index_to_position(big_index, big_strides)
    assert small_pos == index_to_position(small_index, small_strides)
    return int(small_pos)


def tensor_map(fn):
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        mid_storage = [fn(v) for v in in_storage]
        for out_pos in range(len(out)):
            in_pos = big_pos2small_pos(
                out_pos, out_shape, out_strides, in_shape, in_strides
            )
            out[out_pos] = mid_storage[in_pos]

    return _map


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Simple version::

        for i:
            for j:
                out[i, j] = fn(a[i, j])

    Broadcasted version (`a` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0])

    Args:
        fn: function from float-to-float to apply.
        a (:class:`TensorData`): tensor to map over
        out (:class:`TensorData`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_map(fn)

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        for out_pos in range(len(out)):
            a_pos = big_pos2small_pos(
                out_pos, out_shape, out_strides, a_shape, a_strides
            )
            b_pos = big_pos2small_pos(
                out_pos, out_shape, out_strides, b_shape, b_strides
            )
            if out_shape.tolist() == [50, 2, 2]:
                a_index = [0] * 3
                b_index = [0] * 3
                out_index = [0] * 3
                to_index(a_pos, a_shape, a_index)
                to_index(b_pos, b_shape, b_index)
                to_index(out_pos, out_shape, out_index)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      out = fn_zip(a, b)

    Simple version ::

        for i:
            for j:
                out[i, j] = fn(a[i, j], b[i, j])

    Broadcasted version (`a` and `b` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0], b[0, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to zip over
        b (:class:`TensorData`): tensor to zip over

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_zip(fn)

    def ret(a, b):
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """

    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):
        for out_pos in range(len(out)):
            out_index = zeros_list(len(out_shape))
            to_index(out_pos, out_shape, out_index)
            assert out_index[reduce_dim] == 0
            a_index = list(out_index)
            ans = 0
            for i in range(a_shape[reduce_dim]):
                a_index[reduce_dim] = i
                a_pos = index_to_position(a_index, a_strides)
                if i == 0:
                    ans = a_storage[a_pos]
                else:
                    ans = fn(a_storage[a_pos], ans)
            out[out_pos] = ans

    return _reduce


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`TensorData` : new tensor
    """

    f = tensor_reduce(fn)

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


class TensorOps:
    map = map
    zip = zip
    reduce = reduce
