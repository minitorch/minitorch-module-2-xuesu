"""
Implementation of the autodifferentiation Functions for Tensor.
"""

from .autodiff import FunctionBase
from .tensor_ops import TensorOps
import numpy as np
from . import operators
from .tensor import Tensor
import random


# Constructors
class Function(FunctionBase):
    data_type = Tensor

    @staticmethod
    def variable(data, back):
        return Tensor(data[0], back, backend=data[1])

    @staticmethod
    def data(a):
        return (a._tensor, a.backend)


def make_tensor_constant(c, shape, backend):
    sz = 1
    for shapei in shape:
        sz *= shapei
    return Tensor.make([c] * sz, shape=shape, backend=backend)


def make_tensor_backend(tensor_ops, is_cuda=False):
    """
    Dynamically construct a tensor backend based on a `tensor_ops` object
    that implements map, zip, and reduce higher-order functions.

    Args:
        tensor_ops (:class:`TensorOps`) : tensor operations object see `tensor_ops.py`
        is_cuda (bool) : is the operations object CUDA / GPU based

    Returns :
        backend : a collection of tensor functions

    """
    # Maps
    neg_map = tensor_ops.map(operators.neg)
    sigmoid_map = tensor_ops.map(operators.sigmoid)
    relu_map = tensor_ops.map(operators.relu)
    log_map = tensor_ops.map(operators.log)
    exp_map = tensor_ops.map(operators.exp)
    id_map = tensor_ops.map(operators.id)
    inv_map = tensor_ops.map(operators.inv)

    # Zips
    add_zip = tensor_ops.zip(operators.add)
    mul_zip = tensor_ops.zip(operators.mul)
    lt_zip = tensor_ops.zip(operators.lt)
    eq_zip = tensor_ops.zip(operators.eq)
    is_close_zip = tensor_ops.zip(operators.is_close)
    relu_back_zip = tensor_ops.zip(operators.relu_back)
    log_back_zip = tensor_ops.zip(operators.log_back)
    inv_back_zip = tensor_ops.zip(operators.inv_back)

    # Reduce
    add_reduce = tensor_ops.reduce(operators.add, 0.0)
    mul_reduce = tensor_ops.reduce(operators.mul, 1.0)

    def make_sure_shape(t, shape):
        if t.shape == shape:
            return t
        return add_zip(make_tensor_constant(0, shape, t.backend), t)

    def make_sure_gradient_shape(gradient, x_shape):
        if gradient.size == 1:
            return make_tensor_constant(
                gradient._tensor._storage[0], x_shape, gradient.backend
            )
        if gradient.shape == x_shape:
            return gradient
        else:  # gradient.shape: [1,1,3,5,5], x_shape: [3,1,5], y.shape[1,1,3,5,5]
            ext_dim = len(gradient.shape) - len(x_shape)
            assert ext_dim >= 0
            for i in range(ext_dim):
                gradient = add_reduce(gradient, i)
            for i in range(ext_dim, len(gradient.shape)):
                if gradient.shape[i] != x_shape[i - ext_dim]:
                    assert (
                        gradient.shape[i] > x_shape[i - ext_dim]
                        and x_shape[i - ext_dim] == 1
                    )
                    # a is broadcasted
                    gradient = add_reduce(gradient, i)
            if ext_dim > 0:
                return Tensor.make(
                    gradient._tensor._storage,
                    gradient._tensor.shape[ext_dim:],
                    gradient._tensor.strides[ext_dim:],
                    backend=gradient.backend,
                ).contiguous()
        return gradient

    class Backend:
        cuda = is_cuda
        _id_map = id_map
        _add_reduce = add_reduce

        class Neg(Function):
            @staticmethod
            def forward(ctx, t1):
                return neg_map(t1)

            @staticmethod
            def backward(ctx, grad_output):
                return neg_map(grad_output)

        class Inv(Function):
            @staticmethod
            def forward(ctx, t1):
                ctx.save_for_backward(t1)
                return inv_map(t1)

            @staticmethod
            def backward(ctx, grad_output):
                t1 = ctx.saved_values
                return inv_back_zip(t1, grad_output)

        class Add(Function):
            @staticmethod
            def forward(ctx, t1, t2):
                ctx.save_for_backward(t1.shape, t2.shape)
                return add_zip(t1, t2)

            @staticmethod
            def backward(ctx, grad_output):
                t1_shape, t2_shape = ctx.saved_values
                t1_grad, t2_grad = make_sure_gradient_shape(
                    grad_output, t1_shape
                ), make_sure_gradient_shape(grad_output, t2_shape)
                return t1_grad, t2_grad

        class Mul(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return mul_zip(a, b)

            @staticmethod
            def backward(ctx, grad_output):
                a, b = ctx.saved_values
                return make_sure_gradient_shape(
                    mul_zip(grad_output, b), a.shape
                ), make_sure_gradient_shape(mul_zip(grad_output, a), b.shape)

        class Sigmoid(Function):
            @staticmethod
            def forward(ctx, a):
                ctx.save_for_backward(a)
                return sigmoid_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                a = ctx.saved_values
                s = sigmoid_map(a)
                _ans = mul_zip(
                    mul_zip(
                        s,
                        add_zip(
                            make_tensor_constant(1.0, a.shape, a.backend),
                            neg_map(s),
                        ),
                    ),
                    grad_output,
                )
                return _ans

        class ReLU(Function):
            @staticmethod
            def forward(ctx, a):
                ctx.save_for_backward(a)
                return relu_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                a = ctx.saved_values
                return relu_back_zip(a, grad_output)

        class Log(Function):
            @staticmethod
            def forward(ctx, a):
                ctx.save_for_backward(a)
                return log_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                a = ctx.saved_values
                return log_back_zip(a, grad_output)

        class Exp(Function):
            @staticmethod
            def forward(ctx, a):
                ctx.save_for_backward(a)
                return exp_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                a = ctx.saved_values
                return mul_zip(exp_map(a), grad_output)

        class Sum(Function):
            @staticmethod
            def forward(ctx, a, dim=None):
                ctx.save_for_backward(a.shape, dim)
                if dim is not None:
                    return add_reduce(a, dim)
                else:
                    return add_reduce(
                        a.contiguous().view(int(operators.prod(a.shape))), 0
                    )

            @staticmethod
            def backward(ctx, grad_output):
                a_shape, dim = ctx.saved_values
                if dim is None:
                    out = grad_output.zeros(a_shape)
                    out._tensor._storage[:] = grad_output[0]
                    return out
                else:
                    _ans = add_zip(
                        make_tensor_constant(0.0, a_shape, backend=grad_output.backend),
                        grad_output,
                    )
                    return _ans

        class All(Function):
            @staticmethod
            def forward(ctx, a, dim):
                if dim is not None:
                    return mul_reduce(a, dim)
                else:
                    return mul_reduce(
                        a.contiguous().view(int(operators.prod(a.shape))), 0
                    )

        class LT(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return lt_zip(a, b)

            @staticmethod
            def backward(ctx, grad_output):
                a, b = ctx.saved_values
                return make_tensor_constant(
                    0, a.shape, a.backend
                ), make_tensor_constant(0, b.shape, b.backend)

        class EQ(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return eq_zip(a, b)

            @staticmethod
            def backward(ctx, grad_output):
                a, b = ctx.saved_values
                return make_tensor_constant(
                    0, a.shape, a.backend
                ), make_tensor_constant(0, b.shape, b.backend)

        class IsClose(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return is_close_zip(a, b)

        class Permute(Function):
            @staticmethod
            def forward(ctx, a, order):
                ctx.save_for_backward(a, order)
                need_permute = not all(i == o for i, o in enumerate(order))
                if need_permute:
                    _ans = a._new(a._tensor.permute(*order))
                    # print("forward", "order", order, "_ans", _ans.shape, _ans, "a", a.unique_id, a.shape, a)
                    return _ans
                else:
                    # print("forward", "order", order, "a", a.unique_id, a.shape, a)
                    return a

            @staticmethod
            def backward(ctx, grad_output):
                (a, order) = ctx.saved_values
                # print("backward", "order", order, "a", a.unique_id, a.shape, a)

                after_ind_org_ind_prs = [(o, i) for i, o in enumerate(order)]
                after_ind_org_ind_prs.sort()
                back_order = [pr[1] for pr in after_ind_org_ind_prs]
                need_permute = not all(i == o for i, o in enumerate(back_order))
                if need_permute:
                    if grad_output.size == 1:
                        return grad_output

                    assert a.size == grad_output.size
                    _ans = grad_output._new(grad_output._tensor.permute(*back_order))
                    # print("backward", "order", order, "back_order", back_order, "grad_output", grad_output.shape, grad_output, "_ans", _ans.shape, _ans, "a", a.unique_id, a.shape, a)
                    assert _ans.shape == a.shape
                    return _ans
                else:
                    return grad_output

        class View(Function):
            @staticmethod
            def forward(ctx, a, shape):
                ctx.save_for_backward(a, shape)
                assert a._tensor.is_contiguous(), "Must be contiguous to view"
                return Tensor.make(a._tensor._storage, shape, backend=a.backend)

            @staticmethod
            def backward(ctx, grad_output):
                (a, _) = ctx.saved_values
                _ans = Tensor.make(
                    grad_output._tensor._storage, a.shape, backend=grad_output.backend
                )
                return _ans

        class Copy(Function):
            @staticmethod
            def forward(ctx, a):
                return id_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        class MatMul(Function):
            @staticmethod
            def forward(ctx, t1, t2):
                ctx.save_for_backward(t1, t2)
                return tensor_ops.matrix_multiply(t1, t2)

            @staticmethod
            def backward(ctx, grad_output):
                t1, t2 = ctx.saved_values

                def transpose(a):
                    order = list(range(a.dims))
                    order[-2], order[-1] = order[-1], order[-2]
                    return a._new(a._tensor.permute(*order))

                return (
                    tensor_ops.matrix_multiply(grad_output, transpose(t2)),
                    tensor_ops.matrix_multiply(transpose(t1), grad_output),
                )

    return Backend


TensorFunctions = make_tensor_backend(TensorOps)


# Helpers for Constructing tensors
def zeros(shape, backend=TensorFunctions):
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend

    Returns:
        :class:`Tensor` : new tensor
    """
    return Tensor.make([0] * int(operators.prod(shape)), shape, backend=backend)


def rand(shape, backend=TensorFunctions, requires_grad=False):
    """
    Produce a random tensor of size `shape`.

    Args:
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(ls, shape=None, backend=TensorFunctions, requires_grad=False):
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls (list): data for tensor
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    tensor = Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(ls, backend=TensorFunctions, requires_grad=False):
    """
    Produce a tensor with data and shape from ls

    Args:
        ls (list): data for tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def shape(ls):
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls):
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape = shape(ls)
    return _tensor(cur, tuple(shape), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(f, *vals, arg=0, epsilon=1e-6, ind=None):
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f, *vals):
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        if abs(x.grad[ind] - check) > 1e-3:
            print("catch it")
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
