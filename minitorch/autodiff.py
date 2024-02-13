from typing import Any, List


variable_count = 1


# ## Module 1

# Variable is the main class for autodifferentiation logic for scalars
# and tensors.


class Variable:
    """
    Attributes:
        history (:class:`History` or None) : the Function calls that created this variable or None if constant
        derivative (variable type): the derivative with respect to this variable
        grad (variable type) : alias for derivative, used for tensors
        name (string) : a globally unique name of the variable
    """

    def __init__(self, history, name=None):
        global variable_count
        assert history is None or isinstance(history, History), history

        self.history = history
        self._derivative = None

        # This is a bit simplistic, but make things easier.
        variable_count += 1
        self.unique_id = "Variable" + str(variable_count)

        # For debugging can have a name.
        if name is not None:
            self.name = name
        else:
            self.name = self.unique_id
        self.used = 0

    def requires_grad_(self, val):
        """
        Set the requires_grad flag to `val` on variable.

        Ensures that operations on this variable will trigger
        backpropagation.

        Args:
            val (bool): whether to require grad
        """
        if val and self.history is None:
            self.history = History()
        if not val:
            self.history = None

    def backward(self, d_output=None):
        assert isinstance(d_output, Variable)
        backpropagate(self, d_output)

    @property
    def derivative(self):
        return self._derivative

    def is_leaf(self):
        "True if this variable created by the user (no `last_fn`)"
        return self.history.last_fn is None

    def accumulate_derivative(self, val):
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            val (number): value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self._derivative is None:
            self._derivative = self.zeros()
        self._derivative += val

    def zero_derivative_(self):  # pragma: no cover
        """
        Reset the derivative on this variable.
        """
        self._derivative = self.zeros()

    def zero_grad_(self):  # pragma: no cover
        """
        Reset the derivative on this variable.
        """
        self.zero_derivative_()

    def expand(self, x):
        "Placeholder for tensor variables"
        return x

    # Helper functions for children classes.

    def __radd__(self, b):
        return self + b

    def __rmul__(self, b):
        return self * b

    def zeros(self):
        return 0.0


# Some helper functions for handling optional tuples.


def wrap_tuple(x):
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


# Classes for Functions.


class Context:
    """
    Context class is used by `Function` to store information during the forward pass.

    Attributes:
        no_grad (bool) : do not save gradient information
        saved_values (tuple) : tuple of values saved for backward pass
        saved_tensors (tuple) : alias for saved_values
    """

    def __init__(self, no_grad=False):
        self._saved_values = None
        self.no_grad = no_grad

    def save_for_backward(self, *values):
        """
        Store the given `values` if they need to be used during backpropagation.

        Args:
            values (list of values) : values to save for backward
        """
        if self.no_grad:
            return
        self._saved_values = values

    @property
    def saved_values(self):
        assert not self.no_grad, "Doesn't require grad"
        assert self._saved_values is not None, "Did you forget to save values?"
        return unwrap_tuple(self._saved_values)

    @property
    def saved_tensors(self):  # pragma: no cover
        return self.saved_values


class History:
    """
    `History` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn (:class:`FunctionBase`) : The last Function that was called.
        ctx (:class:`Context`): The context for that Function.
        inputs (list of inputs) : The inputs that were given when `last_fn.forward` was called.

    """

    def __init__(self, last_fn=None, ctx=None, inputs=None):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs

    def backprop_step(self, grad_output):
        """
        Run one step of backpropagation by calling chain rule.

        Args:
            d_output : a derivative with respect to this variable

        Returns:
            list of numbers : a derivative with respect to `inputs`
        """
        if self.last_fn is not None:
            vdlist = self.last_fn.chain_rule(self.ctx, self.inputs, grad_output)
            return vdlist
        return []


class FunctionBase:
    """
    A function that can act on :class:`Variable` arguments to
    produce a :class:`Variable` output, while tracking the internal history.

    Call by :func:`FunctionBase.apply`.

    """

    @staticmethod
    def variable(raw, history):
        # Implement by children class.
        raise NotImplementedError()

    @classmethod
    def apply(cls, *vals):
        """
        Apply is called by the user to run the Function.
        Internally it does three things:

        a) Creates a Context for the function call.
        b) Calls forward to run the function.
        c) Attaches the Context to the History of the new variable.

        There is a bit of internal complexity in our implementation
        to handle both scalars and tensors.

        Args:
            vals (list of Variables or constants) : The arguments to forward

        Returns:
            `Variable` : The new variable produced

        """
        # Go through the variables to see if any needs grad.
        raw_vals = []
        need_grad = False
        for v in vals:
            if isinstance(v, Variable):
                if v.history is not None:
                    need_grad = True
                v.used += 1
                raw_vals.append(v.get_data())
            else:
                raw_vals.append(v)

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls.forward(ctx, *raw_vals)
        assert isinstance(c, cls.data_type), "Expected return typ %s got %s" % (
            cls.data_type,
            type(c),
        )
        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = History(cls, ctx, vals)
        return cls.variable(cls.data(c), back)

    @classmethod
    def chain_rule(cls, ctx, inputs, d_output):
        """
        Implement the derivative chain-rule.

        Args:
            ctx (:class:`Context`) : The context from running forward
            inputs (list of args) : The args that were passed to :func:`FunctionBase.apply` (e.g. :math:`x, y`)
            d_output (number) : The `d_output` value in the chain rule.

        Returns:
            list of (`Variable`, number) : A list of non-constant variables with their derivatives
            (see `is_constant` to remove unneeded variables)

        """
        dlist = cls.backward(ctx, d_output)
        if not isinstance(dlist, tuple):
            dlist = (dlist,)
        return [
            (input_v, d)
            for input_v, d in zip(inputs[: len(dlist)], dlist)
            if not is_constant(input_v)
        ]
        # Tip: Note when implementing this function that
        # cls.backward may return either a value or a tuple.


# Algorithms for backpropagation


def is_constant(val):
    return not isinstance(val, Variable) or val.history is None


def topological_sort(variable: Variable) -> List[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        list of Variables : Non-constant Variables in topological order
                            starting from the right.
    """
    vis = set()
    que = [variable]
    id2cnt: dict[int, int] = dict()
    id2v = dict()
    while len(que) > 0:
        top_v = que[0]
        que = que[1:]
        id2v[top_v.unique_id] = top_v
        if not is_constant(top_v):
            if top_v.history is not None and top_v.history.inputs is not None:
                # print("top_v.history.inputs", top_v.history.inputs)
                for parent_v in top_v.history.inputs:
                    if not isinstance(parent_v, Variable):
                        continue
                    id2cnt[parent_v.unique_id] = id2cnt.get(parent_v.unique_id, 0) + 1
                    if parent_v.unique_id not in vis:
                        vis.add(parent_v.unique_id)
                        que.append(parent_v)
    _ans = []
    que = [v for i, v in id2v.items() if id2cnt.get(i, 0) == 0]
    assert variable in que
    vis = set([v.unique_id for v in que])
    while len(que) > 0:
        top_v = que[0]
        que = que[1:]
        if not is_constant(top_v):
            _ans.append(top_v)
            if top_v.history is not None and top_v.history.inputs is not None:
                for parent_v in top_v.history.inputs:
                    if not isinstance(parent_v, Variable):
                        continue
                    org_cnt = id2cnt.get(parent_v.unique_id, 0)
                    assert org_cnt > 0
                    id2cnt[parent_v.unique_id] = org_cnt - 1
                    if org_cnt == 1 and parent_v.unique_id not in vis:
                        vis.add(parent_v.unique_id)
                        que.append(parent_v)
    assert len(_ans) == len([v for v in id2v.values() if not is_constant(v)])
    return _ans


def print_out_graph(varaible):
    que = topological_sort(varaible)
    for v in que:
        res = []
        res.append(v.unique_id)
        if v.history is not None and v.history.last_fn is not None:

            res += ["<-", v.history.last_fn.__name__, "<-"]
            for p_v in v.history.inputs:
                if isinstance(p_v, Variable):
                    res += [
                        p_v.unique_id,
                        "is_constant: " + repr(p_v) if is_constant(p_v) else "",
                    ]
                else:
                    res.append(p_v)
        print(*res)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    See :doc:`backpropagate` for details on the algorithm.

    Args:
        variable (:class:`Variable`): The right-most variable
        deriv (number) : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    que = topological_sort(variable)
    # print_out_graph(variable)

    id2d = {variable.unique_id: deriv}
    for v in que:
        d = id2d[v.unique_id]
        from minitorch.tensor import Tensor

        assert (
            isinstance(d, Tensor)
            and isinstance(v, Tensor)
            and (d.size == 1 or v.shape == d.shape)
        )
        if v.is_leaf():
            v.accumulate_derivative(d)
        else:
            # print(v.history.last_fn)
            for p_v, p_d in v.history.backprop_step(d):
                assert p_v.shape == p_d.shape or p_d.size == 1
                if p_v.unique_id not in id2d:
                    id2d[p_v.unique_id] = p_d
                else:
                    id2d[p_v.unique_id] += p_d
    assert len(que) == len(id2d)
