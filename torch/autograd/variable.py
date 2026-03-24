"""
Autograd Variable — wraps a Tensor with gradient tracking.

Mirrors PyTorch 0.1's torch.autograd.Variable:
  - .data       → underlying Tensor
  - .grad       → gradient (Variable or None)
  - .requires_grad
  - .volatile
  - .creator    → the Function that produced this Variable
  - .backward() → run backward pass through computation graph
  - .register_hook() → register gradient hook
"""
from ..tensor import Tensor, _ones


class _SumFunction:
    """Built-in autograd function for Variable.sum()."""

    def __init__(self, input_var):
        self._prev_variables = (input_var,)
        self._input_shape = list(input_var.data.size())

    def backward(self, grad_output):
        # d(sum)/d(x_i) = 1 for all i, scaled by grad_output
        grad = _ones(self._input_shape)
        if grad_output.numel() == 1:
            scale = float(grad_output)
            grad = grad * scale
        else:
            grad = grad * grad_output
        return (grad,)


class _MulFunction:
    """Built-in autograd function for Variable * Variable (element-wise)."""

    def __init__(self, var_a, var_b):
        self._prev_variables = (var_a, var_b)
        self._a_data = var_a.data
        self._b_data = var_b.data

    def backward(self, grad_output):
        # d(a*b)/da = b, d(a*b)/db = a
        grad_a = grad_output * self._b_data
        grad_b = grad_output * self._a_data
        return (grad_a, grad_b)


class Variable:
    """Wraps a Tensor with autograd capability."""

    def __init__(self, data, requires_grad=False, volatile=False):
        if isinstance(data, Tensor):
            self.data = data
        else:
            # Assume it's something Tensor can wrap
            self.data = Tensor(data)
        self.requires_grad = requires_grad
        self.volatile = volatile
        self.grad = None
        self.creator = None  # The Function that produced this
        self._hooks = []

    def register_hook(self, fn):
        """Register a hook called with the gradient during backward."""
        self._hooks.append(fn)

    def backward(self, grad_output=None, retain_variables=False):
        """
        Run backward pass from this variable.

        For scalar outputs, grad_output defaults to 1.0.
        Accumulates gradients into .grad for all leaf variables with requires_grad.
        """
        if grad_output is None:
            # Default: scalar loss, gradient = 1.0
            grad_output = _ones(list(self.data.size()))
        elif isinstance(grad_output, Variable):
            grad_output = grad_output.data
        elif not isinstance(grad_output, Tensor):
            grad_output = Tensor(grad_output)

        _backward_engine(self, grad_output, retain_variables)

    def sum(self):
        """Convenience: sum all elements, returns scalar Variable with graph."""
        result_data = self.data.sum()
        result = Variable(result_data, requires_grad=self.requires_grad)
        if self.requires_grad and not self.volatile:
            fn = _SumFunction(self)
            result.creator = fn
        return result

    def __mul__(self, other):
        """Element-wise multiply two Variables."""
        if isinstance(other, Variable):
            result_data = self.data * other.data
            requires_grad = self.requires_grad or other.requires_grad
            result = Variable(result_data, requires_grad=requires_grad)
            if requires_grad:
                fn = _MulFunction(self, other)
                result.creator = fn
            return result
        # scalar multiply
        result_data = self.data * other
        result = Variable(result_data, requires_grad=self.requires_grad)
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"Variable(data={self.data}, requires_grad={self.requires_grad})"


def _backward_engine(root_var, grad_output, retain_variables):
    """
    Backward engine: reverse topological traversal of the computation graph.

    Algorithm:
      1. DFS from root_var to discover all Function nodes and compute
         how many downstream consumers each Variable has (in-degree).
      2. Start from root_var, push gradient through its creator.
      3. For each input Variable of a Function, accumulate gradient.
         When all downstream grads have arrived (in-degree drops to 0),
         process that Variable (either store leaf grad or continue backward).
    """
    # --- Phase 1: discover graph, count how many times each Variable is consumed ---
    # in_degree[id(var)] = number of downstream Functions that consume this var
    in_degree = {}
    # var_map[id(var)] = var (to keep references alive)
    var_map = {}

    def _discover(var):
        vid = id(var)
        if vid in var_map:
            return
        var_map[vid] = var
        in_degree[vid] = 0
        if var.creator is not None:
            for prev in var.creator._prev_variables:
                if isinstance(prev, Variable) and prev.requires_grad:
                    _discover(prev)

    _discover(root_var)

    # Count in-degrees: for each Function, its input variables get +1
    def _count(var):
        if var.creator is not None:
            for prev in var.creator._prev_variables:
                if isinstance(prev, Variable) and prev.requires_grad:
                    in_degree[id(prev)] = in_degree.get(id(prev), 0) + 1

    visited_fn = set()

    def _count_all(var):
        if var.creator is not None:
            fn_id = id(var.creator)
            if fn_id in visited_fn:
                return
            visited_fn.add(fn_id)
            for prev in var.creator._prev_variables:
                if isinstance(prev, Variable) and prev.requires_grad:
                    in_degree[id(prev)] = in_degree.get(id(prev), 0) + 1
                    _count_all(prev)

    _count_all(root_var)

    # --- Phase 2: backward traversal using ready queue ---
    # grad_acc[id(var)] = (Tensor, borrowed)
    #   borrowed=True means the tensor is not owned (came directly from backward output),
    #   must clone before in-place mutation. Mirrors PyTorch 0.1.11 GradBuffer copy-on-write.
    grad_acc = {id(root_var): (grad_output, True)}
    # ready queue: variables whose in_degree has reached 0
    ready = [root_var]

    while ready:
        var = ready.pop(0)
        vid = id(var)
        grad, _ = grad_acc[vid]

        if var.creator is None:
            # Leaf variable
            if var.requires_grad:
                _accumulate_grad(var, grad)
                for hook in var._hooks:
                    hook(Variable(grad))
            continue

        fn = var.creator
        if not retain_variables:
            var.creator = None

        # Call backward
        grad_inputs = fn.backward(grad)
        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)

        # Distribute gradients to input variables
        for prev_var, g in zip(fn._prev_variables, grad_inputs):
            if not isinstance(prev_var, Variable):
                continue
            if not prev_var.requires_grad:
                continue
            if g is None:
                continue

            pvid = id(prev_var)
            # Accumulate gradient with copy-on-write
            if pvid in grad_acc:
                existing, borrowed = grad_acc[pvid]
                if borrowed:
                    # First accumulation: clone to get a private buffer
                    existing = existing.clone()
                existing += g  # in-place addition
                grad_acc[pvid] = (existing, False)
            else:
                grad_acc[pvid] = (g, True)  # borrowed — don't mutate original

            # Decrement in-degree; when 0, all downstream grads have arrived
            in_degree[pvid] -= 1
            if in_degree[pvid] == 0:
                ready.append(prev_var)


def _accumulate_grad(var, grad):
    """Accumulate gradient into var.grad (supports repeated backward calls).
    Uses in-place addition when grad already exists, avoiding new Tensor allocation.
    Mirrors PyTorch 0.1.11 Variable::backward."""
    if var.grad is None:
        var.grad = Variable(grad.clone())
    else:
        var.grad.data += grad  # in-place accumulation
