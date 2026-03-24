"""
Autograd Function base class.

Mirrors PyTorch 0.1's torch.autograd.Function:
  - User subclasses override forward() and backward()
  - __call__ runs forward, builds computation graph
  - save_for_backward / saved_tensors for caching
"""


class Function:
    """
    Base class for differentiable operations.

    Subclasses must implement:
      forward(self, *inputs)  → Tensor (raw, not Variable)
      backward(self, grad_output) → tuple of Tensor gradients
    """

    def __call__(self, *variables):
        from .variable import Variable

        # Determine if any input requires grad and none is volatile
        any_volatile = any(v.volatile for v in variables if isinstance(v, Variable))
        any_requires_grad = any(
            v.requires_grad for v in variables if isinstance(v, Variable)
        )

        # Extract raw tensor data from Variables
        raw_inputs = []
        for v in variables:
            if isinstance(v, Variable):
                raw_inputs.append(v.data)
            else:
                raw_inputs.append(v)

        # Initialize saved state
        self._saved_tensors = []
        self._input_variables = variables

        # Run forward
        raw_output = self.forward(*raw_inputs)

        # Wrap output in Variable
        if any_volatile:
            # Volatile: no graph, no grad
            result = Variable(raw_output, volatile=True)
            return result

        requires_grad = any_requires_grad
        result = Variable(raw_output, requires_grad=requires_grad)
        if requires_grad:
            result.creator = self
            # Store references to input variables for backward graph traversal
            self._prev_variables = variables
        return result

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def save_for_backward(self, *tensors):
        """Save tensors for use in backward(). Called during forward().
        Records each tensor's version counter so that in-place modifications
        can be detected at backward time (mirrors PyTorch 0.1.11 VariableVersion)."""
        self._saved_tensors = [(t, t._version) for t in tensors]

    @property
    def saved_tensors(self):
        result = []
        for tensor, expected_version in self._saved_tensors:
            if tensor._version != expected_version:
                raise RuntimeError(
                    "one of the variables needed for gradient computation "
                    "has been modified by an inplace operation")
            result.append(tensor)
        return tuple(result)
