from dataclasses import dataclass


class VariableTracker:
    pass


@dataclass
class ConstantVariable(VariableTracker):
    value: object


@dataclass
class TensorVariable(VariableTracker):
    node: object


@dataclass
class TorchModuleVariable(VariableTracker):
    module: object


@dataclass
class UserFunctionVariable(VariableTracker):
    fn: object


@dataclass
class PythonObjectVariable(VariableTracker):
    value: object


@dataclass
class TorchOperatorVariable(VariableTracker):
    name: str
    fn: object


@dataclass
class TensorMethodVariable(VariableTracker):
    base: TensorVariable
    name: str
