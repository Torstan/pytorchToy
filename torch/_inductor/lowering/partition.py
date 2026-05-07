"""
Partitioned graph lowering for the toy inductor backend.
"""

from dataclasses import dataclass

from torch._compile.ops import POINTWISE_TARGETS, run_eager_target, target_name
from torch._inductor.lowering.pointwise import (
    PointwiseLoweringError,
    lower_pointwise_graph,
)
from torch._inductor.lowering.single_op import try_compile_single_op


@dataclass
class CompiledRegion:
    start_index: int
    end_index: int
    input_nodes: list
    output_node: object
    compiled_kernel: object

    def run(self, env):
        args = [env[node.name] for node in self.input_nodes]
        env[self.output_node.name] = self.compiled_kernel.run(args)


@dataclass
class CompiledOpStep:
    target: str
    input_nodes: list
    output_node: object
    compiled_kernel: object

    def run(self, env):
        args = [env[node.name] for node in self.input_nodes]
        env[self.output_node.name] = self.compiled_kernel.run(args)


@dataclass
class CompiledGraph:
    placeholders: list
    steps: list
    output_value: object

    def run(self, args):
        from torch.fx.graph import resolve

        if len(args) != len(self.placeholders):
            raise RuntimeError(
                f"compiled graph expected {len(self.placeholders)} inputs, got {len(args)}"
            )

        env = {}
        for node, value in zip(self.placeholders, args):
            env[node.name] = value

        for step in self.steps:
            if isinstance(step, (CompiledRegion, CompiledOpStep)):
                step.run(env)
            else:
                env[step.name] = _run_call_function_node(step, env)

        return resolve(self.output_value, env)


def compile_graph_module(graph_module, example_inputs, *, allow_requires_grad=False):
    try:
        return lower_pointwise_graph(
            graph_module,
            example_inputs,
            allow_requires_grad=allow_requires_grad,
        ).compile()
    except PointwiseLoweringError:
        compiled_graph = _compile_partitioned_graph(
            graph_module,
            example_inputs,
            allow_requires_grad=allow_requires_grad,
        )
        if compiled_graph is None:
            raise
        return compiled_graph


def _compile_partitioned_graph(graph_module, example_inputs, *, allow_requires_grad=False):
    graph = graph_module.graph
    placeholders = []
    steps = []
    users = _build_users(graph.nodes)
    env_example = {}
    compiled_any = False
    output_value = None
    arg_idx = 0
    idx = 0

    while idx < len(graph.nodes):
        node = graph.nodes[idx]

        if node.op == "placeholder":
            if arg_idx >= len(example_inputs):
                return None
            placeholders.append(node)
            env_example[node.name] = example_inputs[arg_idx]
            arg_idx += 1
            idx += 1
            continue

        if node.op == "call_function":
            region = _try_compile_region(
                graph.nodes,
                idx,
                users,
                env_example,
                allow_requires_grad=allow_requires_grad,
            )
            if region is not None:
                compiled_any = True
                region.run(env_example)
                steps.append(region)
                idx = region.end_index + 1
                continue

            compiled_step = try_compile_single_op(node, env_example)
            if compiled_step is not None:
                compiled_any = True
                compiled_step.run(env_example)
                steps.append(compiled_step)
                idx += 1
                continue

            env_example[node.name] = _run_call_function_node(node, env_example)
            steps.append(node)
            idx += 1
            continue

        if node.op == "output":
            if len(node.args) != 1:
                return None
            output_value = node.args[0]
            idx += 1
            continue

        return None

    if not compiled_any or output_value is None:
        return None
    return CompiledGraph(placeholders, steps, output_value)


def _try_compile_region(nodes, start_index, users, env_example, *, allow_requires_grad=False):
    start_node = nodes[start_index]
    if start_node.op != "call_function" or target_name(start_node.target) not in POINTWISE_TARGETS:
        return None

    max_end_index = start_index
    while max_end_index + 1 < len(nodes):
        next_node = nodes[max_end_index + 1]
        if next_node.op != "call_function" or target_name(next_node.target) not in POINTWISE_TARGETS:
            break
        max_end_index += 1

    for end_index in range(max_end_index, start_index - 1, -1):
        region_nodes = nodes[start_index:end_index + 1]
        if not _region_has_single_output(region_nodes, users):
            continue
        try:
            region_graph_module, region_inputs, input_nodes = _build_region_graph_module(
                region_nodes, env_example
            )
            compiled_kernel = lower_pointwise_graph(
                region_graph_module,
                region_inputs,
                allow_requires_grad=allow_requires_grad,
            ).compile()
        except PointwiseLoweringError:
            continue

        return CompiledRegion(
            start_index=start_index,
            end_index=end_index,
            input_nodes=input_nodes,
            output_node=region_nodes[-1],
            compiled_kernel=compiled_kernel,
        )

    return None


def _build_region_graph_module(region_nodes, env_example):
    from torch.fx import Graph, GraphModule, Node

    graph = Graph()
    mapping = {}
    input_nodes = []
    seen_inputs = set()

    def register_input(node):
        if node not in seen_inputs:
            seen_inputs.add(node)
            input_nodes.append(node)
            mapping[node] = graph.placeholder(node.name)

    def rewrite(value):
        if isinstance(value, Node):
            if value in mapping:
                return mapping[value]
            if value.name not in env_example:
                raise PointwiseLoweringError(
                    f"region input is missing example value: {value.name}"
                )
            register_input(value)
            return mapping[value]
        if isinstance(value, tuple):
            return tuple(rewrite(item) for item in value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    for node in region_nodes:
        new_node = graph.call_function(
            node.target,
            args=rewrite(node.args),
            kwargs=rewrite(node.kwargs),
        )
        mapping[node] = new_node

    graph.output(mapping[region_nodes[-1]])
    region_inputs = [env_example[node.name] for node in input_nodes]
    return GraphModule(graph), region_inputs, input_nodes


def _region_has_single_output(region_nodes, users):
    region_set = set(region_nodes)
    for node in region_nodes[:-1]:
        if any(user not in region_set for user in users.get(node, ())):
            return False
    return True


def _build_users(nodes):
    from torch.fx import Node

    users = {}

    def visit_arg(owner, value):
        if isinstance(value, Node):
            users.setdefault(value, []).append(owner)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit_arg(owner, item)
        elif isinstance(value, dict):
            for item in value.values():
                visit_arg(owner, item)

    for node in nodes:
        for arg in node.args:
            visit_arg(node, arg)
        for value in node.kwargs.values():
            visit_arg(node, value)
    return users


def _run_call_function_node(node, env):
    from torch.fx.graph import resolve

    call_args = tuple(resolve(arg, env) for arg in node.args)
    call_kwargs = {key: resolve(value, env) for key, value in node.kwargs.items()}
    return run_eager_target(node.target, call_args, call_kwargs)
