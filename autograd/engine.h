#pragma once

#include "function.h"
#include "variable.h"
#include "../ops.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stdexcept>
#include <cstring>

// ============================================================
// Engine — 反向传播引擎
// 对应 PyTorch 的 torch::autograd::Engine
//
// 从根节点出发，拓扑排序计算图，依次执行每个节点的 backward，
// 将梯度分发给上游节点，最终在叶子变量上累加梯度。
// ============================================================

class Engine {
public:
    static void backward(
        std::shared_ptr<AutogradFunction> root_fn,
        Tensor grad_output,
        bool retain_graph = false
    ) {
        if (!root_fn) {
            throw std::runtime_error("backward: root function is null");
        }

        // 1. 拓扑排序
        std::vector<AutogradFunction*> order = topological_sort(root_fn.get());

        // 2. 梯度缓冲
        std::unordered_map<AutogradFunction*, std::vector<Tensor>> grad_buffer;
        grad_buffer[root_fn.get()] = {grad_output};

        // 3. 按拓扑序执行 backward
        for (AutogradFunction* fn : order) {
            auto it = grad_buffer.find(fn);
            if (it == grad_buffer.end()) continue;

            // 合并梯度
            std::vector<Tensor> merged = merge_grads(it->second);

            // 调用 backward
            std::vector<Tensor> grad_inputs = fn->apply(merged);

            // 分发梯度给输入
            for (int i = 0; i < static_cast<int>(fn->inputs.size()); i++) {
                if (i >= static_cast<int>(grad_inputs.size())) break;

                const InputInfo& info = fn->inputs[i];

                if (info.fn) {
                    // 非叶子：传给上游函数节点
                    grad_buffer[info.fn.get()].push_back(grad_inputs[i]);
                } else if (info.leaf) {
                    // 新系统叶子：累加梯度到 TensorImpl
                    if (info.leaf->requires_grad_) {
                        Tensor g = grad_inputs[i];
                        Tensor cg = g.is_contiguous() ? g : native::contiguous(g);
                        info.leaf->accumulate_grad(
                            cg.data_ptr(), cg.numel(),
                            std::vector<int>(cg.sizes()));
                    }
                } else if (info.variable) {
                    // System A 兼容：累加梯度到 VariableImpl
                    if (info.variable->requires_grad) {
                        info.variable->accumulate_grad(grad_inputs[i]);
                        info.variable->run_hooks();
                    }
                }
            }

            if (!retain_graph) {
                grad_buffer.erase(it);
            }
        }
    }

private:
    static std::vector<AutogradFunction*> topological_sort(AutogradFunction* root) {
        std::unordered_map<AutogradFunction*, int> in_degree;
        std::unordered_set<AutogradFunction*> visited;
        std::queue<AutogradFunction*> bfs;

        bfs.push(root);
        visited.insert(root);
        while (!bfs.empty()) {
            AutogradFunction* fn = bfs.front();
            bfs.pop();
            in_degree[fn]; // 确保存在

            for (auto& info : fn->inputs) {
                if (info.fn) {
                    in_degree[info.fn.get()]++;
                    if (visited.find(info.fn.get()) == visited.end()) {
                        visited.insert(info.fn.get());
                        bfs.push(info.fn.get());
                    }
                }
            }
        }

        // Kahn 算法
        std::queue<AutogradFunction*> ready;
        for (auto& [fn, deg] : in_degree) {
            if (deg == 0) ready.push(fn);
        }

        std::vector<AutogradFunction*> order;
        while (!ready.empty()) {
            AutogradFunction* fn = ready.front();
            ready.pop();
            order.push_back(fn);

            for (auto& info : fn->inputs) {
                if (info.fn) {
                    in_degree[info.fn.get()]--;
                    if (in_degree[info.fn.get()] == 0) {
                        ready.push(info.fn.get());
                    }
                }
            }
        }

        return order;
    }

    static std::vector<Tensor> merge_grads(const std::vector<Tensor>& grads) {
        if (grads.empty()) return {};
        if (grads.size() == 1) return grads;

        Tensor merged = native::empty(std::vector<int>(grads[0].sizes()));
        float* pm = merged.data_ptr();
        int n = merged.numel();
        // 用第一个梯度直接初始化，省去零初始化的额外遍历
        auto* first = grads[0].unsafeGetTensorImpl();
        if (first->is_contiguous()) {
            const float* src = first->data_ptr();
            std::memcpy(pm, src, n * sizeof(float));
        } else {
            for (int i = 0; i < n; i++) pm[i] = first->read_logical(i);
        }
        // 累加剩余梯度，连续张量走快速路径
        for (size_t gi = 1; gi < grads.size(); gi++) {
            auto* impl = grads[gi].unsafeGetTensorImpl();
            if (impl->is_contiguous()) {
                const float* src = impl->data_ptr();
                for (int i = 0; i < n; i++) pm[i] += src[i];
            } else {
                for (int i = 0; i < n; i++) pm[i] += impl->read_logical(i);
            }
        }
        return {merged};
    }
};
