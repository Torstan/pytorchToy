#pragma once

#include "../ops.h"
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace pointwise {

enum class ValueKind : int {
    Input = 0,
    Temp = 1,
    Const = 2,
};

enum class OpKind : int {
    Sin = 0,
    Cos = 1,
    Relu = 2,
    Tanh = 3,
    Neg = 4,
    Add = 5,
    Sub = 6,
    Mul = 7,
    Div = 8,
};

struct ValueRef {
    ValueKind kind;
    int index;
};

struct Instruction {
    OpKind op;
    int dst;
    ValueRef lhs;
    ValueRef rhs;
};

class CompiledPointwiseProgram {
public:
    using EncodedInstruction = std::tuple<int, int, int, int, int, int>;

    CompiledPointwiseProgram(
        std::vector<int> shape,
        int num_inputs,
        int num_temps,
        std::vector<float> consts,
        std::vector<EncodedInstruction> instructions,
        int output_kind,
        int output_index)
        : shape_(std::move(shape))
        , num_inputs_(num_inputs)
        , num_temps_(num_temps)
        , consts_(std::move(consts))
        , output_(ValueRef{decode_value_kind(output_kind), output_index}) {
        if (num_inputs_ <= 0)
            throw std::runtime_error("CompiledPointwiseProgram: num_inputs must be > 0");
        if (num_temps_ <= 0)
            throw std::runtime_error("CompiledPointwiseProgram: num_temps must be > 0");

        numel_ = 1;
        for (int size : shape_) {
            if (size < 0)
                throw std::runtime_error("CompiledPointwiseProgram: negative dimension is not allowed");
            numel_ *= size;
        }

        instructions_.reserve(instructions.size());
        for (const auto& encoded : instructions) {
            int rhs_kind = std::get<4>(encoded);
            int rhs_index = std::get<5>(encoded);
            instructions_.push_back(Instruction{
                decode_op_kind(std::get<0>(encoded)),
                std::get<1>(encoded),
                ValueRef{decode_value_kind(std::get<2>(encoded)), std::get<3>(encoded)},
                rhs_kind == -1
                    ? ValueRef{ValueKind::Const, -1}
                    : ValueRef{decode_value_kind(rhs_kind), rhs_index},
            });
        }
    }

    Tensor run(const std::vector<Tensor>& inputs) const {
        validate_inputs(inputs);

        Tensor result = native::empty(shape_);
        if (numel_ == 0)
            return result;

        std::vector<const float*> input_ptrs;
        input_ptrs.reserve(inputs.size());
        for (const auto& input : inputs)
            input_ptrs.push_back(input.data_ptr());

        std::vector<float> temps(num_temps_, 0.0f);
        float* out_ptr = result.data_ptr();

        for (int flat_idx = 0; flat_idx < numel_; ++flat_idx) {
            for (const auto& inst : instructions_) {
                float lhs = load_ref(inst.lhs, input_ptrs, temps, flat_idx);
                float value = 0.0f;
                switch (inst.op) {
                    case OpKind::Sin:
                        value = std::sin(lhs);
                        break;
                    case OpKind::Cos:
                        value = std::cos(lhs);
                        break;
                    case OpKind::Relu:
                        value = lhs > 0.0f ? lhs : 0.0f;
                        break;
                    case OpKind::Tanh:
                        value = std::tanh(lhs);
                        break;
                    case OpKind::Neg:
                        value = -lhs;
                        break;
                    case OpKind::Add:
                        value = lhs + load_ref(inst.rhs, input_ptrs, temps, flat_idx);
                        break;
                    case OpKind::Sub:
                        value = lhs - load_ref(inst.rhs, input_ptrs, temps, flat_idx);
                        break;
                    case OpKind::Mul:
                        value = lhs * load_ref(inst.rhs, input_ptrs, temps, flat_idx);
                        break;
                    case OpKind::Div:
                        value = lhs / load_ref(inst.rhs, input_ptrs, temps, flat_idx);
                        break;
                }
                temps[inst.dst] = value;
            }
            out_ptr[flat_idx] = load_ref(output_, input_ptrs, temps, flat_idx);
        }

        return result;
    }

private:
    static ValueKind decode_value_kind(int kind) {
        switch (kind) {
            case 0:
                return ValueKind::Input;
            case 1:
                return ValueKind::Temp;
            case 2:
                return ValueKind::Const;
            default:
                throw std::runtime_error("CompiledPointwiseProgram: unknown value kind");
        }
    }

    static OpKind decode_op_kind(int op) {
        switch (op) {
            case 0:
                return OpKind::Sin;
            case 1:
                return OpKind::Cos;
            case 2:
                return OpKind::Relu;
            case 3:
                return OpKind::Tanh;
            case 4:
                return OpKind::Neg;
            case 5:
                return OpKind::Add;
            case 6:
                return OpKind::Sub;
            case 7:
                return OpKind::Mul;
            case 8:
                return OpKind::Div;
            default:
                throw std::runtime_error("CompiledPointwiseProgram: unknown op kind");
        }
    }

    float load_ref(
        const ValueRef& ref,
        const std::vector<const float*>& input_ptrs,
        const std::vector<float>& temps,
        int flat_idx) const {
        switch (ref.kind) {
            case ValueKind::Input:
                if (ref.index < 0 || ref.index >= static_cast<int>(input_ptrs.size()))
                    throw std::runtime_error("CompiledPointwiseProgram: input index out of range");
                return input_ptrs[ref.index][flat_idx];
            case ValueKind::Temp:
                if (ref.index < 0 || ref.index >= static_cast<int>(temps.size()))
                    throw std::runtime_error("CompiledPointwiseProgram: temp index out of range");
                return temps[ref.index];
            case ValueKind::Const:
                if (ref.index < 0 || ref.index >= static_cast<int>(consts_.size()))
                    throw std::runtime_error("CompiledPointwiseProgram: const index out of range");
                return consts_[ref.index];
        }
        throw std::runtime_error("CompiledPointwiseProgram: invalid value ref");
    }

    void validate_inputs(const std::vector<Tensor>& inputs) const {
        if (static_cast<int>(inputs.size()) != num_inputs_)
            throw std::runtime_error("CompiledPointwiseProgram: input count mismatch");

        for (const auto& input : inputs) {
            if (input.requires_grad())
                throw std::runtime_error("CompiledPointwiseProgram: requires_grad input is not supported");
            if (!input.is_contiguous())
                throw std::runtime_error("CompiledPointwiseProgram: non-contiguous input is not supported");
            if (input.sizes() != shape_)
                throw std::runtime_error("CompiledPointwiseProgram: input shape mismatch");
        }
    }

private:
    std::vector<int> shape_;
    int num_inputs_;
    int num_temps_;
    int numel_ = 0;
    std::vector<float> consts_;
    std::vector<Instruction> instructions_;
    ValueRef output_;
};

} // namespace pointwise
