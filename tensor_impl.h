#pragma once

#include "intrusive_ptr.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>


// ============================================================
// 模拟 c10::TensorImpl
// 对应 PyTorch: c10/core/TensorImpl.h
//
// TensorImpl 是 Tensor 的底层实现，持有真正的数据和元信息。
// 它是引用计数对象，多个 TensorBase/Tensor 可以共享同一个 TensorImpl。
// ============================================================

enum class DeviceType { CPU, CUDA };

struct Storage {
    std::vector<float> data;

    explicit Storage(int size, float fill_value = 0.0f) : data(size, fill_value) {}
    Storage() = default;

    int nbytes() const { return static_cast<int>(data.size()); }
    float* data_ptr() { return data.data(); }
    const float* data_ptr() const { return data.data(); }
};

class TensorImpl : public IntrusivePtrTarget {
public:
    // 数据存储（模拟 c10::Storage）
    std::shared_ptr<Storage> storage_;

    // 元信息
    std::vector<int> sizes_;
    std::vector<int> strides_;
    int storage_offset_ = 0;
    DeviceType device_ = DeviceType::CPU;
    bool requires_grad_ = false;
public:
    // 标准构造函数：分配新 storage，填充初始值
    TensorImpl(std::vector<int> sizes, float fill_value = 0.0f)
        : sizes_(std::move(sizes)) {
        compute_contiguous_strides();
        int n = numel();
        storage_ = std::make_shared<Storage>(n, fill_value);
    }

    // View 构造函数：共享已有 storage，使用不同的元信息（零拷贝）
    // 这是 PyTorch view 机制的核心：transpose/slice/reshape 等操作
    // 只创建新的 TensorImpl，共享同一块 storage 内存
    TensorImpl(std::shared_ptr<Storage> storage, int storage_offset,
               std::vector<int> sizes, std::vector<int> strides)
        : storage_(std::move(storage))
        , sizes_(std::move(sizes))
        , strides_(std::move(strides))
        , storage_offset_(storage_offset) {}

    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;

    int dim() const { return static_cast<int>(sizes_.size()); }

    // 根据 sizes 计算元素总数（不依赖 storage 大小，view 场景下两者可能不同）
    int numel() const {
        int n = 1;
        for (int s : sizes_) n *= s;
        return n;
    }

    float* data_ptr() { return storage_->data.data() + storage_offset_; }
    const float* data_ptr() const { return storage_->data.data() + storage_offset_; }

    // 判断 tensor 是否是行主序连续存储（C-contiguous）
    // 连续意味着 strides 严格等于从 sizes 计算出的行主序 strides
    bool is_contiguous() const {
        int expected = 1;
        for (int i = dim() - 1; i >= 0; i--) {
            if (sizes_[i] == 1) continue; // size=1 的维度 stride 无关紧要
            if (strides_[i] != expected) return false;
            expected *= sizes_[i];
        }
        return true;
    }

    // 根据多维索引计算在 storage 中的实际偏移
    int offset_at(const std::vector<int>& indices) const {
        int offset = storage_offset_;
        for (int i = 0; i < dim(); i++)
            offset += indices[i] * strides_[i];
        return offset;
    }

    // 按逻辑平坦索引读取元素（stride 感知）
    // flat_idx 是按行主序排列的逻辑索引 (0, 1, 2, ...)
    float read_logical(int flat_idx) const {
        if (is_contiguous()) return data_ptr()[flat_idx];
        int offset = storage_offset_;
        for (int i = dim() - 1; i >= 0; i--) {
            int coord = flat_idx % sizes_[i];
            flat_idx /= sizes_[i];
            offset += coord * strides_[i];
        }
        return storage_->data[offset];
    }

private:
    // 计算行主序（C-contiguous）strides
    void compute_contiguous_strides() {
        strides_.resize(sizes_.size());
        int stride = 1;
        for (int i = static_cast<int>(sizes_.size()) - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= sizes_[i];
        }
    }
};
