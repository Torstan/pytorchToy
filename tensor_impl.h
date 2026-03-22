#pragma once
#include <vector>
#include <atomic>
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
};

class TensorImpl {
public:
    // 引用计数（模拟 c10::intrusive_ptr_target）
    mutable std::atomic<int> refcount_{1};

    // 数据存储（模拟 c10::Storage）
    std::shared_ptr<Storage> storage_;

    // 元信息
    std::vector<int> sizes_;
    std::vector<int> strides_;
    int storage_offset_ = 0;
    DeviceType device_ = DeviceType::CPU;
    bool requires_grad_ = false;
public:
    TensorImpl(std::vector<int> sizes, float fill_value = 0.0f)
        : sizes_(std::move(sizes)) {
        int numel = 1;
        for (int s : sizes_) numel *= s;

        storage_ = std::make_shared<Storage>();
        storage_->data.assign(numel, fill_value);

        // 计算行主序 strides
        strides_.resize(sizes_.size());
        int stride = 1;
        for (int i = static_cast<int>(sizes_.size()) - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= sizes_[i];
        }
    }

    int dim() const { return static_cast<int>(sizes_.size()); }
    int numel() const { return static_cast<int>(storage_->data.size()); }

    float* data_ptr() { return storage_->data.data() + storage_offset_; }
    const float* data_ptr() const { return storage_->data.data() + storage_offset_; }

    // 引用计数管理
    void retain() const { refcount_.fetch_add(1); }
    void release() const {
        if (refcount_.fetch_sub(1) == 1) {
            delete this;
        }
    }
};
