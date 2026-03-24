#pragma once

#include <cstdint>
#include <type_traits>

// ============================================================
// 模拟 c10::intrusive_ptr<T>
// 对应 PyTorch: c10/util/intrusive_ptr.h
//
// 通用侵入式智能指针模板。T 必须继承 IntrusivePtrTarget。
// 引用计数存储在对象自身中，避免像 shared_ptr 那样额外分配控制块。
// ============================================================

struct IntrusivePtrTarget {
    mutable int refcount_;
    IntrusivePtrTarget() : refcount_(1) {}
    IntrusivePtrTarget(const IntrusivePtrTarget&) = delete;
    IntrusivePtrTarget& operator=(const IntrusivePtrTarget&) = delete;
    virtual ~IntrusivePtrTarget() = default;

    void retain() const { ++ refcount_; }
    void release() const {
        if (--refcount_== 0) {
            delete this;
        }
    }
};

template<typename T>
class IntrusivePtr {
    static_assert(std::is_base_of<IntrusivePtrTarget, T>::value,
        "T must inherit from IntrusivePtrTarget");
    T* ptr_ = nullptr;
    
public:
    IntrusivePtr() = default;
    explicit IntrusivePtr(T* p) : ptr_(p) {}

    IntrusivePtr(const IntrusivePtr& other) : ptr_(other.ptr_) {
        if (ptr_) ptr_->retain();
    }
    IntrusivePtr(IntrusivePtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    IntrusivePtr& operator=(const IntrusivePtr& other) {
        if (this != &other) {
            if (ptr_) ptr_->release();
            ptr_ = other.ptr_;
            if (ptr_) ptr_->retain();
        }
        return *this;
    }
    IntrusivePtr& operator=(IntrusivePtr&& other) noexcept {
        if (this != &other) {
            if (ptr_) ptr_->release();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    ~IntrusivePtr() { if (ptr_) ptr_->release(); }

    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    int use_count() const { return ptr_ ? ptr_->refcount_ : 0; }
};
