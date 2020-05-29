// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the 
// specific language governing permissions and limitations under the License.

#ifndef TNN_CPU_COMPUTE_NORMALIZED_BBOX_HPP
#define TNN_CPU_COMPUTE_NORMALIZED_BBOX_HPP
#include <cstddef>
#include <cstring>
#include "tnn/core/status.h"

namespace TNN_NS {

class NormalizedBBox {
public:
    NormalizedBBox() {
        memset(_has_bits_, 0, sizeof(_has_bits_));
        size_ = 0.;
        xmin_ = 0.;
        xmax_ = 0.;
        ymin_ = 0.;
        ymax_ = 0.;
        // clear_has_size();
    }

    virtual ~NormalizedBBox() {
        memset(_has_bits_, 0, sizeof(_has_bits_));
    }

    // NormalizedBBox(const NormalizedBBox& from);

    /*inline NormalizedBBox& operator=(const NormalizedBBox& from) {
        CopyFrom(from);
        return *this;
    }

    inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
        return _unknown_fields_;
    }

    inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
        return &_unknown_fields_;
    }

    static const ::google::protobuf::Descriptor* descriptor();
    static const NormalizedBBox& default_instance();

    void Swap(NormalizedBBox* other);

    // implements Message ----------------------------------------------

    NormalizedBBox* New() const;

    //void CopyFrom(const ::google::protobuf::Message& from);
    //void MergeFrom(const ::google::protobuf::Message& from);
    void CopyFrom(const NormalizedBBox& from);
    void MergeFrom(const NormalizedBBox& from);
    void Clear();
    bool IsInitialized() const;

    int ByteSize() const;

    bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8*
  SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const; int
  GetCachedSize() const { return _cached_size_; } private: void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;*/
public:
    //::google::protobuf::Metadata GetMetadata() const;

    // nested types ----------------------------------------------------

    // accessors -------------------------------------------------------

    // optional float xmin = 1;
    inline bool has_xmin() const;
    inline void clear_xmin();
    static const int kXminFieldNumber = 1;
    inline float xmin() const;
    inline void set_xmin(float value);

    // optional float ymin = 2;
    inline bool has_ymin() const;
    inline void clear_ymin();
    static const int kYminFieldNumber = 2;
    inline float ymin() const;
    inline void set_ymin(float value);

    // optional float xmax = 3;
    inline bool has_xmax() const;
    inline void clear_xmax();
    static const int kXmaxFieldNumber = 3;
    inline float xmax() const;
    inline void set_xmax(float value);

    // optional float ymax = 4;
    inline bool has_ymax() const;
    inline void clear_ymax();
    static const int kYmaxFieldNumber = 4;
    inline float ymax() const;
    inline void set_ymax(float value);

    // optional int32 label = 5;
    inline bool has_label() const;
    inline void clear_label();
    static const int kLabelFieldNumber = 5;
    inline int label() const;
    inline void set_label(int value);

    // optional bool difficult = 6;
    inline bool has_difficult() const;
    inline void clear_difficult();
    static const int kDifficultFieldNumber = 6;
    inline bool difficult() const;
    inline void set_difficult(bool value);

    // optional float score = 7;
    inline bool has_score() const;
    inline void clear_score();
    static const int kScoreFieldNumber = 7;
    inline float score() const;
    inline void set_score(float value);

    // optional float size = 8;
    inline bool has_size() const;
    inline void clear_size();
    static const int kSizeFieldNumber = 8;
    inline float size() const;
    inline void set_size(float value);

    // @@protoc_insertion_point(class_scope:caffe.NormalizedBBox)
private:
    inline void set_has_xmin();
    inline void clear_has_xmin();
    inline void set_has_ymin();
    inline void clear_has_ymin();
    inline void set_has_xmax();
    inline void clear_has_xmax();
    inline void set_has_ymax();
    inline void clear_has_ymax();
    inline void set_has_label();
    inline void clear_has_label();
    inline void set_has_difficult();
    inline void clear_has_difficult();
    inline void set_has_score();
    inline void clear_has_score();
    inline void set_has_size();
    inline void clear_has_size();

    //::google::protobuf::UnknownFieldSet _unknown_fields_;

    //::google::protobuf::uint32 _has_bits_[1];
    size_t _has_bits_[1];
    mutable int _cached_size_ = 0;
    float xmin_ = 0.f;
    float ymin_ = 0.f;
    float xmax_ = 0.f;
    float ymax_ = 0.f;
    //::google::protobuf::int32 label_;
    int label_ = 0;
    bool difficult_ = false;
    float score_ = 0.f;
    float size_ = 0.f;
    // friend void  protobuf_AddDesc_caffe_2eproto();
    // friend void protobuf_AssignDesc_caffe_2eproto();
    // friend void protobuf_ShutdownFile_caffe_2eproto();

    void InitAsDefaultInstance();
    static NormalizedBBox* default_instance_;
};

// NormalizedBBox

// optional float xmin = 1;
inline bool NormalizedBBox::has_xmin() const {
    return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void NormalizedBBox::set_has_xmin() {
    _has_bits_[0] |= 0x00000001u;
}
inline void NormalizedBBox::clear_has_xmin() {
    _has_bits_[0] &= ~0x00000001u;
}
inline void NormalizedBBox::clear_xmin() {
    xmin_ = 0;
    clear_has_xmin();
}
inline float NormalizedBBox::xmin() const {
    // @@protoc_insertion_point(field_get:caffe.NormalizedBBox.xmin)
    return xmin_;
}
inline void NormalizedBBox::set_xmin(float value) {
    set_has_xmin();
    xmin_ = value;
    // @@protoc_insertion_point(field_set:caffe.NormalizedBBox.xmin)
}

// optional float ymin = 2;
inline bool NormalizedBBox::has_ymin() const {
    return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void NormalizedBBox::set_has_ymin() {
    _has_bits_[0] |= 0x00000002u;
}
inline void NormalizedBBox::clear_has_ymin() {
    _has_bits_[0] &= ~0x00000002u;
}
inline void NormalizedBBox::clear_ymin() {
    ymin_ = 0;
    clear_has_ymin();
}
inline float NormalizedBBox::ymin() const {
    // @@protoc_insertion_point(field_get:caffe.NormalizedBBox.ymin)
    return ymin_;
}
inline void NormalizedBBox::set_ymin(float value) {
    set_has_ymin();
    ymin_ = value;
    // @@protoc_insertion_point(field_set:caffe.NormalizedBBox.ymin)
}

// optional float xmax = 3;
inline bool NormalizedBBox::has_xmax() const {
    return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void NormalizedBBox::set_has_xmax() {
    _has_bits_[0] |= 0x00000004u;
}
inline void NormalizedBBox::clear_has_xmax() {
    _has_bits_[0] &= ~0x00000004u;
}
inline void NormalizedBBox::clear_xmax() {
    xmax_ = 0;
    clear_has_xmax();
}
inline float NormalizedBBox::xmax() const {
    // @@protoc_insertion_point(field_get:caffe.NormalizedBBox.xmax)
    return xmax_;
}
inline void NormalizedBBox::set_xmax(float value) {
    set_has_xmax();
    xmax_ = value;
    // @@protoc_insertion_point(field_set:caffe.NormalizedBBox.xmax)
}

// optional float ymax = 4;
inline bool NormalizedBBox::has_ymax() const {
    return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void NormalizedBBox::set_has_ymax() {
    _has_bits_[0] |= 0x00000008u;
}
inline void NormalizedBBox::clear_has_ymax() {
    _has_bits_[0] &= ~0x00000008u;
}
inline void NormalizedBBox::clear_ymax() {
    ymax_ = 0;
    clear_has_ymax();
}
inline float NormalizedBBox::ymax() const {
    // @@protoc_insertion_point(field_get:caffe.NormalizedBBox.ymax)
    return ymax_;
}
inline void NormalizedBBox::set_ymax(float value) {
    set_has_ymax();
    ymax_ = value;
    // @@protoc_insertion_point(field_set:caffe.NormalizedBBox.ymax)
}

// optional int32 label = 5;
inline bool NormalizedBBox::has_label() const {
    return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void NormalizedBBox::set_has_label() {
    _has_bits_[0] |= 0x00000010u;
}
inline void NormalizedBBox::clear_has_label() {
    _has_bits_[0] &= ~0x00000010u;
}
inline void NormalizedBBox::clear_label() {
    label_ = 0;
    clear_has_label();
}
inline int NormalizedBBox::label() const {
    // @@protoc_insertion_point(field_get:caffe.NormalizedBBox.label)
    return label_;
}
inline void NormalizedBBox::set_label(int value) {
    set_has_label();
    label_ = value;
    // @@protoc_insertion_point(field_set:caffe.NormalizedBBox.label)
}

// optional bool difficult = 6;
inline bool NormalizedBBox::has_difficult() const {
    return (_has_bits_[0] & 0x00000020u) != 0;
}
inline void NormalizedBBox::set_has_difficult() {
    _has_bits_[0] |= 0x00000020u;
}
inline void NormalizedBBox::clear_has_difficult() {
    _has_bits_[0] &= ~0x00000020u;
}
inline void NormalizedBBox::clear_difficult() {
    difficult_ = false;
    clear_has_difficult();
}
inline bool NormalizedBBox::difficult() const {
    // @@protoc_insertion_point(field_get:caffe.NormalizedBBox.difficult)
    return difficult_;
}
inline void NormalizedBBox::set_difficult(bool value) {
    set_has_difficult();
    difficult_ = value;
    // @@protoc_insertion_point(field_set:caffe.NormalizedBBox.difficult)
}

// optional float score = 7;
inline bool NormalizedBBox::has_score() const {
    return (_has_bits_[0] & 0x00000040u) != 0;
}
inline void NormalizedBBox::set_has_score() {
    _has_bits_[0] |= 0x00000040u;
}
inline void NormalizedBBox::clear_has_score() {
    _has_bits_[0] &= ~0x00000040u;
}
inline void NormalizedBBox::clear_score() {
    score_ = 0;
    clear_has_score();
}
inline float NormalizedBBox::score() const {
    // @@protoc_insertion_point(field_get:caffe.NormalizedBBox.score)
    return score_;
}
inline void NormalizedBBox::set_score(float value) {
    set_has_score();
    score_ = value;
    // @@protoc_insertion_point(field_set:caffe.NormalizedBBox.score)
}

// optional float size = 8;
inline bool NormalizedBBox::has_size() const {
    return (_has_bits_[0] & 0x00000080u) != 0;
}
inline void NormalizedBBox::set_has_size() {
    _has_bits_[0] |= 0x00000080u;
}
inline void NormalizedBBox::clear_has_size() {
    _has_bits_[0] &= ~0x00000080u;
}
inline void NormalizedBBox::clear_size() {
    size_ = 0;
    clear_has_size();
}
inline float NormalizedBBox::size() const {
    // @@protoc_insertion_point(field_get:caffe.NormalizedBBox.size)
    return size_;
}
inline void NormalizedBBox::set_size(float value) {
    set_has_size();
    size_ = value;
    // @@protoc_insertion_point(field_set:caffe.NormalizedBBox.size)
}

}  // namespace TNN_NS

#endif  // TNN_NORMALIZED_BBOX_HPP
