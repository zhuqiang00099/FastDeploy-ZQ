// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy/vision/ocr/ppocr/det_preprocessor.h"

#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

std::array<int, 4> DBDetectorPreprocessor::OcrDetectorGetInfo(
    FDMat* img, int max_size_len) {
  int w = img->Width();
  int h = img->Height();
  if (static_shape_infer_) {
    return {w, h, det_image_shape_[2], det_image_shape_[1]};
  }

  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = float(max_size_len) / float(h);
    } else {
      ratio = float(max_size_len) / float(w);
    }
  }
  int resize_h = int(float(h) * ratio);
  int resize_w = int(float(w) * ratio);
  resize_h = std::max(int(std::round(float(resize_h) / 32) * 32), 32);
  resize_w = std::max(int(std::round(float(resize_w) / 32) * 32), 32);

  return {w, h, resize_w, resize_h};
  /*
   *ratio_h = float(resize_h) / float(h);
   *ratio_w = float(resize_w) / float(w);
   */
}

std::array<int, 4> DBDetectorPreprocessor::OcrDetectorGetInfoV5(FDMat* img) {
  // Mirrors PaddleOCR cpp_infer DetResizeForTest::ResizeImageType0.
  // Supports limit_type in {"max", "min", "resize_long"} and clamps the
  // longest side to max_side_limit_. Returns {ori_w, ori_h, resize_w,
  // resize_h} with both resize dims rounded up to multiples of 32.
  int w = img->Width();
  int h = img->Height();
  if (static_shape_infer_) {
    return {w, h, det_image_shape_[2], det_image_shape_[1]};
  }

  // PaddleOCR pads tiny images (<64 px on either side) before resizing.
  if (h + w < 64) {
    int pad_h = std::max(32, h);
    int pad_w = std::max(32, w);
    w = pad_w;
    h = pad_h;
  }

  float ratio = 1.f;
  if (limit_type_ == "max") {
    if (std::max(h, w) > limit_side_len_) {
      ratio = float(limit_side_len_) / std::max(h, w);
    }
  } else if (limit_type_ == "min") {
    if (std::min(h, w) < limit_side_len_) {
      ratio = float(limit_side_len_) / std::min(h, w);
    }
  } else if (limit_type_ == "resize_long") {
    ratio = float(limit_side_len_) / std::max(h, w);
  } else {
    FDERROR << "Not supported limit_type: " << limit_type_ << std::endl;
    return {w, h, w, h};
  }

  int resize_h = int(h * ratio);
  int resize_w = int(w * ratio);

  if (std::max(resize_h, resize_w) > max_side_limit_) {
    ratio = float(max_side_limit_) / std::max(resize_h, resize_w);
    resize_h = int(resize_h * ratio);
    resize_w = int(resize_w * ratio);
  }
  resize_h = std::max(int(std::round(resize_h / 32.0) * 32), 32);
  resize_w = std::max(int(std::round(resize_w / 32.0) * 32), 32);

  return {w, h, resize_w, resize_h};
}

DBDetectorPreprocessor::DBDetectorPreprocessor() {
  resize_op_ = std::make_shared<Resize>(-1, -1);

  std::vector<float> value = {0, 0, 0};
  pad_op_ = std::make_shared<Pad>(0, 0, 0, 0, value);

  normalize_permute_op_ = std::make_shared<NormalizeAndPermute>(
      std::vector<float>({0.485f, 0.456f, 0.406f}),
      std::vector<float>({0.229f, 0.224f, 0.225f}), true);
}

bool DBDetectorPreprocessor::ResizeImage(FDMat* img, int resize_w, int resize_h,
                                         int max_resize_w, int max_resize_h) {
  resize_op_->SetWidthAndHeight(resize_w, resize_h);
  (*resize_op_)(img);

  pad_op_->SetPaddingSize(0, max_resize_h - resize_h, 0,
                          max_resize_w - resize_w);
  (*pad_op_)(img);
  return true;
}

bool DBDetectorPreprocessor::Apply(FDMatBatch* image_batch,
                                   std::vector<FDTensor>* outputs) {
  int max_resize_w = 0;
  int max_resize_h = 0;
  batch_det_img_info_.clear();
  batch_det_img_info_.resize(image_batch->mats->size());
  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat* mat = &(image_batch->mats->at(i));
    if (use_v5_resize_) {
      batch_det_img_info_[i] = OcrDetectorGetInfoV5(mat);
    } else {
      batch_det_img_info_[i] = OcrDetectorGetInfo(mat, max_side_len_);
    }
    max_resize_w = std::max(max_resize_w, batch_det_img_info_[i][2]);
    max_resize_h = std::max(max_resize_h, batch_det_img_info_[i][3]);
  }
  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat* mat = &(image_batch->mats->at(i));
    ResizeImage(mat, batch_det_img_info_[i][2], batch_det_img_info_[i][3],
                max_resize_w, max_resize_h);
  }

  if (!disable_normalize_ && !disable_permute_) {
    (*normalize_permute_op_)(image_batch);
  }

  outputs->resize(1);
  FDTensor* tensor = image_batch->Tensor();
  (*outputs)[0].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);
  return true;
}

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
