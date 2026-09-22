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

#pragma once
#include "fastdeploy/vision/common/processors/manager.h"
#include "fastdeploy/vision/common/processors/resize.h"
#include "fastdeploy/vision/common/processors/pad.h"
#include "fastdeploy/vision/common/processors/normalize_and_permute.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {

namespace ocr {
/*! @brief Preprocessor object for DBDetector serials model.
 */
class FASTDEPLOY_DECL DBDetectorPreprocessor : public ProcessorManager {
 public:
  DBDetectorPreprocessor();

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] image_batch The input image batch
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess successed, otherwise false
   */
  virtual bool Apply(FDMatBatch* image_batch, std::vector<FDTensor>* outputs);

  /// Set max_side_len for the detection preprocess, default is 960
  /// Deprecated: kept for backward compatibility. For PP-OCRv5/v6 prefer
  /// SetLimitSideLen / SetLimitType / SetMaxSideLimit.
  void SetMaxSideLen(int max_side_len) { max_side_len_ = max_side_len; }

  /// Get max_side_len of the detection preprocess
  int GetMaxSideLen() const { return max_side_len_; }

  /// Set limit_side_len for the detection preprocess (PP-OCRv5/v6 default 64).
  /// When set together with limit_type, max_side_len_ is ignored.
  void SetLimitSideLen(int limit_side_len) { limit_side_len_ = limit_side_len; }

  /// Get limit_side_len of the detection preprocess
  int GetLimitSideLen() const { return limit_side_len_; }

  /// Set limit_type for the detection preprocess, supported values:
  /// "max" (default, behave like V2/V3/V4), "min", "resize_long".
  /// PP-OCRv5/v6 default is "min".
  void SetLimitType(const std::string& limit_type) { limit_type_ = limit_type; }

  /// Get limit_type of the detection preprocess
  std::string GetLimitType() const { return limit_type_; }

  /// Set max_side_limit for the detection preprocess (PP-OCRv5/v6 default 4000).
  /// Used to clamp the longest side after ratio scaling.
  void SetMaxSideLimit(int max_side_limit) { max_side_limit_ = max_side_limit; }

  /// Get max_side_limit of the detection preprocess
  int GetMaxSideLimit() const { return max_side_limit_; }

  /// Set preprocess normalize parameters, please call this API to customize
  /// the normalize parameters, otherwise it will use the default normalize
  /// parameters.
  void SetNormalize(const std::vector<float>& mean,
                    const std::vector<float>& std,
                    bool is_scale) {
    normalize_permute_op_ =
        std::make_shared<NormalizeAndPermute>(mean, std, is_scale);
  }

  /// Get the image info of the last batch, return a list of array
  /// {image width, image height, resize width, resize height}
  const std::vector<std::array<int, 4>>* GetBatchImgInfo() {
    return &batch_det_img_info_;
  }

  /// This function will disable normalize in preprocessing step.
  void DisableNormalize() { disable_permute_ = true; }
  /// This function will disable hwc2chw in preprocessing step.
  void DisablePermute() { disable_normalize_ = true; }

  /// Set det_image_shape for the detection preprocess.
  /// This api is usually used when you retrain the model.
  /// Generally, you do not need to use it.
  void SetDetImageShape(const std::vector<int>& det_image_shape) {
    det_image_shape_ = det_image_shape;
  }
  /// Get cls_image_shape for the classification preprocess
  std::vector<int> GetDetImageShape() const { return det_image_shape_; }

  /// Set static_shape_infer is true or not. When deploy PP-OCR
  /// on hardware which can not support dynamic input shape very well,
  /// like Huawei Ascned, static_shape_infer needs to to be true.
  void SetStaticShapeInfer(bool static_shape_infer) {
    static_shape_infer_ = static_shape_infer;
  }
  /// Get static_shape_infer of the recognition preprocess
  bool GetStaticShapeInfer() const { return static_shape_infer_; }

  /// Switch to PP-OCRv5/v6 resize path (limit_type/limit_side_len/
  /// max_side_limit). When false (default) the preprocessor keeps the
  /// legacy V2/V3/V4 max_side_len behavior.
  void UseV5Resize(bool use_v5_resize) { use_v5_resize_ = use_v5_resize; }
  bool GetUseV5Resize() const { return use_v5_resize_; }

 private:
  bool ResizeImage(FDMat* img, int resize_w, int resize_h, int max_resize_w,
                   int max_resize_h);
  // for recording the switch of hwc2chw
  bool disable_permute_ = false;
  // for recording the switch of normalize
  bool disable_normalize_ = false;
  int max_side_len_ = 960;
  // PP-OCRv5/v6 detection resize parameters. Defaults keep V2/V3/V4 behavior.
  int limit_side_len_ = 960;
  std::string limit_type_ = "max";
  int max_side_limit_ = 4000;
  std::vector<std::array<int, 4>> batch_det_img_info_;
  std::shared_ptr<Resize> resize_op_;
  std::shared_ptr<Pad> pad_op_;
  std::shared_ptr<NormalizeAndPermute> normalize_permute_op_;
  std::vector<int> det_image_shape_ = {3, 960, 960};
  bool static_shape_infer_ = false;
  bool use_v5_resize_ = false;
  std::array<int, 4> OcrDetectorGetInfo(FDMat* img, int max_size_len);
  // New resize info path used by PP-OCRv5/v6. Implements limit_type
  // ("max"/"min"/"resize_long") and max_side_limit clamp, mirroring
  // PaddleOCR cpp_infer ResizeImageType0.
  std::array<int, 4> OcrDetectorGetInfoV5(FDMat* img);
};

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
