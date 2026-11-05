#pragma once


#include <memory>
#include <NvInfer.h>
#include "common.h"


// 固定 shape 的 IPluginV3，输入为 NCHW、权重（FP16 已打包）、bias
// 注册 creator 以支持引擎反序列化；返回的构建期实例由调用方持有
// 此底层 v1 Plugin 固定使用第三个 bias tensor，不读取 p.use_bias；无 bias 由构建端提供零常量
std::unique_ptr<nvinfer1::IPluginV3> make_dwconv_baseline_plugin(const Conv2DParam &p, int n);

std::unique_ptr<nvinfer1::IPluginV3> make_dwconv_opt_plugin(const Conv2DParam &p, int n);
