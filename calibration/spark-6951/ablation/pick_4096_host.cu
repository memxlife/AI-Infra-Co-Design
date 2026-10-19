// tilelang target: {"kind":"c","tag":"","keys":["cpu"]}
#define TVM_EXPORTS
#include "tvm/runtime/base.h"
#include "tvm/runtime/c_backend_api.h"
#include "tvm/ffi/c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#if defined(_MSC_VER)
#define TL_ALIGN(N) __declspec(align(N))
#else
#define TL_ALIGN(N) __attribute__((aligned(N)))
#endif
#ifdef __OBJC__
#include "tvm/runtime/device_api.h"
#include "tvm/ffi/function.h"
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <torch/mps.h>
#endif
void* __tvm_ffi__library_ctx = NULL;
static void* __tvm_set_device_packed = NULL;
static void* __tvm_tensormap_create_tiled_packed = NULL;
static void* matmul_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_matmul(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t TVMFFIEnvTensorAlloc(void*, void*);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_matmul(void* self_handle, void* args, int32_t num_args, void* result) {
  TL_ALIGN(128) TVMFFIAny stack[17];
  void* stack_ffi_any = stack;
  if (!((num_args == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "matmul: num_args should be 3", (long long)(num_args), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "matmul: args pointer is NULL");
    return -1;
  }
  int32_t A_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((A_type_index == 0) || (A_type_index == 4)) || (A_type_index == 7)) || (64 <= A_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel matmul input A expected pointer or tensor handle");
    return -1;
  }
  int32_t B_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((B_type_index == 0) || (B_type_index == 4)) || (B_type_index == 7)) || (64 <= B_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel matmul input B expected pointer or tensor handle");
    return -1;
  }
  int32_t allocator_anchor_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!((allocator_anchor_type_index == 70))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "matmul: allocator anchor must be a Tensor", (long long)(allocator_anchor_type_index), (long long)(70));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!((((TVMFFIAny*)args)[2].v_ptr) == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "matmul: allocator anchor is NULL");
    return -1;
  }
  if (!(((((DLTensor*)((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + (int64_t)24)))[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "matmul: allocator anchor has the wrong device type", (long long)((((DLTensor*)((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + (int64_t)24)))[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t dev_id = (((DLTensor*)((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + (int64_t)24)))[0].device.device_id);
  void* A = ((A_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* B = ((B_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  bool matmul_A_is_null = (A == NULL);
  if (!(!matmul_A_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "matmul.A is expected to have non-NULL pointer");
    return -1;
  }
  bool matmul_B_is_null = (B == NULL);
  if (!(!matmul_B_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "matmul.B is expected to have non-NULL pointer");
    return -1;
  }
  void* matmul_A_shape = (((DLTensor*)A)[0].shape);
  void* matmul_B_shape = (((DLTensor*)B)[0].shape);
  if (!(((((DLTensor*)A)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input A ndim mismatch, expected 2", (long long)((((DLTensor*)A)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* matmul_A_strides = (((DLTensor*)A)[0].strides);
  void* A_1 = (((DLTensor*)A)[0].data);
  if (!(((((DLTensor*)B)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input B ndim mismatch, expected 2", (long long)((((DLTensor*)B)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* matmul_B_strides = (((DLTensor*)B)[0].strides);
  void* B_1 = (((DLTensor*)B)[0].data);
  if (!(((((((DLTensor*)A)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)A)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)A)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel matmul input A dtype mismatch, expected float16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)matmul_A_shape)[0]) == 4096))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input A shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)matmul_A_shape)[0])), (long long)(4096));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)matmul_A_shape)[1]) == 4096))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input A shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)matmul_A_shape)[1])), (long long)(4096));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((matmul_A_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)matmul_A_strides)[1]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((matmul_A_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)matmul_A_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input A strides[1] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((matmul_A_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)matmul_A_strides)[0]);
  }
  if (!((condval_2 == 4096))) {
    int32_t condval_3;
    if ((matmul_A_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)matmul_A_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input A strides[0] violates packed ABI constraint", (long long)(condval_3), (long long)(4096));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)A)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input A byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)A)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)A)[0].device.device_id) == dev_id))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input A device_id violates packed ABI constraint", (long long)((((DLTensor*)A)[0].device.device_id)), (long long)(dev_id));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)A)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input A device_type mismatch, expected cuda", (long long)((((DLTensor*)A)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(A_1 == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel matmul input A data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)B)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)B)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)B)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel matmul input B dtype mismatch, expected float16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)matmul_B_shape)[0]) == 4096))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input B shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)matmul_B_shape)[0])), (long long)(4096));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)matmul_B_shape)[1]) == 4096))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input B shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)matmul_B_shape)[1])), (long long)(4096));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((matmul_B_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)matmul_B_strides)[1]);
  }
  if (!((condval_4 == 1))) {
    int32_t condval_5;
    if ((matmul_B_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)matmul_B_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input B strides[1] violates packed ABI constraint", (long long)(condval_5), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((matmul_B_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)matmul_B_strides)[0]);
  }
  if (!((condval_6 == 4096))) {
    int32_t condval_7;
    if ((matmul_B_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)matmul_B_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input B strides[0] violates packed ABI constraint", (long long)(condval_7), (long long)(4096));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)B)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input B byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)B)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)B)[0].device.device_id) == dev_id))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input B device_id violates packed ABI constraint", (long long)((((DLTensor*)B)[0].device.device_id)), (long long)(dev_id));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)B)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input B device_type mismatch, expected cuda", (long long)((((DLTensor*)B)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(B_1 == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel matmul input B data pointer is NULL");
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_1[1];
  void* callee_allocated_output_storage = stack_1;
  TVMFFIAny stack_2[1];
  void* callee_allocated_output_shape_storage = stack_2;
  TVMFFIAny stack_3[3];
  void* callee_allocated_output_prototype_storage = stack_3;
  (((int64_t*)callee_allocated_output_shape_storage)[0]) = (int64_t)4096;
  (((int64_t*)callee_allocated_output_shape_storage)[1]) = (int64_t)4096;
  uint64_t v_ = (uint64_t)0;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].data) = (*(void* *)(&(v_)));
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].shape) = (int64_t*)((void*)((char*)callee_allocated_output_shape_storage + (int64_t)0));
  uint64_t v__1 = (uint64_t)0;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].strides) = (int64_t*)(*(void* *)(&(v__1)));
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].ndim) = 2;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].dtype.code) = (uint8_t)2;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].dtype.bits) = (uint8_t)16;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].dtype.lanes) = (uint16_t)1;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].byte_offset) = (uint64_t)0;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].device.device_id) = dev_id;
  (((DLTensor*)callee_allocated_output_prototype_storage)[0].device.device_type) = (DLDeviceType)2;
  int32_t C_allocation_status = TVMFFIEnvTensorAlloc((((DLTensor*)callee_allocated_output_prototype_storage) + 0), ((void*)((char*)callee_allocated_output_storage + (int64_t)8)));
  if (C_allocation_status != 0) {
    return C_allocation_status;
  }
  void* C_handle = ((void*)((char*)(((TVMFFIAny*)callee_allocated_output_storage)[0].v_ptr) + (int64_t)24));
  bool matmul_C_is_null = (C_handle == NULL);
  if (!(!matmul_C_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "matmul.C is expected to have non-NULL pointer");
    return -1;
  }
  void* matmul_C_shape = (((DLTensor*)C_handle)[0].shape);
  if (!(((((DLTensor*)C_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input C ndim mismatch, expected 2", (long long)((((DLTensor*)C_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* matmul_C_strides = (((DLTensor*)C_handle)[0].strides);
  void* C = (((DLTensor*)C_handle)[0].data);
  if (!(((((((DLTensor*)C_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)C_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)C_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel matmul input C dtype mismatch, expected float16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)matmul_C_shape)[0]) == 4096))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input C shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)matmul_C_shape)[0])), (long long)(4096));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)matmul_C_shape)[1]) == 4096))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input C shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)matmul_C_shape)[1])), (long long)(4096));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_8;
  if ((matmul_C_strides == NULL)) {
    condval_8 = 1;
  } else {
    condval_8 = ((int32_t)((int64_t*)matmul_C_strides)[1]);
  }
  if (!((condval_8 == 1))) {
    int32_t condval_9;
    if ((matmul_C_strides == NULL)) {
      condval_9 = 1;
    } else {
      condval_9 = ((int32_t)((int64_t*)matmul_C_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input C strides[1] violates packed ABI constraint", (long long)(condval_9), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_10;
  if ((matmul_C_strides == NULL)) {
    condval_10 = 1;
  } else {
    condval_10 = ((int32_t)((int64_t*)matmul_C_strides)[0]);
  }
  if (!((condval_10 == 4096))) {
    int32_t condval_11;
    if ((matmul_C_strides == NULL)) {
      condval_11 = 1;
    } else {
      condval_11 = ((int32_t)((int64_t*)matmul_C_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input C strides[0] violates packed ABI constraint", (long long)(condval_11), (long long)(4096));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)C_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input C byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)C_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)C_handle)[0].device.device_id) == dev_id))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input C device_id violates packed ABI constraint", (long long)((((DLTensor*)C_handle)[0].device.device_id)), (long long)(dev_id));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)C_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel matmul input C device_type mismatch, expected cuda", (long long)((((DLTensor*)C_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(C == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel matmul input C data pointer is NULL");
    return -1;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)dev_id);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
  if (__tvm_set_device_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_set_device", &__tvm_set_device_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_1;
  result_1.type_index = kTVMFFINone;
  result_1.zero_padding = 0;
  result_1.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_set_device_packed, (TVMFFIAny*) stack_ffi_any, 2, &result_1) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_4[16];
  void* A_desc = stack_4;
  if (A_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = A_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)6);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)2);
  if (A_1 == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = A_1;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)4096);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)4096);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)8192);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)32);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 16, &result_2) != 0) {
    return -1;
  }
  TL_ALIGN(128) TVMFFIAny stack_5[16];
  void* B_desc = stack_5;
  if (B_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = B_desc;
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)6);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)2);
  if (B_1 == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = B_1;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)4096);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)4096);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)8192);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)32);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)3);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)0);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = (int64_t)0;
  if (__tvm_tensormap_create_tiled_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_tensormap_create_tiled", &__tvm_tensormap_create_tiled_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_3;
  result_3.type_index = kTVMFFINone;
  result_3.zero_padding = 0;
  result_3.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed, (TVMFFIAny*) stack_ffi_any, 16, &result_3) != 0) {
    return -1;
  }
  if (A_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = A_desc;
  if (B_desc == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = B_desc;
  if (C == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = C;
  (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)16);
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)32);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)640);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)73728);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = (int64_t)0;
  if (matmul_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "matmul_kernel", &matmul_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_4;
  result_4.type_index = kTVMFFINone;
  result_4.zero_padding = 0;
  result_4.v_int64 = 0;
  if (TVMFFIFunctionCall(matmul_kernel_packed, (TVMFFIAny*) stack_ffi_any, 9, &result_4) != 0) {
    return -1;
  }
  (((TVMFFIAny*)result)[0].type_index) = 70;
  (((TVMFFIAny*)result)[0].zero_padding) = 0;
  (((TVMFFIAny*)result)[0].v_int64) = 0;
  (((TVMFFIAny*)result)[0].v_ptr) = (((TVMFFIAny*)callee_allocated_output_storage)[0].v_ptr);
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return __tvm_ffi_matmul(self, args, num_args, result);
}
