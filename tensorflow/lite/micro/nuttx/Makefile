# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Warning!!! This Makefile can only used by nuttx-apps, will:
#	read	APPDIR
#	read	INCDIR_PREFIX
#	read	TFLM_DIR
#	write	COMMON_FLAGS
#	write	CXXSRCS
NEW_SRCS :=
OLD_SRCS :=

ifneq ($(CONFIG_TFLITEMICRO_TOOL),)
  # Enable all the third-party hardware operators.
  COMMON_FLAGS += -DTFLITE_MODEL_COMPILER
  CXXSRCS += $(TFLM_DIR)/tensorflow/lite/micro/kernels/beco/conv.cc
  CXXSRCS += $(TFLM_DIR)/tensorflow/lite/micro/kernels/beco/fully_connected.cc
else
  # Substitute operators with hardware support sepectively.
ifneq ($(CONFIG_BES_BECO),)
  COMMON_FLAGS += ${INCDIR_PREFIX}$(APPDIR)/vendor/bes/$(CONFIG_BES_DDK_DIR)/include/beco
  COMMON_FLAGS += ${INCDIR_PREFIX}$(APPDIR)/vendor/bes/$(CONFIG_BES_DDK_DIR)/include/beco/beco_nn
  COMMON_FLAGS += ${INCDIR_PREFIX}$(APPDIR)/vendor/bes/$(CONFIG_BES_DDK_DIR)/include/cmsis_dsp
  COMMON_FLAGS += ${INCDIR_PREFIX}$(APPDIR)/vendor/bes/$(CONFIG_BES_DDK_DIR)/platform/cmsis/inc
  COMMON_FLAGS += ${INCDIR_PREFIX}$(APPDIR)/vendor/bes/$(CONFIG_BES_DDK_DIR)/platform/hal
endif

ifneq ($(CONFIG_BES_BECO_CONV),)
  NEW_SRCS += $(TFLM_DIR)/tensorflow/lite/micro/kernels/beco/conv.cc
endif

ifneq ($(CONFIG_BES_BECO_FULLY_CONNECTED),)
  NEW_SRCS += $(TFLM_DIR)/tensorflow/lite/micro/kernels/beco/fully_connected.cc
endif

OLD_SRCS += $(addprefix $(TFLM_DIR)/tensorflow/lite/micro/kernels/, $(notdir $(NEW_SRCS)))
OLD_SRCS += $(addprefix $(TFLM_DIR)/tensorflow/lite/micro/kernels/cmsis_nn/, $(notdir $(NEW_SRCS)))
CXXSRCS := $(filter-out $(OLD_SRCS), $(CXXSRCS))
CXXSRCS += $(NEW_SRCS)
endif
