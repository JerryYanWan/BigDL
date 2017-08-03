/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Applies the rectified linear unit (ReLU) function element-wise to the input Tensor
 * Thus the output is a Tensor of the same dimension
 * ReLU function is defined as:
 * f(x) = max(0, x)
 *
 * @param ip inplace mode
 */
@SerialVersionUID(1208478077576570643L)
class ReLU[T: ClassTag](ip: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends Threshold[T](0, 0, ip) {
  @transient
  private var buffer1: Array[T] = null

  private def calcRegularWave(input: Tensor[T]): Unit = {
    if (buffer1 == null || buffer1.length < input.nElement) {
      buffer1 = new Array[T](input.nElement)
    }
    val offset = input.storageOffset - 1
    var i = 0
    while (i < input.nElement) {
      if (ev.isGreater(input.storage.array()(offset + i), ev.zero)) {
        buffer1(i) = ev.one
      }
      i += 1
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    calcRegularWave(input)
    if (ip) {
      output = input
    } else {
      output.resizeAs(input)
    }
    ev.vMul(input.nElement,
      input.storage.array,
      input.storageOffset - 1,
      buffer1,
      0,
      output.storage.array,
      output.storageOffset - 1)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (ip) {
      gradInput = gradOutput
    } else {
      gradInput.resizeAs(gradOutput)
    }
    ev.vMul(gradOutput.nElement,
      gradOutput.storage.array,
      gradOutput.storageOffset - 1,
      buffer1,
      0,
      gradInput.storage.array,
      gradInput.storageOffset - 1)
    gradInput
  }

  override def clearState() : this.type = {
    super.clearState()
    buffer1 = null
    this
  }
}

object ReLU {
  def apply[@specialized(Float, Double) T: ClassTag](
      ip: Boolean = false)(implicit ev: TensorNumeric[T]) : ReLU[T] = {
    new ReLU[T](ip)
  }
}
