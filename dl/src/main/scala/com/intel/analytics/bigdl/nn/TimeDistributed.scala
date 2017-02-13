/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This layer is intended to wrap layers that do not support higher Dimensions.
 * For instance, the Linear layer do not accept 3D input. The TimeDistributed
 * Layer can wrap the Linear layer, accept 3D input, and feed a sequence of 2D
 * Tensor to the wrapped Linear layer by reshaping the first two dimensions.
 *
 * @param ev
 * @tparam T
 */

class TimeDistributed[T : ClassTag] ()
(implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  private var layer: Module[T] = _
  private var fInput: Tensor[T] = _
  private var fGradOutput: Tensor[T] = _
  private var inputSize: Array[Int] = _
  private var gradOutputSize: Array[Int] = _
  private var outputSize: Array[Int] = _

  private def combine(src: Array[Int], target: Array[Int]): Unit = {
    require(src.length == target.length + 1,
      "In Recurrent: combine method requires src.length == target.length + 1" +
        s" Current src.length = ${src.length}" +
        s" Current target.length = ${target.length}")

    target(0) = src(0) * src(1)
    var j = 1
    while (j < target.length) {
      target(j) = src(j + 1)
      j += 1
    }
  }

  private def split(src: Array[Int], target: Array[Int], dim1: Int, dim2: Int): Unit = {
    require(src.length == target.length - 1,
      "In Recurrent: split method requires src.length == target.length - 1" +
        s" Current src.length = ${src.length}" +
        s" Current target.length = ${target.length}")
    require(dim1 * dim2 == src(0),
    "In Recurrent: split method requires dim1 * dim2 == src(0), " +
      s"Current dim1 = ${dim1}, dim2 = ${dim2}, src(0) = ${src(0)}")

    target(0) = dim1
    target(1) = dim2
    var j = 1
    while (j < src.length) {
      target(j + 1) = src(j)
      j += 1
    }
  }


  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim >= 3,
      "TimeDistributed: input should be at least a 3D Tensor, e.g [batch, time, inputDim]. " +
        s"Current input.dim = ${input.dim}")
    require(modules.length == 1,
      "TimeDistributed: container can only process one layer, " +
        s"current container has ${modules.length} layers.")

    layer = modules(0)

    if (inputSize == null) {
      inputSize = new Array[Int](input.size.length - 1)
    }
    if (outputSize == null) {
      outputSize = new Array[Int](input.size.length)
    }

    /**
     * combine: [B, T, D] => [B * T, D]
     * split:   [B * T, D] => [B, T, D]
     */
    combine(input.size, inputSize)
    fInput = input.reshape(inputSize)
    val _output = layer.updateOutput(fInput).toTensor[T]
    split(_output.size, outputSize, input.size(1), input.size(2))
    output = _output.reshape(outputSize)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradOutputSize = inputSize
    combine(gradOutput.size, gradOutputSize)
    fGradOutput = gradOutput.reshape(gradOutputSize)
    val _gradInput = layer.updateGradInput(fInput, fGradOutput).toTensor[T]
    gradInput = _gradInput.reshape(input.size)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    layer.accGradParameters(fInput, fGradOutput)
  }

  override def toString(): String = {
    val str = "nn.TimeDistributed"
    str
  }
}

object TimeDistributed {
  def apply[@specialized(Float, Double) T: ClassTag]()
  (implicit ev: TensorNumeric[T]): TimeDistributed[T] = {
    new TimeDistributed[T]()
  }
}
