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
 * @param timeDim the dimension for layer to roll on
 * @param inputShape the input shape for the layer
 * @param outputShape the output shape for the layer
 * @param ev
 * @tparam T
 */

class TimeDistributed[T : ClassTag] (
  timeDim: Int = 2,
  inputShape: Array[Int],
  outputShape: Array[Int])
(implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  private val batchDim: Int = 1
  private var layer: Module[T] = _
  private var fInput: Tensor[T] = _
  private var fGradOutput: Tensor[T] = _
  private var times: Int = _
  private val outputSize: Array[Int] = Array(1) ++ outputShape

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim >= 3,
      "TimeDistributed: input should be at least a 3D Tensor, e.g [batch, time, inputDim]. " +
        s"Current input.dim = ${input.dim}")
    require(modules.length == 1,
      "TimeDistributed: container can only process one layer, " +
        s"current container has ${modules.length} layers.")

    layer = modules(0)
    fInput = input.transpose(batchDim, timeDim)
    outputSize(0) = input.size(batchDim)
    output.resize(outputSize)
    times = input.size(timeDim)

    /**
     * The program will roll along the timeDim.
     * e.g. If 1 == timeDim, the layer will iterate over batchSize.
     */

    var i = 1
    while (i <= times) {
      val _output = layer.updateOutput(fInput(i)).toTensor[T]
      output.select(timeDim, i).copy(_output)
      i += 1
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    fGradOutput = gradOutput.transpose(batchDim, timeDim)
    gradInput.resizeAs(input)
    var i = 1
    while (i <= times) {
      val _gradInput = layer.updateGradInput(fInput(i), fGradOutput(i)).toTensor[T]
      gradInput.select(timeDim, i).copy(_gradInput)
      i += 1
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    var i = 1
    while (i <= times) {
      layer.accGradParameters(fInput(i), fGradOutput(i))
      i += 1
    }
  }

  override def toString(): String = {
    val str = "nn.TimeDistributed"
    str
  }
}

object TimeDistributed {
  def apply[@specialized(Float, Double) T: ClassTag](
    timeDim: Int = 2,
    inputShape: Array[Int],
    outputShape: Array[Int])
  (implicit ev: TensorNumeric[T]): TimeDistributed[T] = {
    new TimeDistributed[T](timeDim, inputShape, outputShape)
  }
}
