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
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class TimeDistributed[T : ClassTag] (outputDim: Int, dim: Int = 2)
(implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  private val fInput: Tensor[T] = Tensor[T]()
  private val fGradOutput: Tensor[T] = Tensor[T]()
  private var times = 0
  private val size: Array[Int] = new Array(2)
  private var layer: Module[T] = _

  @inline
  private def getDim(input: Tensor[T]): Unit = {
    size(0) = input.size(1)
    size(1) = input.size(2)
    times = input.size(dim)
    var i = dim
    while (i < 3) {
      size(i - 1) = input.size(i + 1)
      i += 1
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 3,
      "TimeDistributed: input should be a 3D Tensor, e.g [batch, time, inputDim]")
    require(modules.length == 1,
      "TimeDistributed: can only execute one module at a time")

    getDim(input)
    layer = modules(0)
    fInput.resize(Array(times, size(0), size(1)))
    output.resize(Array(size(0), times, input.size(3)))
    var i = 1
    while (i <= times) {
      fInput(i).copy(input.select(2, i))
      println(fInput(i).size.mkString(","))
      val _output = layer.updateOutput(fInput(i)).asInstanceOf[Tensor[T]]
      println(_output.size.mkString(","))
      output.select(2, i).copy(
        layer.updateOutput(fInput(i)).asInstanceOf[Tensor[T]])
      i += 1
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    fGradOutput.resize(Array(times, gradOutput.size(1), gradOutput.size(3)))
    var i = 1
    while (i <= times) {
      fGradOutput(i).copy(gradOutput.select(2, i))
      val _gradInput = layer.updateGradInput(fInput(i), fGradOutput(i))
      gradInput.select(2, i).copy(fGradOutput(i))
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
    var str = "nn.TimeDistributed"
    str
  }
}

object TimeDistributed {
  def apply[@specialized(Float, Double) T: ClassTag](outputDim: Int, dim: Int = 2)
  (implicit ev: TensorNumeric[T]): TimeDistributed[T] = {
    new TimeDistributed[T](outputDim, dim)
  }
}
