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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import scala.reflect.ClassTag

/**
 * This class is intended to support inputs with 3 or more dimensions.
 * Apply Negative Log Likelihood Criterion to every temporal slice of an input.
 * @param weights
 * @param sizeAverage
 * @param timeDim
 */

class ClassNLLCriterion3d[T : ClassTag](
  weights: Tensor[T] = null,
  sizeAverage: Boolean = true,
  timeDim: Int = 2)
(implicit ev: TensorNumeric[T]) extends ClassNLLCriterion[T] {

  private val batchDim: Int = 1
  private var fInput: Tensor[T] = _
  private var fTarget: Tensor[T] = _
  private var times: Int = _

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.dim() >= 3,
      "input should be at least a 3D Tensor, e.g.[batch, time, inputDim]. "
        + s"Current input.dim = ${input.dim}")

    fInput = input.transpose(batchDim, timeDim)
    fTarget = target.transpose(batchDim, timeDim)
    times = input.size(timeDim)

    var i = 1
    while (i <= times) {
      val _output = super.updateOutput(fInput(i), fTarget(i))
      output = ev.plus(output, _output)
      i += 1
    }
    output = ev.divide(output, ev.fromType[Int](times))
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)
    var i = 1
    while (i <= times) {
      val _gradInput = super.updateGradInput(fInput(i), fTarget(i)).toTensor[T]
      gradInput.select(timeDim, i).copy(_gradInput)
      i += 1
    }
    gradInput
  }

  object ClassNLLCriterion3d {
    def apply[@specialized(Float, Double) T: ClassTag](
        weights: Tensor[T] = null,
        sizeAverage: Boolean = true,
        timeDim: Int = 2)(implicit ev: TensorNumeric[T]) : ClassNLLCriterion3d[T] = {
      new ClassNLLCriterion3d[T](weights, sizeAverage, timeDim)
    }
  }
}

