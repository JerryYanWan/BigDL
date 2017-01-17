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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, Activity, TensorCriterion}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * This criterion combines LogSoftMax and ClassNLLCriterion in one single class.
 *
 * @param weights A tensor assigning weight to each of the classes
 */

@SerialVersionUID(- 5446858218997354022L)
class CrossEntropyCriterion[T: ClassTag](
   val weights: Tensor[T] = null,
   val squeezeFlag: Boolean = false)
   (implicit ev: TensorNumeric[T]) extends AbstractCriterion[Tensor[T], Activity, T]{
  private val lsm = new LogSoftMax[T]()
  private val nll = new ClassNLLCriterion[T](weights)

  private val joinTableTarget = JoinTable(1, 0)

  private def toTensor(target: Activity): Tensor[T] = {
    val _target = target match {
      case tensorTarget: Tensor[T] =>
        if (squeezeFlag) tensorTarget.squeeze()
        tensorTarget
      case tableTarget: Table =>
        joinTableTarget.updateOutput(tableTarget)
      case _ => throw new IllegalArgumentException("Target should be Activity")
    }
    _target
  }

  override def updateOutput(input: Tensor[T], target: Activity): T = {
    val _target = toTensor(target)
    if (squeezeFlag) input.squeeze()
    lsm.updateOutput(input)
    nll.updateOutput(lsm.output, _target)
    output = nll.output
    output
  }

  override def updateGradInput(input: Tensor[T], target: Activity): Tensor[T] = {
    val _target = toTensor(target)
    val size = input.size()
    var _gradInput = Tensor[T]()
    if (squeezeFlag) input.squeeze()
    _gradInput = nll.updateGradInput(lsm.output, _target)
    lsm.updateGradInput(input, _gradInput)
    gradInput.resizeAs(lsm.gradInput).copy(lsm.gradInput).view(size)
    gradInput.toTensor
  }

  override def equals(other: Any): Boolean = other match {
    case that: CrossEntropyCriterion[T] =>
      (that.eq(this)) &&
        weights == that.weights
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(weights)
    state.map(getHashCode).foldLeft(0)((a, b) => 37 * a + b)
  }

  override def toString(): String = {
    s"nn.CrossEntropyCriterion"
  }
}

object CrossEntropyCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      weights: Tensor[T] = null,
      squeezeFlag: Boolean = false)
      (implicit ev: TensorNumeric[T]) : CrossEntropyCriterion[T] = {
    new CrossEntropyCriterion[T](weights, squeezeFlag)
  }
}
