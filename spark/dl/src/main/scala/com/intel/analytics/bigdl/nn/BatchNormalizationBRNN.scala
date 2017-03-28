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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.concurrent.Future
import scala.reflect.ClassTag

class BatchNormalizationBRNN[@specialized(Float, Double) T: ClassTag](
  val gmean: Double = 0.0,
  val eps: Double = 1e-5,
  val gvar: Double = 0.0,
  val gamma: Double = 1.0,
  val beta: Double = 0.0)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  reset()


  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    output.apply1(x => ev.divide(x, ev.sqrt(ev.fromType[Float](0.001f))))
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradOutput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double): Unit = {
  }


  override def toString(): String = s"BatchNormalizationBRNN()"

  override def canEqual(other: Any): Boolean = other.isInstanceOf[BatchNormalizationBRNN[T]]

  override def equals(other: Any): Boolean = other match {
    case that: BatchNormalizationBRNN[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        gmean == that.gmean &&
        eps == that.eps &&
        gvar == that.gvar &&
        gamma == that.gamma &&
        beta == that.beta
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), gmean, eps, gvar, gamma, beta)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object BatchNormalizationBRNN {
  def apply[@specialized(Float, Double) T: ClassTag](
    gmean: Double = 0.0,
    eps: Double = 1e-3,
    gvar: Double = 0.0,
    gamma: Double = 1.0,
    beta: Double = 0.0) (implicit ev: TensorNumeric[T]): BatchNormalizationBRNN[T] = {
    new BatchNormalizationBRNN[T](gmean, eps, gvar, gamma, beta)
  }
}
