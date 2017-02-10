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

import org.scalatest.FlatSpec
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}

import scala.math.abs
import scala.util.Random

class TimeDistributedSpec extends FlatSpec with Matchers {
  "A TimeDistributed Module " should "generate correct output and grad for Linear" +
    " when time dimension is 2" in {
    val batchSize = 5
    val times = 2
    val inputDim = 3
    val outputDim = 4
    val timeDim = 2

    val input = Tensor[Float](Array(batchSize, times, inputDim)).randn()
    val linear = Linear[Float](inputDim, outputDim)
    val model = Sequential[Float]()
      .add(TimeDistributed[Float](
        timeDim = timeDim,
        inputShape = Array(times, inputDim))
        .add(linear))

    val output = model.forward(input).toTensor[Float].clone
    var i = 1
    while (i <= times) {
      val expectedOut = linear.forward(input.select(timeDim, i))
      output.select(timeDim, i) should be (expectedOut)
      i += 1
    }

    val gradOutput = Tensor[Float](Array(batchSize, times, outputDim)).randn()
    val gradInput = model.backward(input, gradOutput).toTensor[Float].clone
    i = 1
    while (i <= times) {
      val expectedOut = linear.backward(input.select(timeDim, i), gradOutput.select(timeDim, i))
      gradInput.select(timeDim, i) should be (expectedOut)
      i += 1
    }
  }

  "A TimeDistributed Module " should "generate correct output and grad for Linear " +
    "when time dimension is 1" in {
    val batchSize = 5
    val times = 2
    val inputDim = 3
    val outputDim = 4
    val timeDim = 1
    val input = Tensor[Float](Array(batchSize, times, inputDim)).randn()
    val linear = Linear[Float](inputDim, outputDim)
    val model = Sequential[Float]()
      .add(TimeDistributed[Float](
        timeDim = timeDim,
        inputShape = Array(2, 3))
        .add(linear))

    val output = model.forward(input).toTensor[Float].clone
    var i = 1
    while (i <= times) {
      val expectedOut = linear.forward(input.select(timeDim, i))
      output.select(timeDim, i) should be (expectedOut)
      i += 1
    }

    val gradOutput = Tensor[Float](Array(batchSize, times, outputDim)).randn()
    val gradInput = model.backward(input, gradOutput).toTensor[Float].clone
    i = 1
    while (i <= times) {
      val expectedOut = linear.backward(input.select(timeDim, i), gradOutput.select(timeDim, i))
      gradInput.select(timeDim, i) should be (expectedOut)
      i += 1
    }
  }

  "A TimeDistributed Module " should "generate correct output and grad for logSoftMax " +
    "when time dimension is 2" in {
    val batchSize = 5
    val times = 2
    val inputDim = 4
    val outputDim = 4
    val timeDim = 2

    val input = Tensor[Float](Array(batchSize, times, inputDim)).randn()
    val logSoftMax = LogSoftMax[Float]()
    val model = Sequential[Float]()
      .add(TimeDistributed[Float](inputShape = Array(times, inputDim))
        .add(logSoftMax))

    val output = model.forward(input).toTensor[Float].clone
    var i = 1
    while (i <= times) {
      val expectedOut = logSoftMax.forward(input.select(timeDim, i))
      output.select(timeDim, i) should be (expectedOut)
      i += 1
    }

    val gradOutput = Tensor[Float](Array(batchSize, times, outputDim)).randn()
    val gradInput = model.backward(input, gradOutput).toTensor[Float].clone
    i = 1
    while (i <= times) {
      val expectedOut = logSoftMax.backward(input.select(timeDim, i), gradOutput.select(timeDim, i))
      gradInput.select(timeDim, i) should be (expectedOut)
      i += 1
    }
  }

  "A TimeDistributed Module " should "generate correct output and grad for SpatialConvolution" in {
    val seed = 100
    RNG.setSeed(seed)

    val timeDim = 1
    val nInputPlane = 3
    val nOutputPlane = 64
    val kW = 11
    val kH = 11
    val dW = 4
    val dH = 4
    val padW = 2
    val padH = 2
    val layer1 = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)
    val layer2 = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)
    layer2.weight.copy(layer1.weight)
    layer2.bias.copy(layer1.bias)
    val outputWidth = (224 + 2 * padW - kW) / dW + 1
    val outputHeight = (224 + 2 * padH - kH) / dH + 1
    Random.setSeed(seed)
    val input1 = Tensor[Double](16, 3, 224, 224).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double]().resizeAs(input1).copy(input1)

    val wrapper = TimeDistributed[Double](
      timeDim = timeDim,
      inputShape = Array(3, 224, 224))
      .add(layer2)
    val output1 = layer1.updateOutput(input1)
    val output2 = wrapper.updateOutput(input2).toTensor[Double]

    output1.map(output2, (a, x) => {
      a should be (x)
      x
    })

    val gradOutput = Tensor[Double](output1.size).randn()
    val gradInput1 = layer1.updateGradInput(input1, gradOutput)
    val gradInput2 = wrapper.updateGradInput(input1, gradOutput)

    gradInput1.map(gradInput2, (a, x) => {
      a should be (x)
      x
    })
  }
}
