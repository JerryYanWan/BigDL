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

package com.intel.analytics.bigdl.example.deepspeech2

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.language.existentials
import scala.reflect.ClassTag

class DeepSpeech2[T : ClassTag]()
                               (implicit ev: TensorNumeric[T]) {

  /**
   * The configuration of convolution for dp2.
   */
  val nInputPlane = 1
  val nOutputPlane = 1152
  val kW = 11
  val kH = 13
  val dW = 3
  val dH = 1
  val padW = 5
  val padH = 0
  val conv = SpatialConvolution(nInputPlane, nOutputPlane,
    kW, kH, dW, dH, padW, padH)

  val nOutputDim = 2
  val outputHDim = 3
  val outputWDim = 4
  val inputSize = nOutputPlane
  val hiddenSize = nOutputPlane
  val nChar = 29
  val brnn = BiRecurrent()
    .add(RnnCell(inputSize, hiddenSize, ReLU[T]()))
  val linear1 = TimeDistributed[T](Linear[T](hiddenSize, hiddenSize, withBias = false))
  val linear2 = TimeDistributed[T](Linear[T](hiddenSize, nChar, withBias = false))

  val model = Sequential[T]()
    .add(conv)
    .add(Transpose(Array((nOutputDim, outputWDim), (outputHDim, outputWDim))))
    .add(Squeeze(4))
    .add(brnn)
//    .add(linear1)
//    .add(ReLU[T]())
//    .add(linear2)

  def reset(): Unit = {
    conv.weight.fill(ev.fromType[Double](0.001))
    conv.bias.fill(ev.fromType[Double](0.0))
  }

  def evaluate(): Unit = {

    val inputData = InputProcess.get()
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 1, 13, 398))
    val output = model.forward(input).toTensor[T]

    println(output)
    println(output.size.mkString(","))
  }


  def setConvWeight(weights: Array[T]): Unit = {
    val temp = Tensor[T](Storage(weights), 1, Array(1, 1152, 1, 13, 11))
    conv.weight.set(Storage[T](weights), 1, conv.weight.size())

  }

  def setBiRNN0Weight(weights: Array[T]): Unit = {
    val temp = Tensor[T](Storage(weights), 1, Array(1, 1152, 1, 13, 11))
    val leftRnn = brnn.layer.parameters()._1
    val rightRnn = brnn.revLayer.parameters()._2

    var offset = 1
    for (i <- 0 until leftRnn.length) {
      leftRnn(i).set(Storage[T](weights), offset, leftRnn(i).size)
      offset += leftRnn(i).nElement()
    }
    for (i <- 0 until rightRnn.length) {
      rightRnn(i).set(Storage[T](weights), offset, rightRnn(i).size)
      offset += rightRnn(i).nElement()
    }
  }

}

object DeepSpeech2 {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("com").setLevel(Level.WARN)

    val spark = SparkSession.builder().master("local").appName("test").getOrCreate()

    val origin = spark.sparkContext.textFile("/tmp/conv.txt")
      .map(_.split("\\s+").map(_.toDouble)).flatMap(t => t).collect()
    val weights = convert(origin)

    val birnnOrigin0 = spark.sparkContext.textFile("/tmp/weights/layer0.txt")
      .map(_.split(",").map(_.toDouble)).flatMap(t => t).collect()
    val weightsBirnn0 = convertBiRNN(birnnOrigin0)

    val dp2 = new DeepSpeech2[Double]()
    dp2.reset()
    dp2.setConvWeight(weights)
    dp2.setBiRNN0Weight(weightsBirnn0)
    dp2.evaluate()
  }

  def convert(origin: Array[Double]): Array[Double] = {
    val channel = 1152
    val buffer = new ArrayBuffer[Double]()
    val groups = origin.grouped(1152).toArray

    for(i <- 0 until channel)
      for (j <- 0 until groups.length)
        buffer += groups(j)(i)
    buffer.toArray
  }

  def convertBiRNN(origin: Array[Double]): Array[Double] = {
    val nIn = 1152
    val nOut = 1152
    val heights = 2 * (nIn + nOut + 1)
    val widths = nOut

    val buffer = new ArrayBuffer[Double]()
    val groups = origin.grouped(nOut).toArray

    /**
     * left-to-right rnn U, W, and bias
     */

    for (j <- 0 until nOut) {
      for (i <- 0 until nIn) {
        buffer += groups(i)(j)
      }
    }
    for (j <- 0 until nOut) {
      for (i <- 2 * nIn until (2 * nIn + nOut)) {
        buffer += groups(i)(j)
      }
    }
    for (j <- 0 until nOut) {
      for (i <- 2 * (nIn + nOut + 1) - 2 until 2 * (nIn + nOut + 1) - 1) {
        buffer += groups(i)(j)
      }
    }

    /**
     * right-to-left rnn U, W, and bias
     */

    for (j <- 0 until nOut) {
      for (i <- nIn until 2 * nIn) {
        buffer += groups(i)(j)
      }
    }
    for (j <- 0 until nOut) {
      for (i <- (2 * nIn + nOut) until (2 * nIn + 2 * nOut)) {
        buffer += groups(i)(j)
      }
    }
    for (j <- 0 until nOut) {
      for (i <- 2 * (nIn + nOut + 1) - 1 until 2 * (nIn + nOut + 1)) {
        buffer += groups(i)(j)
      }
    }
    buffer.toArray
  }
}
