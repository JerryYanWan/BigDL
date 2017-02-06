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
import org.scalatest.{FlatSpec, Matchers}

import scala.math.abs

class TimeDistributedSpec extends FlatSpec with Matchers {
  "A TimeDistributed Module " should "generate correct output and grad" in {
    val batchSize = 5
    val times = 2
    val inputDim = 3
    val outputDim = 4

    val input = Tensor[Float](Array(batchSize, times, inputDim)).randn()
    val linear = Linear[Float](inputDim, outputDim)
    val model = Sequential[Float]()
      .add(TimeDistributed[Float]()
        .add(linear))

    val output = model.forward(input).asInstanceOf[Tensor[Float]]
    var i = 1
    while (i <= times) {
      val expectedOut = linear.forward(input.select(2, i))
      output.select(2, i) should be (expectedOut)
      i += 1
    }
  }
}
