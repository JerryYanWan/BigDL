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

package com.intel.analytics.bigdl.models.rnn


import java.io.PrintWriter

import com.intel.analytics.bigdl.dataset.{DataSet, LocalDataSet, MiniBatch, SampleToBatch}
import com.intel.analytics.bigdl.dataset.text.{LabeledSentence, LabeledSentenceToSample}
import com.intel.analytics.bigdl.nn.{LogSoftMax, Module}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}

import scala.util.Random

object Test {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Utils._
  val logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    testParser.parse(args, new TestParams()).map(param => {

      val vocab = new Dictionary(
        param.folder,
        param.dictionary,
        param.discard)

      val model = Module.load[Float](param.modelSnapshot.get)
      Engine.setCoreNumber(param.coreNumber)

      val logSoftMax = LogSoftMax[Float]()
      val lines = readSentence(param.folder, param.test)
      val input = lines.map(x =>
      x.map(t => vocab.getIndex(t).toFloat))

      val sentence_start_index = vocab.getIndex("SENTENCE_START")
      val sentence_end_index = vocab.getIndex("SENTENCE_END")

      var labeledInput = input.map(x =>
        new LabeledSentence[Float](x, x))

      val batchSize = 1

      var index = 0
      while (index < param.numOfWords.getOrElse(0)) {
        index += 1

        val validationSet = DataSet.array(labeledInput)
          .transform(LabeledSentenceToSample(vocab.length + 1))
          .transform(SampleToBatch(batchSize = batchSize))
          .asInstanceOf[LocalDataSet[MiniBatch[Float]]]

        val dataIter = validationSet.data(train = false)
        val predict = dataIter.map(batch => {
          require(batch.size == 1, "predict sentence one by one")
          val output = model.forward(batch.data)
            .asInstanceOf[Tensor[Float]]
          val target = logSoftMax.forward(output(output.size(1)))
            .max(1)._2.valueAt(1) - 1
          if (target == sentence_end_index) {
            sentence_start_index
          } else target
        }).toArray
        labeledInput = (labeledInput zip predict).map(x => {
            val addedInput = x._1.asInstanceOf[LabeledSentence[Float]]
              .data() ++ Array(x._2)
            new LabeledSentence[Float](addedInput, addedInput)
        })
      }

      val results = labeledInput.map(x => x.data()
        .map(t => if (t == sentence_start_index ||
            t == sentence_end_index) {
          "" } else {
          vocab.getWord(t)
        }))
      val output = results.map(x => {
        logger.info(x.mkString(" "))
        x.mkString(" ")
      })
      new PrintWriter(param.folder + "/output.txt") {
        write(output.mkString("\n")); close
      }
    })
  }
}
