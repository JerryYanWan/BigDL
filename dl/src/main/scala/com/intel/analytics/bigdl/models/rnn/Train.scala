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

import java.io.File

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, SampleToBatch}
import com.intel.analytics.bigdl.dataset.text.{Dictionary, DocumentTokenizer, LabeledSentenceToSample, TextToLabeledSentence}
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import org.apache.log4j.Logger
import org.apache.spark.SparkContext

import scala.io.Source


object Train {

  import Utils._
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {

      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark").map(conf => {
        conf.setAppName("Train rnn on text")
          .set("spark.akka.frameSize", 64.toString)
          .set("spark.task.maxFailures", "1")
        new SparkContext(conf)
      })

      val (trainSet, validationSet, dictionaryLength) = if (!sc.isDefined) {
        val logData = Source.fromFile(param.dataFolder + "/" + "train.txt").getLines().toArray
        val tokens = DataSet.array(logData.filter(!_.isEmpty))
          .transform(DocumentTokenizer())
        val dictionary = Dictionary(tokens.toLocal().data(false), param.vocabSize)
        dictionary.save(param.saveFolder)
        val isTable = if (param.batchSize > 1) true else false
        println("vocabulary size = " + dictionary.vocabSize())
        (tokens
            .transform(TextToLabeledSentence(dictionary))
            .transform(LabeledSentenceToSample(dictionary.vocabSize() + 1))
            .transform(SampleToBatch(batchSize = param.batchSize, isTable = isTable)),
          DataSet.array(Source.fromFile(param.dataFolder + "/" + "val.txt").getLines()
            .toArray.filter(!_.isEmpty))
            .transform(DocumentTokenizer())
            .transform(TextToLabeledSentence(dictionary))
            .transform(LabeledSentenceToSample(dictionary.vocabSize() + 1))
            .transform(SampleToBatch(batchSize = param.batchSize, isTable = isTable)),
          dictionary.vocabSize() + 1)
      } else {
        val tokens = DataSet.rdd(sc.get.textFile(param.dataFolder + "/" + "train.txt")
          .filter(!_.isEmpty)).transform(DocumentTokenizer())
        val dictionary = Dictionary(tokens.toDistributed().data(false),
          param.vocabSize)
        dictionary.save(param.saveFolder)
        val isTable = if (param.batchSize > 1) true else false

        (tokens
            .transform(TextToLabeledSentence(dictionary))
            .transform(LabeledSentenceToSample(dictionary.vocabSize() + 1))
            .transform(SampleToBatch(batchSize = param.batchSize, isTable = isTable)),
          DataSet.rdd(sc.get.textFile(param.dataFolder + "/" + "val.txt")
            .filter(!_.isEmpty))
            .transform(DocumentTokenizer())
            .transform(TextToLabeledSentence(dictionary))
            .transform(LabeledSentenceToSample(dictionary.vocabSize() + 1))
            .transform(SampleToBatch(batchSize = param.batchSize, isTable = isTable)),
          dictionary.vocabSize() + 1)
      }

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val curModel = SimpleRNN(
          inputSize = dictionaryLength,
          hiddenSize = param.hiddenSize,
          outputSize = dictionaryLength,
          bpttTruncate = param.bptt)
        curModel.reset()
        curModel
      }

      val state = if (param.stateSnapshot.isDefined) {
        T.load(param.stateSnapshot.get)
      } else {
        T("learningRate" -> param.learningRate,
          "momentum" -> param.momentum,
          "weightDecay" -> param.weightDecay,
          "dampening" -> param.dampening)
      }

      // Engine.init(1, param.coreNumber, false)
      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = CrossEntropyCriterion[Float](squeezeFlag = true)
      )
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      optimizer
        .setValidation(Trigger.everyEpoch, validationSet, Array(new Loss[Float]))
        .setState(state)
        .setEndWhen(Trigger.maxEpoch(param.nEpochs))
        .optimize()
    })
  }
}
