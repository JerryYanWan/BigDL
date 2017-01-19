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

package com.intel.analytics.bigdl.dataset.text

import com.intel.analytics.bigdl.dataset.Transformer

import scala.collection.Iterator

object TextToLabeledSentence {
  def apply(dictionary: Dictionary)
  : TextToLabeledSentence =
    new TextToLabeledSentence(dictionary)
}

class TextToLabeledSentence(dictionary: Dictionary)
  extends Transformer[Array[String], LabeledSentence[Float]] {
  private val buffer = new LabeledSentence[Float]()

  override def apply(prev: Iterator[Array[String]]): Iterator[LabeledSentence[Float]] = {
    prev.map(sentence => {
      val indexes = sentence.map(x =>
      dictionary.word2Index().getOrElse(x, dictionary.vocabSize()))
      val nWords = indexes.length - 1
      val data = indexes.take(nWords)
      val label = indexes.drop(1)
      val input = new Array[Float](nWords)
      val target = new Array[Float](nWords)
      var i = 0
      while (i < nWords) {
        input(i) = data(i).toFloat
        target(i) = label(i).toFloat
        i += 1
      }
      buffer.copy(input, target)
    })
  }
}
