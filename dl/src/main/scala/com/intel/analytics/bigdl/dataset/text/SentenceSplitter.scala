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
import scala.collection.mutable.ArrayBuffer

import smile.nlp.tokenizer.SimpleSentenceSplitter


class SentenceSplitter() extends Transformer[String, Array[String]] {
  override def apply(prev: Iterator[String]): Iterator[Array[String]] =
    prev.map(x => {
//      val sentences = ArrayBuffer[String]()
//      val sentences_split = SimpleSentenceSplitter.getInstance.split(x)
//      var i = 0
//      while (i < sentences_split.length) {
//        sentences.append(sentences_split(i))
//      }
//      sentences.toArray
      SimpleSentenceSplitter.getInstance.split(x)
    })
}

object SentenceSplitter {
  def apply(): SentenceSplitter = new SentenceSplitter()
}


