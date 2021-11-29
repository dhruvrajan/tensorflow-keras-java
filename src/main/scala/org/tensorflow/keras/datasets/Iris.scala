package org.tensorflow.keras.datasets

import org.tensorflow.data.{GraphLoader, GraphModeTensorFrame}
import org.tensorflow.keras.utils.{DataUtils, Keras}
import org.tensorflow.types.TFloat32
import org.tensorflow.utils.Tensors

import java.io.{BufferedReader, FileReader, IOException}
import java.util

object Iris {
  private val IRIS_ORIGIN   = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
  private val NUM_EXAMPLES  = 151
  private val INPUT_LENGTH  = 4
  private val OUTPUT_LENGTH = 3
  private val LOCAL_PREFIX  = "datasets/iris/"
  private val LOCAL_FILE    = "iris.data"

  private object COLOR extends Enumeration {
    type COLOR = Value
    val setosa, versicolor, virginica = Value

    def valueOf(name: String): COLOR = name match {
      case "setosa"     => setosa
      case "versicolor" => versicolor
      case "virginica"  => virginica
    }

//    private val value = 0d ef this (value: Int) {
//      this ()
//      this.value = value
//    }
//
//    private[datasets] def getValue = this.value
  }

  @throws[IOException]
  def main(args: Array[String]): Unit = ()

  @throws[IOException]
  def download(): Unit =
    DataUtils.getFile(LOCAL_PREFIX + LOCAL_FILE, IRIS_ORIGIN)

  @throws[IOException]
  def loadData(val_split: Double): (GraphLoader[TFloat32], GraphLoader[TFloat32]) = {
    Iris.download()
    val fr = new FileReader(Keras.kerasPath(LOCAL_PREFIX + LOCAL_FILE).toFile)
    try {
      val br = new BufferedReader(fr)
      try {
        val X = Array.ofDim[Float](NUM_EXAMPLES, INPUT_LENGTH)
        val y = Array.ofDim[Float](NUM_EXAMPLES, OUTPUT_LENGTH)
        val trainSize = (NUM_EXAMPLES * (1 - val_split)).toInt
        val XTrain = Array.ofDim[Float](trainSize, INPUT_LENGTH)
        val yTrain = Array.ofDim[Float](trainSize, OUTPUT_LENGTH)
        val XVal = Array.ofDim[Float](NUM_EXAMPLES - trainSize, INPUT_LENGTH)
        val yVal = Array.ofDim[Float](NUM_EXAMPLES - trainSize, OUTPUT_LENGTH)
        var line: String = null
        var count = 0
        while ({
          line = br.readLine()
          line != null && count < trainSize && line != ""
        }) {
          val values    = line.split(",Iris-")
          val xstring   = values(0).split(",")
          val xvector   = new Array[Float](xstring.length)
          for (i <- 0 until xstring.length) {
            xvector(i) = java.lang.Float.parseFloat(xstring(i))
          }
          val yvector   = oneHot(COLOR.valueOf(values(1)).id, COLOR.values.size)
          XTrain(count) = xvector
          yTrain(count) = yvector
          count += 1
        }

        while ({
          line = br.readLine()
          line != null && count < NUM_EXAMPLES && line != ""
        }) {
          val values = line.split(",Iris-")
          val xstring = values(0).split(",")
          val xvector = new Array[Float](xstring.length)
          for (i <- 0 until xstring.length) {
            xvector(i) = java.lang.Float.parseFloat(xstring(i))
          }
          val yvector = oneHot(COLOR.valueOf(values(1)).id, COLOR.values.size)
          XVal(count - trainSize) = xvector
          yVal(count - trainSize) = yvector
          count += 1
        }

        (
          new GraphModeTensorFrame(classOf[TFloat32], Tensors.create(XTrain), Tensors.create(yTrain)),
          new GraphModeTensorFrame(classOf[TFloat32], Tensors.create(XVal)  , Tensors.create(yVal))
        )
      } finally br.close()
    } finally fr.close()
  }

  private def oneHot(label: Int, total: Int) = {
    if (label >= total) throw new IllegalArgumentException("Invalid Index for One-Hot Vector")
    val oneHot = new Array[Float](total)
    util.Arrays.fill(oneHot, 0)
    oneHot(label) = 1.0f
    oneHot
  }
}
