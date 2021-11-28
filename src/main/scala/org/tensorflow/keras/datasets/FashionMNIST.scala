package org.tensorflow.keras.datasets

import org.tensorflow.data.GraphLoader
import org.tensorflow.data.GraphModeTensorFrame
import org.tensorflow.keras.utils.DataUtils
import org.tensorflow.keras.utils.Keras
import org.tensorflow.types.TFloat32
import org.tensorflow.utils.Pair
import org.tensorflow.utils.Tensors
import java.io.IOException

object FashionMNIST {
  private val ORIGIN_BASE   = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
  private val TRAIN_IMAGES  = "train-images-idx3-ubyte.gz"
  private val TRAIN_LABELS  = "train-labels-idx1-ubyte.gz"
  private val TEST_IMAGES   = "t10k-images-idx3-ubyte.gz"
  private val TEST_LABELS   = "t10k-labels-idx1-ubyte.gz"
  private val LOCAL_PREFIX  = "datasets/fasion_mnist/"

  /**
    * Download MNIST dataset files to the local .keras/ directory.
    *
    * @throws IOException when the download fails
    */
  @throws[IOException]
  def download(): Unit = {
    DataUtils.getFile(LOCAL_PREFIX + TRAIN_IMAGES, ORIGIN_BASE + TRAIN_IMAGES, "8d4fb7e6c68d591d4c3dfef9ec88bf0d", DataUtils.Checksum.md5)
    DataUtils.getFile(LOCAL_PREFIX + TRAIN_LABELS, ORIGIN_BASE + TRAIN_LABELS, "25c81989df183df01b3e8a0aad5dffbe", DataUtils.Checksum.md5)
    DataUtils.getFile(LOCAL_PREFIX + TEST_IMAGES, ORIGIN_BASE + TEST_IMAGES, "bef4ecab320f06d8554ea6380940ec79", DataUtils.Checksum.md5)
    DataUtils.getFile(LOCAL_PREFIX + TEST_LABELS, ORIGIN_BASE + TEST_LABELS, "bb300cfdad3c16e7a12a480ee83cd310", DataUtils.Checksum.md5)
  }

  @throws[IOException]
  def graphLoaders: Pair[GraphLoader[TFloat32], GraphLoader[TFloat32]] = { // Download MNIST files if they don't exist.
    FashionMNIST.download()
    // Read data files into arrays
    val trainImages = MNIST.readImages(Keras.kerasPath(LOCAL_PREFIX, TRAIN_IMAGES).toString)
    val trainLabels = MNIST.readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX, TRAIN_LABELS).toString)
    val testImages = MNIST.readImages(Keras.kerasPath(LOCAL_PREFIX, TEST_IMAGES).toString)
    val testLabels = MNIST.readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX + TEST_LABELS).toString)
    // Return a pair of graph loaders; train and test sets
    new Pair(
      new GraphModeTensorFrame(classOf[TFloat32], Tensors.create(trainImages) , Tensors.create(trainLabels)),
      new GraphModeTensorFrame(classOf[TFloat32], Tensors.create(testImages)  , Tensors.create(testLabels))
    )
  }

  @throws[IOException]
  def graphLoaders2D: Pair[GraphLoader[TFloat32], GraphLoader[TFloat32]] = {
    FashionMNIST.download()
    val trainImages = MNIST.readImages2D(Keras.kerasPath(LOCAL_PREFIX, TRAIN_IMAGES).toString)
    val trainLabels = MNIST.readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX, TRAIN_LABELS).toString)
    val testImages  = MNIST.readImages2D(Keras.kerasPath(LOCAL_PREFIX, TEST_IMAGES).toString)
    val testLabels  = MNIST.readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX + TEST_LABELS).toString)
    new Pair(
      new GraphModeTensorFrame(classOf[TFloat32], Tensors.create(trainImages) , Tensors.create(trainLabels)),
      new GraphModeTensorFrame(classOf[TFloat32], Tensors.create(testImages)  , Tensors.create(testLabels))
    )
  }
}
