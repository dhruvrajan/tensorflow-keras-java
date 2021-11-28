package org.tensorflow.keras.datasets

import org.tensorflow.data.GraphLoader
import org.tensorflow.data.GraphModeTensorFrame
import org.tensorflow.keras.utils.DataUtils
import org.tensorflow.keras.utils.Keras
import org.tensorflow.types.TFloat32
import org.tensorflow.utils.Pair
import org.tensorflow.utils.Tensors
import java.io.DataInputStream
import java.io.FileInputStream
import java.io.IOException
import java.util.zip.GZIPInputStream

/**
  * Code based on example found at:
  * https://github.com/karllessard/models/tree/master/samples/languages/java/mnist/src/main/java/org/tensorflow/model/sample/mnist
  * <p>
  * Utility for downloading and using MNIST data with a local keras installation.
  */
object MNIST {
  private val IMAGE_MAGIC     = 2051
  private val LABELS_MAGIC    = 2049
  private val OUTPUT_CLASSES  = 10
  private val TRAIN_IMAGES    = "train-images-idx3-ubyte.gz"
  private val TRAIN_LABELS    = "train-labels-idx1-ubyte.gz"
  private val TEST_IMAGES     = "t10k-images-idx3-ubyte.gz"
  private val TEST_LABELS     = "t10k-labels-idx1-ubyte.gz"
  private val ORIGIN_BASE     = "http://yann.lecun.com/exdb/mnist/"
  private val LOCAL_PREFIX    = "datasets/mnist/"

  /**
    * Downloads MNIST dataset files to the local .keras/ directory.
    *
    * @throws IOException when the download fails
    */
  @throws[IOException]
  def download(): Unit = {
    DataUtils.getFile(LOCAL_PREFIX + TRAIN_IMAGES , ORIGIN_BASE + TRAIN_IMAGES, "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609", DataUtils.Checksum.sha256)
    DataUtils.getFile(LOCAL_PREFIX + TRAIN_LABELS , ORIGIN_BASE + TRAIN_LABELS, "fcdfeedb53b53c99384b2cd314206a08fdf6aa97070e19921427a179ea123d19", DataUtils.Checksum.sha256)
    DataUtils.getFile(LOCAL_PREFIX + TEST_IMAGES  , ORIGIN_BASE + TEST_IMAGES , "beb4b4806386107117295b2e3e08b4c16a6dfb4f001bfeb97bf25425ba1e08e4", DataUtils.Checksum.sha256)
    DataUtils.getFile(LOCAL_PREFIX + TEST_LABELS  , ORIGIN_BASE + TEST_LABELS , "986c5b8cbc6074861436f5581f7798be35c7c0025262d33b4df4c9ef668ec773", DataUtils.Checksum.sha256)
  }

  @throws[IOException]
  def graphLoaders: Pair[GraphLoader[TFloat32], GraphLoader[TFloat32]] = { // Download MNIST files if they don't exist.
    MNIST.download()
    // Read data files into arrays
    val trainImages = readImages(Keras.kerasPath(LOCAL_PREFIX, TRAIN_IMAGES).toString)
    val trainLabels = readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX, TRAIN_LABELS).toString)
    val testImages  = readImages(Keras.kerasPath(LOCAL_PREFIX, TEST_IMAGES).toString)
    val testLabels  = readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX + TEST_LABELS).toString)
    // Return a pair of graph loaders; train and test sets
    new Pair(
      new GraphModeTensorFrame(classOf[TFloat32], Tensors.create(trainImages) , Tensors.create(trainLabels)),
      new GraphModeTensorFrame(classOf[TFloat32], Tensors.create(testImages)  , Tensors.create(testLabels))
    )
  }

  @throws[IOException]
  def graphLoaders2D: Pair[GraphLoader[TFloat32], GraphLoader[TFloat32]] = {
    MNIST.download()
    val trainImages = readImages2D(Keras.kerasPath(LOCAL_PREFIX, TRAIN_IMAGES).toString)
    val trainLabels = readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX, TRAIN_LABELS).toString)
    val testImages  = readImages2D(Keras.kerasPath(LOCAL_PREFIX, TEST_IMAGES).toString)
    val testLabels  = readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX + TEST_LABELS).toString)
    new Pair(
      new GraphModeTensorFrame(classOf[TFloat32], Tensors.create(trainImages) , Tensors.create(trainLabels)),
      new GraphModeTensorFrame(classOf[TFloat32], Tensors.create(testImages)  , Tensors.create(testLabels))
    )
  }

  /**
    * Reads MNIST images into an array, given the image datafile path.
    *
    * @param imagesPath MNIST image datafile path
    * @return an array of shape (# examples, # pixels) containing the image data
    * @throws IOException when the file reading fails.
    */
  @throws[IOException]
  private[datasets] def readImages(imagesPath: String) = try {
    val inputStream = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)))
    try {
      if (inputStream.readInt != IMAGE_MAGIC) throw new IllegalArgumentException("Invalid Image Data File")
      val numImages = inputStream.readInt
      val rows = inputStream.readInt
      val cols = inputStream.readInt
      readImageBuffer(inputStream, numImages, rows * cols)
    } finally if (inputStream != null) inputStream.close()
  }

  @throws[IOException]
  private[datasets] def readImages2D(imagesPath: String) = try {
    val inputStream = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)))
    try {
      if (inputStream.readInt != IMAGE_MAGIC) throw new IllegalArgumentException("Invalid Image Data File")
      val numImages = inputStream.readInt
      val rows = inputStream.readInt
      val cols = inputStream.readInt
      readImageBuffer2D(inputStream, numImages, rows)
    } finally if (inputStream != null) inputStream.close()
  }

  /**
    * Reads MNIST label files into an array, given a label datafile path.
    *
    * @param labelsPath MNIST label datafile path
    * @return an array of shape (# examples, # classes) containing the label data
    * @throws IOException when the file reading fails.
    */
  @throws[IOException]
  private[datasets] def readLabelsOneHot(labelsPath: String) = try {
    val inputStream = new DataInputStream(new GZIPInputStream(new FileInputStream(labelsPath)))
    try {
      if (inputStream.readInt != LABELS_MAGIC) throw new IllegalArgumentException("Invalid Label Data File")
      val numLabels = inputStream.readInt
      readLabelBuffer(inputStream, numLabels)
    } finally if (inputStream != null) inputStream.close()
  }

  @throws[IOException]
  private def readBatchedBytes(inputStream: DataInputStream, batches: Int, bytesPerBatch: Int) = {
    val entries = Array.ofDim[Byte](batches, bytesPerBatch)
    for (i <- 0 until batches) {
      inputStream.readFully(entries(i))
    }
    entries
  }

  @throws[IOException]
  private def readImageBuffer2D(inputStream: DataInputStream, numImages: Int, imageWidth: Int) = {
    val unsignedEntries = Array.ofDim[Float](numImages, imageWidth, imageWidth)
    for (i <- 0 until unsignedEntries.length) {
      val entries = readBatchedBytes(inputStream, imageWidth, imageWidth)
      for (j <- 0 until unsignedEntries(0).length) {
        for (k <- 0 until unsignedEntries(0)(0).length) {
          unsignedEntries(i)(j)(k) = (entries(j)(k) & 0xFF).toFloat / 255.0f
        }
      }
    }
    unsignedEntries
  }

  @throws[IOException]
  private def readImageBuffer(inputStream: DataInputStream, numImages: Int, imageSize: Int) = {
    val entries = readBatchedBytes(inputStream, numImages, imageSize)
    val unsignedEntries = Array.ofDim[Float](numImages, imageSize)
    for (i <- 0 until unsignedEntries.length) {
      for (j <- 0 until unsignedEntries(0).length) {
        unsignedEntries(i)(j) = (entries(i)(j) & 0xFF).toFloat / 255.0f
      }
    }
    unsignedEntries
  }

  @throws[IOException]
  private def readLabelBuffer(inputStream: DataInputStream, numLabels: Int) = {
    val entries = readBatchedBytes(inputStream, numLabels, 1)
    val labels = Array.ofDim[Float](numLabels, OUTPUT_CLASSES)
    for (i <- 0 until entries.length) {
      labelToOneHotVector(entries(i)(0) & 0xFF, labels(i))
    }
    labels
  }

  private def labelToOneHotVector(label: Int, oneHot: Array[Float]): Unit = {
    if (label >= oneHot.length) throw new IllegalArgumentException("Invalid Index for One-Hot Vector")
    oneHot(label) = 1.0f
  }
}
