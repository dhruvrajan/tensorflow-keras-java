package org.tensorflow.keras.models

import org.tensorflow.{Graph, Operand, Session, Tensor}
import org.tensorflow.data.GraphLoader
import org.tensorflow.keras.layers.{Input, Layer}
import org.tensorflow.keras.losses.Loss
import org.tensorflow.keras.metrics.Metric
import org.tensorflow.keras.optimizers.Optimizer
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.types.TFloat32
import org.tensorflow.types.family.TNumber

import java.{util => ju}

object Sequential {
  def apply[T <: TNumber](dtype: Class[T], firstLayer: Input[T], layers: Layer[T]*): Sequential[T] =
    new Sequential[T](dtype, firstLayer, layers)
}

class Sequential[T <: TNumber](dtype: Class[T], val firstLayer: Input[T], layers0: Seq[Layer[T]])
  extends Model[T](dtype) {

  private var optimizer: Optimizer[T] = null
  final private val layers: ju.List[Layer[T]] = ju.Arrays.asList(layers0: _*)
  private var loss: Loss = null
  private var _metrics      : ju.List[Metric]       = null
  private var trainableVars : ju.List[Variable[T]]  = null
  private var initializerOpSq: ju.List[Operand [T]]  = null

  def addLayer(layer: Layer[T]): Sequential[T] = {
    layers.add(layer)
    this
  }

  override def computeOutputShape(inputShape: Shape): Shape =
    throw new UnsupportedOperationException("Can't call computeOutputShape on Model")

  final override def call(tf: Ops, inputs: Seq[Operand[T]], training: Option[Boolean]): Operand[T] =
    callOne(tf, inputs.head)

  def callOne(tf: Ops, in: Operand[T]): Operand[T] = {
    var out: Operand[T] = in

    layers.forEach { layer =>
      out = layer.apply(tf, out)
    }
    out
  }

  override protected def build(tf: Ops, inputShape: Shape): Unit =
    throw new UnsupportedOperationException("Cannot build a sequential model")

  override def compile(tf: Ops, optimizer: Optimizer[T], loss: Loss, metrics: ju.List[Metric]): Unit = {
    this.loss           = loss
    this._metrics       = metrics
    this.optimizer      = optimizer
    // Targets for training loop
    this.trainableVars  = new ju.ArrayList[Variable [T]]
    initializerOpSq = new ju.ArrayList[Operand  [T]]
    // Build layers
    this.firstLayer.build(tf, dtype)
    var inputShape: Shape = firstLayer.computeOutputShape

    layers.forEach { layer =>
      layer.build(tf, inputShape, dtype)
      this.trainableVars.addAll(layer.trainableWeights)
      initializerOpSq.addAll(layer.initializerOps)
      inputShape = layer.computeOutputShape(inputShape)
    }
    this.optimizer.build(tf, dtype)
  }

  override def fit(tf: Ops, train: GraphLoader[T], test: GraphLoader[T], epochs: Int, batchSize: Int): Unit = {
    val g: Graph = tf.scope.env.asInstanceOf[Graph] // XXX TODO ugly
    try {
      val session: Session = new Session(g)
      try {
        runTrainingLoop   (tf, session, train, epochs, batchSize, training = true)
        runPredictionLoop (tf, session, test, batchSize)
      } finally {
        session.close()
      }
    } finally {
      g.close()
    }
  }

  private def runPredictionLoop(tf: Ops, session: Session, data: GraphLoader[T], batchSize: Int): Unit =
    runTrainingLoop(tf, session, data, 1, batchSize, training = false)

  private def runTrainingLoop(tf: Ops, session: Session, data: GraphLoader[T], epochs: Int, batchSize: Int,
                              training: Boolean): Unit = {
    data.batch(batchSize)
    data.build(tf)

    val dataOps: Array[Operand[T]] = data.getBatchOperands
    var runner: Session#Runner = null
    val XOp: Operand[T] = dataOps(0)
    val yOp: Operand[T] = dataOps(1)
    // Compute Output / Loss / Accuracy
    val yTrue: Operand[T] = yOp
    val yPred: Operand[T] = this.apply(tf, XOp)

    val batchLoss     : Operand[T] = loss.apply(tf, getDtype, yTrue, yPred)
    val batchAccuracy : Operand[T] = _metrics.get(0).apply(tf, getDtype, yTrue, yPred)

    val minimize: ju.List[Operand[T]] =
      if (training) {
        optimizer.minimize(tf, batchLoss, this.trainableVars)
      } else {
        null
      }

    if (training) {
      runner = session.runner
      // Run initializer ops

      initializerOpSq.forEach { op =>
        runner.addTarget(op)
      }
      runner.run()
    }

    for (epoch <- 0 until epochs) {
      var trainEpochAccuracy = 0f
      var trainEpochLoss     = 0f
      // Load Batches
      var i = 0
      while (i < data.numBatches) {
        runner = session.runner
        data.feedSessionRunner(runner, i)
        if (training) {
          minimize.forEach { op =>
            runner.addTarget(op)
          }
        }
        runner.fetch(batchLoss)
        runner.fetch(batchAccuracy)
        val values: ju.List[Tensor] = runner.run()
        val lossTensor: TFloat32 = values.get(0).asInstanceOf[TFloat32]
        try {
          val accuracyTensor: TFloat32 = values.get(1).asInstanceOf[TFloat32]
          try {
            trainEpochAccuracy += accuracyTensor.getFloat() / data.numBatches
            trainEpochLoss     += lossTensor    .getFloat() / data.numBatches
          } finally {
            accuracyTensor.close()
          }
        } finally {
          lossTensor.close()
        }

        i += 1
      }
      if (training) {
        System.out.println("Epoch " + epoch + " train accuracy: " + trainEpochAccuracy + "  loss: " + trainEpochLoss)
      } else {
        System.out.println("Test accuracy: " + trainEpochAccuracy + " loss: " + trainEpochLoss)
      }
    }
  }
}
