package org.tensorflow.keras.models

import org.tensorflow.data.GraphLoader
import org.tensorflow.keras.layers.Layer
import org.tensorflow.keras.losses.{Loss, Losses}
import org.tensorflow.keras.metrics.{Metric, Metrics}
import org.tensorflow.keras.optimizers.{Optimizer, Optimizers}
import org.tensorflow.op.Ops
import org.tensorflow.types.family.TNumber

import java.{util => ju}

object Model {
  object CompileOptions {
    def builder[T <: TNumber] = new CompileOptions.Builder[T]

    class Builder[T <: TNumber]() {
      final private val options = new Model.CompileOptions[T]

      def setLoss(lossType: Losses): CompileOptions.Builder[T] = setLoss(Losses.select(lossType))

      def setLoss(loss: Loss): CompileOptions.Builder[T] = {
        options._loss = loss
        this
      }

      def setOptimizer(optimizer: Optimizer[T]): CompileOptions.Builder[T] = {
        options._optimizer = optimizer
        this
      }

      def setOptimizer(optimizerType: Optimizers): CompileOptions.Builder[T] =
        setOptimizer(Optimizers.select[T](optimizerType))

      def addMetric(metric: Metrics): CompileOptions.Builder[T] = addMetric(Metrics.select(metric))

      def addMetric(metric: Metric): CompileOptions.Builder[T] = {
        if (options.metrics == null) options._metrics = new ju.ArrayList[Metric]
        options.metrics.add(metric)
        this
      }

      def build: Model.CompileOptions[T] = options
    }
  }

  class CompileOptions[T <: TNumber] {
    private var _metrics   : ju.List[Metric] = null
    private var _optimizer : Optimizer[T]    = null
    private var _loss      : Loss            = null

    def metrics: ju.List[Metric] = _metrics

    def optimizer: Optimizer[T] = _optimizer

    def loss: Loss = _loss
  }

  object FitOptions {
    def defaults: Model.FitOptions = new FitOptions.Builder().setEpochs(1).setBatchSize(32).build

    def builder = new FitOptions.Builder(defaults)

    class Builder() {
      final private var options = new Model.FitOptions

      def this(options: Model.FitOptions) = {
        this()
        this.options = options
      }

      def setEpochs(epochs: Int): FitOptions.Builder = {
        options._epochs = epochs
        this
      }

      def setBatchSize(batchSize: Int): FitOptions.Builder = {
        options._batchSize = batchSize
        this
      }

      def build: Model.FitOptions = options
    }
  }

  class FitOptions {
    private var _epochs    = 0
    private var _batchSize = 0

    def epochs    : Int = _epochs
    def batchSize : Int = _batchSize
  }
}

abstract class Model[T <: TNumber](dtype0: Class[T]) extends Layer[T](1) { // TODO:  For now, models take in only 1 input
  this.dtype = dtype0
  this.built = true

  @throws[Exception]
  def compile(tf: Ops, optimizer: Optimizer[T], loss: Loss, metric: ju.List[Metric]): Unit

  @throws[Exception]
  def compile(tf: Ops, compilerBuilder: Model.CompileOptions[T]): Unit =
    compile(tf, compilerBuilder.optimizer, compilerBuilder.loss, compilerBuilder.metrics)

  def fit(tf: Ops, train: GraphLoader[T], test: GraphLoader[T], epochs: Int, batchSize: Int): Unit

  def fit(tf: Ops, train: GraphLoader[T], test: GraphLoader[T], fitOptions: Model.FitOptions): Unit =
    fit(tf, train, test, fitOptions.epochs, fitOptions.batchSize)
}
