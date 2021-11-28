package org.tensorflow.keras.examples

import org.tensorflow.Graph
import org.tensorflow.keras.activations.Activations
import org.tensorflow.keras.datasets.FashionMNIST
import org.tensorflow.keras.initializers.Initializers
import org.tensorflow.keras.layers.{Dense, Layers}
import org.tensorflow.keras.losses.Losses
import org.tensorflow.keras.metrics.Metrics
import org.tensorflow.keras.models.Model.CompileOptions
import org.tensorflow.keras.models.{Model, Sequential}
import org.tensorflow.keras.optimizers.Optimizers
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32

object FashionMNISTKeras {
  def train(model: Model[TFloat32]): Model[TFloat32] = {
    val graph = new Graph
    try { // Create Tensorflow Ops Accessor
      val tf = Ops.create(graph)
      // Compile Model
      model.compile(tf, compileOptions)
      // Accessors for MNIST Data
      val loaders = FashionMNIST.graphLoaders2D
      // GraphLoader objects contain AutoCloseable `Tensor` objects.
      val train = loaders.first
      try {
        val test = loaders.second
        try // Fit model
          model.fit(tf, train, test, fitOptions)
        finally test.close()
      } finally train.close()
    } finally graph.close()
    model
  }

  def main(args: Array[String]): Unit =
    train(model)

  // Define Neural Network Model
  // Note: Layers can be constructed either from individual
  //       Option.Builder classes, or from the static helper
  //       methods defined in `Layers` which wrap the explicit builders
  //       to decrease verbosity.
  private val model =
  Sequential(classOf[TFloat32],
    Layers.input(28, 28),
    Layers.flatten(),
    Layers.dense(256, Some(Activations.relu),
      kernelInitializer = Initializers.randomNormal,
      biasInitializer   = Initializers.zeros
    ),
    Layers.dense[TFloat32](128, Some(Activations.relu),
      kernelInitializer = Initializers.randomNormal,
      biasInitializer   = Initializers.zeros
    ), // Using static helper Layers.dense(...)
    Layers.dense(10, Some(Activations.softmax),
      kernelInitializer = Initializers.randomNormal,
      biasInitializer   = Initializers.zeros
    )
  )

  // Model Compile Configuration
  private val compileOptions = new CompileOptions.Builder[TFloat32]()
    .setOptimizer(Optimizers.sgd)
    .setLoss(Losses.sparseCategoricalCrossentropy)
    .addMetric(Metrics.accuracy).build
  // Model Training Loop Configuration
  private val fitOptions = Model.FitOptions.builder.setEpochs(10).setBatchSize(100).build
}
