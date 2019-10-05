# Tensorflow-Keras (Java)

This repository contains a JVM implementation of the Keras API, built on [Tensorflow Java](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary).
Keras is a high-level API for building and training deep learning models. 
This implementation aims to mirror the Python [tf-keras](https://www.tensorflow.org/guide/keras) syntax in a clean Java-friendly style.  

It is a work-in-progress; feel free to add issues / comments! Happy to collaborate. 



Example
--
Below is a comparison of the Java and Python implementations of a model trained for the MNIST dataset.
 
 
Python:

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu', kernel_initializer="random_normal", bias_initializer="zeros"),
  tf.keras.layers.Dense(10, activation='softmax', kernel_initializer="random_normal", bias_initializer="zeros")
])

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

(X_train, y_train), (X_val, y_val) = tf.keras.datasets.load_mnist()
model.fit(X_train, y_train, val_data=(X_val, y_val), epochs=10, batch_size=100)
```
 
Java:
```java
import org.tensorflow.Graph;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.datasets.MNIST;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.layers.Dense;
import org.tensorflow.keras.layers.Layers;
import org.tensorflow.keras.losses.Losses;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.keras.models.Model;
import org.tensorflow.keras.models.Sequential;
import org.tensorflow.keras.optimizers.Optimizers;
import org.tensorflow.op.Ops;
import org.tensorflow.utils.Pair;

public class MNISTKeras {
    private static Model<Float> model;
    private static Model.CompileOptions compileOptions;
    private static Model.FitOptions fitOptions;

    static {
        // Define Neural Network Model

        // Note: Layers can be constructed either from individual
        //       Option.Builder classes, or from the static helper
        //       methods defined in `Layers` which wrap the explicit builders
        //       to decrease verbosity.
        model = new Sequential(
                Layers.input(28 * 28),

                // Construct using Dense.Options.Builder
                new Dense(128, Dense.Options.builder()
                        .setActivation(Activations.relu)
                        .setKernelInitializer(Initializers.randomNormal)
                        .setBiasInitializer(Initializers.zeros)
                        .build()),

                // Construct using static helper Layers.dense(...)
                Layers.dense(10, Activations.softmax, Initializers.randomNormal, Initializers.zeros)
        );

        // Model Compile Configuration
        compileOptions = Model.CompileOptions.builder()
                .setOptimizer(Optimizers.sgd)
                .setLoss(Losses.softmax_crossentropy)
                .addMetric(Metrics.accuracy)
                .build();

        // Model Training Loop Configuratoin
        fitOptions = Model.FitOptions.builder()
                .setEpochs(10)
                .setBatchSize(100)
                .build();
    }


    public static Model<Float> train() throws Exception {
        try (Graph graph = new Graph()) {
            // Create Tensorflow Ops Accessor
            Ops tf = Ops.create(graph);

            // Compile Model
            model.compile(tf, compileOptions);

            Pair<GraphLoader<Float>, GraphLoader<Float>> loaders = MNIST.graphLoaders();
            // GraphLoader objects contain AutoCloseable `Tensor` objects.
            try (GraphLoader<Float> train = loaders.first();
                 GraphLoader<Float> test = loaders.second()) {
                // Fit model
                model.fit(tf, train, test, fitOptions);
            }
        }

        return model;
    }

    public static void main(String[] args) throws Exception {
        train();
    }
}
```

Overview
==
Keras is built around a few core abstractions which comprise much of what is needed to build deep learning models. 
These include Layers, Models, Optimizers, Activations, Losses, Metrics, Regularizers, and Initializers.

Basics
--
To use TensorFlow, we must have access to a `Graph` to build and run computations. In Python, this is constructed
implicitly. In TF Java, the `Graph` and `Ops` objects must be created explicitly. Thus, we
leave this (below) for the user to write, and allow the `Ops` object to be passed
throughout keras `Layer` construction calls to provide access to core tensorflow operations.

```java
try(Graph graph = new Graph) {
    Ops tf = Ops.create(graph);
    
    // Keras code here.
}
```

Layer API
--
Layers represent arbitrary graph computations on tensors. They are composable; meant to be "stacked" on each other to
build neural networks. Thus they must encode *differentiable* functions (i.e. built using TF operations), and must contain *trainable weight tensors*
that can be optimized using backpropagation.


1. **Using Layers.** Layers have two main methods: `Layer.build(Ops tf)`, and `Layer.call(Operand op1, Operand op2, ...)`. `Layer.build` is used
    to create the necessary graph components (variables), while `Layer.call` defines the layer's logic in terms of these
    variables, and other tensorflow operators.
    
    TODO([#1]): Currently, the build vs. call functionality is mixed, and handled entirely by the `build` function.        
    
2. **Custom Layers.** New custom layers can be built by subclassing any `Layer` class. Use `Layer.addWeight(String name, Variable var)`
    to add a weight tensor to the layer and `Layer.addInitializer(String name, Assign initOp)` to add its initializer, during
    build (See `keras.layers.Dense` for an example).
    
    TODO([#2]): More robust Initializer support.
    
    TODO([#3]): Add support for Regularizers.   
    
3. **Multiple Input/Output.** Keras Layers can express functions between labeled sets of tensors. Multiple inputs
   are handled by adding arguments to `Layer.call`.
   
   TODO([#4]): Add support for multiple outputs; likely based on the the keras Network Layer.
   
4. **Arbitrarily-Shaped Input.** So long as the layer's comptuations are implemented properly, keras layers
    can operate on arbitrarily shaped input (within the restrictions of the computation). It's really useful not
    to have to worry about things like tensor dimensions and batch size etc.; this is handled for the user at graph
    building time. 
    
    TODO([#5]): To make this more explicit and debuggable, layer definitions should include a `Layer.computeOutputShape(Shape inputShape)`
    method definition.
    
5. **Layer Types** Keras provides a large library of pre-defined Layers, including convolutional, dropout, recurrent layers.
    
    TODO([#6]): Add support for variety of Layer types.
        
    Note: Not all layers, currently, can just be wrappers around TF operations. For example,
    tf.dropout doesn't seem to be implemented as a core tf operation in Java.
    

 
Model API
--
In Keras, Models represent built neural networks, and provide easy abstractions for model training/evaluation,
using the`Model.compile`, `Model.fit`, and `Model.evaluate`. Additionally, the Sequential and Functional model
construction APIs provide concise ways to specify models.

1. **Sequential API** Sequential models are built by stacking `Layer` objects in an array/list.
    
        Model model = new Sequential(
            new Dense(100, Shape.make(10, 500))
                .setActivation("sigmoid"),
            new Dense(50, Shape.make(10, 100))
                .setActivation("relu"),
            new Dense(10, Shape.make(10, 50)),
                .setActivation("softmax")
        );

    TODO([#7]): During `Model.compile`/`Layer.build`, the input shape of each layer should be automatically
    obtained from the previous layer's output shape (with the exception of the first layer, whose input
    shape must be specified by the user.)
    
2. **Model Training** Model training is implemented by the `Model.fit` method. Currently, a `Dataset` object
    must be passed to this function. Currently, this means the whole dataset must be represented and iterated
    over in memory.
    
    TODO([#8]): Allow Stream-based iteration over batches during training.

3. **Model Config** Keras objects (Optimizer, Layer, etc...) are fairly simple, and are nested in a clean way.
    Thus JSON would likely be a good method for representing serializable configurations of Models.
    
    TODO([#9]): Add support for model configuration (Perhaps using RapidJson or something similar)?
    
4. **Trained Model Serialization** Serializing a trained Model for later inference is very useful. This would likely
    entail both serializing the model config, as well as the trained weight tensors. Still need to decide on 
    a good way to do this.
    
    TODO([#10]): Add support for Model Serialization.

Datasets
--
Keras in Python supports input data in a number of forms (lists, numpy arrays, etc.). Efficient data representation /
loading / batch iteration makes quite an impact on training performance, esp. on large datasets. So far, we've defined
`keras.data.TensorDataset`, which takes `Tensor` objects as input, and constructs batches as operands in the tensorflow graph
using `tf.slice()`. There's also the `keras.data.Dataset` class which is based on `float[][]`, which delays the construction
of `Tensor` objects until batch training time, and constructs 1 tensor per batch. It will probably be best to support multiple
variants on this, and even integrate with a Java port of tf.data.

Keras "Calling Convention"
--
Keras, as implemented in Python, provides a large number of pythonic language features to speed up development.
Here we'll describe the differences/similarities in the Java API.  

1. **Variable and Keyword Arguments** Some Keras functions support a large number of keyword arguments with specified
    defaults. Java can't easily provide identical behavior, so we aim to come close by using the Builder Pattern. When
    keyword arguments are used in a method call, we define a simple `Options`/`Builder` class to manage these (see:
    `Model.CompilerBuilder`, `Model.FitBuilder`).

2. **Predefined Hyperparameters** Optimizers, Activations, and Layers often have well-behaving default values that can be used 
    automatically. These objects can be constructed using the default constructor, and their attributes will be populated
    using these defaults.

3. **String Aliases** Keras allows default objects to be addressed by a string name (e.g. `optimizer="adam"`, `activation="sigmoid"`).
    This is supported here in most cases; the names are matched against enums for each object type.
    
Keras Backend
  --
  The Keras Backend interface in python originally allowed for switching between various tensor library backends.
  Since this project is aimed at supporting tensorflow, it seems cleaner for now to depend on the core tensorflow
  implementation of core operations.

Debuggability
--    
The ability to set breakpoints etc. within `Layer` class definitions would be very useful for debugging.
 
 TODO([#11]): Add support for training in tensorflow eager mode.
 
 Callbacks
 --
 TODO([#12]): Add support for Keras callbacks. This would allow things like model checkpointing, early stopping,
  Tensorboard integration, and other useful features.
  
  Scala/Kotlin
  --
  One of the nice things about using the JVM is that we can have clean interface to other JVM languages like Scala and Kotlin, without having to re-implement everything.
  
  TODO: Would be nice to have nice Scala/Kotlin-esque interfaces for those languages in addition to the Java API.
  
[#1]: https://github.com/dhruvrajan/tensorflow-keras-java/issues/1
[#2]: https://github.com/dhruvrajan/tensorflow-keras-java/issues/2
[#3]: https://github.com/dhruvrajan/tensorflow-keras-java/issues/3
[#4]: https://github.com/dhruvrajan/tensorflow-keras-java/issues/4
[#5]: https://github.com/dhruvrajan/tensorflow-keras-java/issues/5
[#6]: https://github.com/dhruvrajan/tensorflow-keras-java/issues/6
[#7]: https://github.com/dhruvrajan/tensorflow-keras-java/issues/7
[#8]: https://github.com/dhruvrajan/tensorflow-keras-java/issues/8
[#9]: https://github.com/dhruvrajan/tensorflow-keras-java/issues/9
[#10]: https://github.com/dhruvrajan/tensorflow-keras-java/issues/10
[#11]: https://github.com/dhruvrajan/tensorflow-keras-java/issues/11
[#12]: https://github.com/dhruvrajan/tensorflow-keras-java/issues/12
