package org.tensorflow.keras.layers

import org.tensorflow.Operand
import org.tensorflow.framework.initializers.Initializer
import org.tensorflow.keras.engine.BaseLayerUtils
import org.tensorflow.keras.initializers.Initializers
import org.tensorflow.keras.initializers.Initializers.Ops
import org.tensorflow.keras.mixin.LayerFunction
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.{Ops => TF}
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Variable
import org.tensorflow.proto.framework.{VariableAggregation, VariableSynchronization}
import org.tensorflow.types.{TBool, TFloat32, TInt32, TUint8}
import org.tensorflow.types.family.TNumber

import java.{util => ju}

/**
  * Base layer class.
  *
  * <p>A layer implements common neural network operations, such as convolution, batch norm, etc.
  * These operations require managing weights, losses, updates, and inter-layer connectivity.
  *
  * @param < T> Numeric type of the output (Float, Double)
  */
abstract class Layer[T <: TNumber](val INPUTS_LENGTH: Int) extends LayerFunction[T] {
  protected var built = false
  final private val _trainableWeights     = new ju.HashMap[String, Variable[T]]
  final private val _nonTrainableWeights  = new ju.HashMap[String, Variable[T]]
  final private val initializerOpMap      = new ju.HashMap[String, Assign[T]]
  // Input() layer needs to access dtype and built.
  protected var dtype: Class[T] = null

  /**
    * Overrides create(Ops) to add variables (weight tensors) to the layer.
    *
    * The addWeight function and some tf ops require passing a Class<T> "dtype" object
    *
    * To get the dtype of this layer in the build function, use Layer.getDtype()
    *
    * @param tf         Tensorflow Ops accessor
    * @param inputShape Shape of the layer's input tensor
    *
    */
  protected def build(tf: TF, inputShape: Shape): Unit

  final def build(tf: TF, inputShape: Shape, dtype: Class[T]): Unit = {
    this.dtype = dtype
    build(tf, inputShape)
    this.built = true
  }

  /**
    * Computes the output shape of the tensor returned by a Layer from the input tensor's shape
    *
    * @param inputShape Shape of an input tensor to this layer
    * @return Shape of the tensor that would be returned by `apply`.
    */
  def computeOutputShape(inputShape: Shape): Shape

  /**
    * Defines the layer's logic, in terms of input operands, and variables.
    *
    * @param tf     Tensorflow Ops accessor.
    * @param inputs A sequence of TF Operands
    * @return The transformed input tensors, according to the layer's logic.
    */
  protected def call(tf: TF, inputs: Seq[Operand[T]], training: Option[Boolean] = None): Operand[T]

  /**
    * Internal wrapper for Layer.call
    */
  override final def apply(tf: TF, inputs: Operand[T]*): Operand[T] = {
    if (!this.built) throw new IllegalStateException("Layer.call() cannot be called before the layer is built (Layer.build())")
    if (inputs.length != INPUTS_LENGTH) throw new IllegalArgumentException("Layer call() expected " + INPUTS_LENGTH + "inputs; received " + inputs.length + ".")
    this.call(tf, inputs/*: _**/)
  }

  /** The dtype of the layer's computations.
    * This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless
    * mixed precision is used, this is the same as `Layer.dtype`, the dtype of
    * the weights.
    * Layers automatically cast their inputs to the compute dtype, which causes
    * computations and the output to be in the compute dtype as well. This is done
    * by the base Layer class in `Layer.__call__`, so you do not have to insert
    * these casts if implementing your own layer.
    * Layers often perform certain internal computations in higher precision when
    * `compute_dtype` is float16 or bfloat16 for numeric stability. The output
    * will still typically be float16 or bfloat16 in such cases.
    *
    * @return  The layer's compute dtype.
    */
  def computeDtype =
    ??? // _dtype_policy.compute_dtype

  /**
    * Adds a new weight tensor to the layer
    *
    * @param name     variable name
    * @param variable variable to add
    * @return the created variable.
    */
  final protected def addWeight(name: String, variable: Variable[T]): Variable[T] = {
    _trainableWeights.put(name, variable)
    variable
  }

//  final protected def addWeight(tf: Ops, name: String, variable: Variable[T], initializerName: String, initializer: Nothing): Variable[T] = addWeight(tf, name, variable, initializerName, Initializers.select(initializer))

  final protected def addWeight(tf: TF, name: String, variable: Variable[T], initializerName: String, initializer: Initializer[T]): Variable[T] = {
    _trainableWeights.put(name, variable)
    initializerOpMap.put(initializerName, initializer.apply(tf, variable, dtype))
    variable
  }


  /** Adds a new variable to the layer.
   *
   * kwargs: Additional keyword arguments. Accepted values are `getter`,
   *    `collections`, `experimental_autocast` and `caching_device`.
   *
   * @param name             Variable name.
   * @param shape            Variable shape. Defaults to scalar if unspecified.
   * @param dtype            The type of the variable. Defaults to `self.dtype`.
   * @param initializer      Initializer instance (function).
   * @param regularizer      Regularizer instance (function).
   * @param trainable        whether the variable should be part of the layer's
   *    "trainable_variables" (e.g. variables, biases)
   *    or "non_trainable_variables" (e.g. BatchNorm mean and variance).
   *    Note that `trainable` cannot be `True` if `synchronization`
   *    is set to `ON_READ`.
   * @param constraint       Constraint instance (function).
   * @param use_resource     Whether to use `ResourceVariable`.
   * @param synchronization  Indicates when a distributed a variable will be
   *    aggregated. Accepted values are constants defined in the class
   *    `tf.VariableSynchronization`. By default the synchronization is set to
   *    `AUTO` and the current `DistributionStrategy` chooses
   *    when to synchronize. If `synchronization` is set to `ON_READ`,
   *    `trainable` must not be set to `True`.
   * @param aggregation      Indicates how a distributed variable will be aggregated.
   *    Accepted values are constants defined in the class
   *    `tf.VariableAggregation`.
   * @return
   */
  protected final def addWeight(tf: TF,
                                   name            : String, //                 = None,
                                   shape           : Shape                   = Shape.scalar() /*unknown()*/,
                                   dtype           : Class[_ <: TNumber]     = this.dtype,
                                   initializerName : String,
                                   initializer     : Initializer[T],
                                   regularizer     : Option[Nothing]         = None,
                                   trainable       : Option[Boolean]         = None,
                                   constraint      : Option[Nothing]         = None,
                                   use_resource    : Boolean                 = false,
                                   synchronization : VariableSynchronization = VariableSynchronization.VARIABLE_SYNCHRONIZATION_AUTO,
                                   aggregation     : VariableAggregation     = VariableAggregation.VARIABLE_AGGREGATION_NONE,
                                  ): Variable[T] = {
    // dtype = tf.as_dtype(dtype)
    // if self._dtype_policy.variable_dtype is None:
    //   // The policy is "_infer", so we infer the policy from the variable dtype.
    //   self._set_dtype_policy(policy.Policy(dtype.base_dtype.name))
    // initializer = initializers.get(initializer)
    // regularizer = regularizers.get(regularizer)
    // constraint  = constraints .get(constraint)

    val _trainable = if (synchronization == VariableSynchronization.VARIABLE_SYNCHRONIZATION_ON_READ) {
      if (trainable.contains(true)) {
        throw new IllegalArgumentException(
          "Synchronization value can be set to VariableSynchronization.ON_READ only for non - trainable variables. " ++
            "You have specified trainable = True and synchronization = VariableSynchronization.ON_READ."
        )
      } else {
        // Set trainable to be false when variable is to be synced on read.
        false
      }
    } else trainable.getOrElse(true)

//    // Initialize variable when no initializer provided
//    val _initializer = initializer.getOrElse {
//      // If dtype is DT_FLOAT, provide a uniform unit scaling initializer
//      if (dtype == classOf[TFloat32] /*.is_floating*/) {
//        Initializers.select[T](Initializers.glorotUniform)
//      } else if (dtype == classOf[TInt32] /*.is_integer*/ /*|| dtype.is_unsigned*/ || dtype == classOf[TBool] /*.is_bool*/) {
//        // If dtype is DT_INT/DT_UINT, provide a default value `zero`
//        // If dtype is DT_BOOL, provide a default value `FALSE`
//        Initializers.select[T](Initializers.zeros)
//      } // NOTES:Do we need to support for handling DT_STRING and DT_COMPLEX here?
//      else /* "getter" not in kwargs */ {
//        // When `getter` is specified, it"s possibly fine for `initializer` to be
//        // None since it"s up to the custom `getter` to raise error in case it
//        // indeed needs `initializer`.
//        throw new IllegalArgumentException(
//          s"An initializer for variable $name of type ${dtype/*.base_dtype*/} is required for layer $this. Received: $initializer."
//        )
//      }
//    }
    val _initializer = initializer

//    val getter = /*kwargs.pop('getter',*/ BaseLayerUtils.makeVariable _ /*)*/

    // autocast bla bla, bloddy Python

    val dtypeC = dtype.asInstanceOf[Class[T]] // XXX TODO
    val variable: Variable[T] =
      tf.variable(shape, dtypeC) // , Variable.sharedName(name)
    initializerOpMap.put(initializerName, _initializer(tf, variable, dtypeC))

    //      _add_variable_with_custom_getter(
//      name = name,
//      shape = shape,
//      // TODO(allenl): a `make_variable` equivalent should be added as a
//      // `Trackable` method.
//      getter = getter,
//      // Manage errors in Layer rather than Trackable.
//      overwrite = true,
//      initializer = initializer,
//      dtype = dtype,
//      constraint = constraint,
//      trainable = _trainable,
//      use_resource = use_resource,
//      collections = collections_arg,
//      synchronization = synchronization,
//      aggregation = aggregation,
//      caching_device = caching_device
//    )

    if (regularizer.isDefined) {
      // TODO(fchollet): in the future, this should be handled at the
      // level of variable creation, and weight regularization losses
      // should be variable attributes.
      ???
//      val name_in_scope = variable.name // [: variable.name.find(':')]
//      _handle_weight_regularization(name_in_scope,
//        variable,
//        regularizer)
    }

    if (false /*BaseLayerUtils.is_split_variable(variable)*/) {
      ???
//      for (v <- variable) {
//        backend.track_variable(v)
//        if (_trainable) {
//          _trainableWeights.add(v)
//        } else {
//          ???
//  //        _non_trainable_weights.append(v)
//        }
//      }
    } else {
      // XXX TODO: is this automatically the case when using tf.variable?
//      backend.track_variable(variable)
      if (_trainable) {
        _trainableWeights.put(name, variable)
      } else {
        _nonTrainableWeights.put(name, variable)
      }
    }

    variable
  }

  def initializerOps      : ju.List[Operand [T]] = new ju.ArrayList[Operand [T]](initializerOpMap     .values)
  def trainableWeights    : ju.List[Variable[T]] = new ju.ArrayList[Variable[T]](_trainableWeights    .values)

  // XXX TODO python is significantly more complicated, using `_gather_children_attribute`
  def nonTrainableWeights : ju.List[Variable[T]] = new ju.ArrayList[Variable[T]](_nonTrainableWeights .values)

  def weights             : ju.List[Variable[T]] = {
    val res = new ju.ArrayList[Variable[T]]
    res.addAll(_nonTrainableWeights.values)
    res.addAll(_trainableWeights   .values)
    res
  }

  def isBuilt : Boolean = this.built
  def hasDtype: Boolean = this.dtype != null

  def getDtype: Class[T] = this.dtype
}
