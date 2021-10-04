package org.tensorflow.keras.layers

import org.tensorflow.{Operand, Tensor}
import org.tensorflow.keras.layers.BatchNormalization.RenormClipping
import org.tensorflow.keras.utils.Backend
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.types.family.TNumber

import scala.util.Try

object BatchNormalization {
  object RenormClipping {
    case object Rmax extends RenormClipping
    case object Rmin extends RenormClipping
    case object Dmax extends RenormClipping
  }
  sealed trait RenormClipping
}
/** Layer that normalizes its inputs.
  * Batch normalization applies a transformation that maintains the mean output
  * close to 0 and the output standard deviation close to 1.
  * Importantly, batch normalization works differently during training and
  * during inference.
  * **During training** (i.e. when using `fit()` or when calling the layer/model
  * with the argument `training=True`), the layer normalizes its output using
  * the mean and standard deviation of the current batch of inputs. That is to
  * say, for each channel being normalized, the layer returns
  * `gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta`, where:
  * - `epsilon` is small constant (configurable as part of the constructor
  * arguments)
  * - `gamma` is a learned scaling factor (initialized as 1), which
  * can be disabled by passing `scale=False` to the constructor.
  * - `beta` is a learned offset factor (initialized as 0), which
  * can be disabled by passing `center=False` to the constructor.
  * **During inference** (i.e. when using `evaluate()` or `predict()`) or when
  * calling the layer/model with the argument `training=False` (which is the
  * default), the layer normalizes its output using a moving average of the
  * mean and standard deviation of the batches it has seen during training. That
  * is to say, it returns
  * `gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon) + beta`.
  * `self.moving_mean` and `self.moving_var` are non-trainable variables that
  * are updated each time the layer in called in training mode, as such:
  * - `moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)`
  * - `moving_var = moving_var * momentum + var(batch) * (1 - momentum)`
  * As such, the layer will only normalize its inputs during inference
  * *after having been trained on data that has similar statistics as the
  * inference data*.
  */
class BatchNormalization[T <: TNumber](
                                        axis0: Seq[Int] = Seq(-1),
                                        momentum: Float = 0.99f,
                                        epsilon: Float = 1e-3f,
                                        center: Boolean = true,
                                        scale: Boolean = true,
                                        //  betaInitializer='zeros',
                                        //  gammaInitializer='ones',
                                        //  movingMeanInitializer='zeros',
                                        //  movingVarianceInitializer='ones',
                                        //  betaRegularizer=None,
                                        //  gammaRegularizer=None,
                                        //  betaConstraint=None,
                                        //  gammaConstraint=None,
                                        renorm: Boolean = false,
                                        renorm_clipping: Map[RenormClipping, Tensor] = Map.empty,
                                        renorm_momentum: Float = 0.99f,
                                        fused0    : Option[Boolean] = None,
                                        trainable : Boolean = true,
                                        virtual_batch_size: Option[Int] = None,
                                        adjustment : Option[Nothing] = None,
                                        //  name=None,
                                      ) extends Layer[T](1) with ScalaLayer[T] {

  private var moving_mean     = Option.empty[Variable[T]]
  private var moving_variance = Option.empty[Variable[T]]

  private var fused = if (fused0.contains(true)) {
    raiseIfFusedCannotBeUsed()
    fused0
  } else if (fused0.isEmpty && !fusedCanBeUsed()) {
    // NOT: We leave fused as None if self._fused_can_be_used()==True, since we
    // still may set it to False in self.build() if the input rank is not 4.
    Some(false)
  } else {
    fused0
  }

  private var axis: Seq[Int] = axis0

  // XXX TODO
  // this.supports_masking = True

  protected val besselsCorrectionTestOnly = true

  // if (renorm) {
  // }

  private def raiseIfFusedCannotBeUsed(): Unit = ???

  private def fusedCanBeUsed(): Boolean = Try(raiseIfFusedCannotBeUsed()).isSuccess

  private var _data_format: String = null

  override protected def build(tf: Ops, inputShape: Shape): Unit = {
    // val inputShape = tf.TensorShape(inputShape)
    if (!inputShape.isUnknown)
      throw new IllegalArgumentException(
        s"Input has undefined rank. Received =  input_shape=$inputShape.")

    val ndims = inputShape.numDimensions()

    // Convert axis to list and resolve negatives
    axis = axis.map {
      case x if x < 0 => ndims + x
      case x => x
    }

    // Validate axes
    if (axis.exists { x => x < 0 || x >= ndims }) {
      throw new IllegalArgumentException(
        "Invalid axis. Expected 0 <= axis < inputs.rank (with " ++
          s"inputs.rank=$ndims). Received =  layer.axis=$axis"
      )
    }
    if (axis != axis.distinct)
      throw new IllegalArgumentException(s"Duplicate axis = $axis")

    if (virtual_batch_size.isDefined) {
      if (virtual_batch_size.exists(_ <= 0))
        throw new IllegalArgumentException(
          "virtual_batch_size must be a positive integer that divides the " ++
            "true batch size of the input tensor. Received =  " ++
            s"virtual_batch_size=$virtual_batch_size"
        )
      // If using virtual batches, the first dimension must be the batch
      // dimension and cannot be the batch norm axis
      if (axis.contains(0))
        throw new IllegalArgumentException(
          "When using virtual_batch_size, the batch dimension " ++
            "must be 0 and thus axis cannot include 0. " ++
            s"Received axis=$axis"
        )

      if (adjustment.isDefined)
        throw new IllegalArgumentException(
          "When using virtual_batch_size, adjustment cannot " ++
            "be specified"
        )
    }

    if (fused.isEmpty || fused.contains(true)) {
      // TODO(yaozhang):  if (input is not 4D, reshape it to 4D and reshape the
      // output back to its original shape accordingly.
      // if (this._USE_V2_BEHAVIOR) {
      if (fused.isEmpty)
        fused = Some(ndims == 4 || ndims == 5)
      else if (fused.contains(true) && (ndims != 4 && ndims != 5))
        throw new IllegalArgumentException(
          "Batch normalization layers with `fused=True` only " ++
            "support 4D or 5D input tensors. " ++
            s"Received tensor with shape =  $inputShape"
        )
      // } else {
      //   assert(fused != None)
      //   this.fused = (ndims in(4, 5) and this._fused_can_be_used())
      // }

      // TODO(chrisying):  fused batch norm is currently not supported for
      // multi-axis batch norm and by extension virtual batches. In some cases,
      // it might be possible to use fused batch norm but would require reshaping
      // the Tensor to 4D with the axis in 1 or 3 (preferred 1) which is
      // particularly tricky. A compromise might be to just support the most
      // common use case (turning 5D w/ virtual batch to NCHW)
    }

    if (fused.contains(true)) {
      if (axis == Seq(1) && ndims == 4)
        this._data_format = "NCHW"
      else if (this.axis == Seq(1) && ndims == 5)
        this._data_format = "NCDHW"
      else if (this.axis == Seq(3) && ndims == 4)
        this._data_format = "NHWC"
      else if (this.axis == Seq(4) && ndims == 5)
        this._data_format = "NDHWC"
      else if (ndims == 5) {
        // 5D tensors that can be passed in but should not use fused batch norm
        // due to unsupported axis.
        fused = Some(false)
      } else if (ndims == 4) {
        throw new IllegalArgumentException(
          "Unsupported axis. The use of `fused=True` is only possible with " ++
            "`axis=1` or `axis=3` for 4D input tensors. Received " ++
            s"axis=$axis"
        )
      } else throw new IllegalArgumentException(
        "Unsupported axis. The use of `fused=True` is only possible with " ++
          "`axis=1` or `axis=4` for 5D input tensors. Received " ++
          s"axis=$axis"
      )
    }

    // val axis_to_dim = {x =  inputShape.dims[x].value for x in this.axis}
    val axis_to_dim = axis.map { x => inputShape.size(x) }
    if (axis_to_dim.contains(Shape.UNKNOWN_SIZE))
      throw new IllegalArgumentException(
        "Input has undefined `axis` dimension. Received input " ++
        s"with shape $inputShape. Axis value =  $axis"
      )

    // XXX TODO
    // this.input_spec = (new InputSpec).ndim(ndims).axes(axis_to_dim)

    var param_shape: Seq[Long] = null

    if (axis_to_dim.size == 1 && virtual_batch_size.isEmpty) {
      // Single axis batch norm (most common/default use-case)
      param_shape = axis_to_dim.head :: Nil // (list(axis_to_dim.values())[0],)
    } else {
      // Parameter shape is the original shape but with 1 in all non-axis dims
      param_shape = Seq.tabulate(ndims) { i =>
        if (axis_to_dim.contains(i)) axis_to_dim(i) else 1L
      }
      if (virtual_batch_size.isDefined) {
        // When using virtual batches, add an extra dim at index 1
        param_shape = param_shape.patch(1, 1L :: Nil, 0)
        axis = axis.map(_ + 1) // Account for added dimension
      }
    }
    if (scale) {
      /*this.gamma =*/ addWeightExt(
        /*name =*/ "gamma",
        /*shape =*/ param_shape,
        dtype = this._param_dtype,
        initializer = this.gammaInitializer,
        regularizer = this.gammaRegularizer,
        constraint  = this.gammaConstraint,
        trainable   = true,
        experimental_autocast = False
      )
    } else {
      // this.gamma = None
      if (fused.contains(true))
        /*this._gamma_const =*/ Backend.constant(
          1.0, dtype = this._param_dtype, shape = param_shape)
    }

    if (this.center) {
      /*this.beta =*/ addWeight(
        /*name =*/ "beta",
        /*shape =*/ param_shape,
        dtype = this._param_dtype,
        initializer = this.betaInitializer,
        regularizer = this.betaRegularizer,
        constraint = this.betaConstraint,
        trainable = true,
        experimental_autocast = false
      )
    } else {
      // this.beta = None
      if (this.fused)
        /*this._beta_const =*/ Backend.constant(
          0.0, dtype = this._param_dtype, shape = param_shape)
    }

    var partitioner: Option[Any] = None

    try {
      // Disable variable partitioning when creating the moving mean and variance
      if (hasattr(this, "_scope") && this._scope) {
        partitioner = this._scope.partitioner
        this._scope.set_partitioner(None)
      } else {
        partitioner = None
      }
      this.moving_mean = Some(addWeight(
        /*name =*/ "moving_mean",
        /*shape =*/ param_shape,
        dtype = this._param_dtype,
        initializer = this.moving_mean_initializer,
        synchronization = tf.VariableSynchronization.ON_READ,
        trainable = false,
        aggregation = tf.VariableAggregation.MEAN,
        experimental_autocast = false
      ))

      this.moving_variance = Some(addWeight(
        /*name =*/ "moving_variance",
        /*shape =*/ param_shape,
        dtype = this._param_dtype,
        initializer = this.moving_variance_initializer,
        synchronization = tf.VariableSynchronization.ON_READ,
        trainable = false,
        aggregation = tf.VariableAggregation.MEAN,
        experimental_autocast = false
      ))

      if (this.renorm) {
        ???
//        // In batch renormalization we track the inference moving stddev instead
//        // of the moving variance to more closely align with the paper.
//        def moving_stddev_initializer(*args, ** kwargs) =
//          tf.math.sqrt(
//            this.moving_variance_initializer(* args, ** kwargs))
//
//        with tf.distribute.get_strategy(
//        ).extended.colocate_vars_with(this.moving_variance) =
//          this.moving_stddev = this.add_weight(
//            name = "moving_stddev",
//            shape = param_shape,
//            dtype = this._param_dtype,
//            initializer = moving_stddev_initializer,
//            synchronization = tf.VariableSynchronization.ON_READ,
//            trainable = False,
//            aggregation = tf.VariableAggregation.MEAN,
//            experimental_autocast = False)
//
//        // Create variables to maintain the moving mean and standard deviation.
//        // These are used in training and thus are different from the moving
//        // averages above. The renorm variables are colocated with moving_mean
//        // and moving_stddev.
//        // NOTE =  below, the outer `with device` block causes the current device
//        // stack to be cleared. The nested ones use a `lambda` to set the desired
//        // device and ignore any devices that may be set by the custom getter.
//        /** Create a renorm variable. */
//        def _renorm_variable(name,
//                             shape,
//                             initializer = tf.compat.v1.zeros_initializer()) =
//
//        val variable = this.add_weight(
//          name = name,
//          shape = shape,
//          dtype = this._param_dtype,
//          initializer = initializer,
//          synchronization = tf.VariableSynchronization.ON_READ,
//          trainable = False,
//          aggregation = tf.VariableAggregation.MEAN,
//          experimental_autocast = False)
//        return variable
//
//        with tf.distribute.get_strategy(
//        ).extended.colocate_vars_with(this.moving_mean) =
//          this.renorm_mean = _renorm_variable("renorm_mean", param_shape,
//            this.moving_mean_initializer)
//        with tf.distribute.get_strategy(
//        ).extended.colocate_vars_with(this.moving_stddev) =
//          this.renorm_stddev = _renorm_variable("renorm_stddev", param_shape,
//            moving_stddev_initializer)
      }
    } finally {
      if (partitioner.isDefined)
        this._scope.set_partitioner(partitioner)
    }
    built = true
  }

  override def computeOutputShape(inputShape: Shape): Shape = inputShape

  override protected def call(tf: Ops, inputs: Operand[T]*): Operand[T] = ???
}
