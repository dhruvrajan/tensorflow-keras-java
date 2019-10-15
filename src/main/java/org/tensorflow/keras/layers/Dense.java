package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.keras.activations.Activation;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.utils.TensorShape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;

public class Dense<T extends Number> extends Layer<T> {
    private static int DENSE_INPUT_LENGTH = 1;
    private int units;

    private static String KERNEL = "kernel";
    private static String KERNEL_INIT = "kernelInit";
    private static String BIAS = "bias";
    private static String BIAS_INIT = "biasInit";

    // weight tensors
    private Variable<T> kernel;
    private Variable<T> bias;

    // initializers
    private Initializer<T> kernelInitializer;
    private Initializer<T> biasInitializer;

    // activation function
    private Activation<T> activation;

    public Dense(int units, Dense.Options options) {
        super(DENSE_INPUT_LENGTH);
        this.units = units;

        this.activation = options.activation;
        this.kernelInitializer = options.kernelInitializer;
        this.biasInitializer = options.biasInitializer;
    }

    private Dense(
            int units,
            Activation<T> activation,
            Initializer<T> kernelInitializer,
            Initializer<T> biasInitializer) {
        super(DENSE_INPUT_LENGTH);
        this.units = units;

        this.activation = activation;
        this.kernelInitializer = kernelInitializer;
        this.biasInitializer = biasInitializer;
    }
    public static Options options() {
        return new Dense.Options();
    }

    @Override
    public void build(Ops tf, Shape inputShape, Class<T> dtype) {
        TensorShape tensorShape = new TensorShape(inputShape);
        tensorShape.assertKnown(tensorShape.numDimensions() - 1);

        Shape kernelShape = Shape.make(inputShape.size(inputShape.numDimensions() - 1), this.units);

        Shape biasShape = Shape.make(this.units);

        // Create dense kernel tensor
        this.kernel = this.addWeight(KERNEL, tf.variable(kernelShape, dtype));
        this.addInitializer(KERNEL_INIT, this.kernelInitializer);
        this.kernelInitializer.build(tf, this.kernel, dtype);

        // Create bias tensor
        this.bias = this.addWeight(BIAS, tf.variable(biasShape, dtype));
        addInitializer(BIAS_INIT, this.biasInitializer);
        this.biasInitializer.build(tf, this.bias, dtype);

        this.activation.build(tf, computeOutputShape(inputShape), dtype);
    }

    public Shape computeOutputShape(Shape inputShape) {
        // leaves unknown dimensions unknown
        return new TensorShape(inputShape).replaceLast(this.units).toShape();
    }

    @SafeVarargs
    public final Operand<T> call(Ops tf, Operand<T>... inputs) {
        return this.call(tf, inputs[0]);
    }

    private Operand<T> call(Ops tf, Operand<T> input) {
        Operand<T> signal = tf.add(tf.matMul(input, this.kernel), this.bias);
        return this.activation.apply(tf, signal);
    }

    public static class Options<T extends Number> {
        public static Activations DEFAULT_ACTIVATION = Activations.linear;
        public static Initializers DEFAULT_KERNEL_INITIALIZER = Initializers.randomNormal;
        public static Initializers DEFAULT_BIAS_INITIALIZER = Initializers.zeros;
        public Class DEFAULT_DTYPE = Float.class;

        // Default parameters
        private Activation<T> activation = Activations.select(DEFAULT_ACTIVATION);
        private Initializer<T> kernelInitializer = Initializers.select(DEFAULT_KERNEL_INITIALIZER);
        private Initializer<T> biasInitializer = Initializers.select(DEFAULT_BIAS_INITIALIZER);

        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private Options options;

            public Builder() {
                this.options = new Options();
            }

            public Builder setActivation(Activations activation) {
                return setActivation(Activations.select(activation));
            }

            public Builder setActivation(Activation<Float> activation) {
                this.options.activation = activation;
                return this;
            }

            public Builder setKernelInitializer(Initializers kernelInitializer) {
                return setKernelInitializer(Initializers.select(kernelInitializer));
            }

            public Builder setKernelInitializer(Initializer kernelInitializer) {
                this.options.kernelInitializer = kernelInitializer;
                return this;
            }

            public Builder setBiasInitializer(Initializers biasInitializer) {
                return setBiasInitializer(Initializers.select(biasInitializer));
            }

            public Builder setBiasInitializer(Initializer biasInitializer) {
                this.options.biasInitializer = biasInitializer;
                return this;
            }

            public Options build() {
                return this.options;
            }
        }
    }

}
