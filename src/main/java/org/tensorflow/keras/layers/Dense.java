package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.keras.activations.Activation;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.utils.TensorShape;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.family.TNumber;

public class Dense<T extends TNumber> extends Layer<T> {
    private static final int DENSE_INPUT_LENGTH = 1;
    private final int units;

    private static final String KERNEL      = "kernel";
    private static final String KERNEL_INIT = "kernelInit";
    private static final String BIAS        = "bias";
    private static final String BIAS_INIT   = "biasInit";

    // weight tensors
    private Variable<T> kernel;
    private Variable<T> bias;

    // initializers
    private final Initializer kernelInitializer;
    private final Initializer biasInitializer;

    // activation function
    private final Activation<T> activation;

    public Dense(int units, Dense.Options options) {
        super(DENSE_INPUT_LENGTH);
        this.units = units;

        this.activation = (Activation<T>) options.activation;
        this.kernelInitializer = options.kernelInitializer;
        this.biasInitializer = options.biasInitializer;
    }

    @Override
    public void build(Ops tf, Shape inputShape) {
        // Check that final dimension is known
        new TensorShape(inputShape).assertKnown(inputShape.numDimensions() - 1);

        // Retrieve Layer's dtype
        Class<T> dtype = this.getDtype();

        // Compute shapes of kernel and bias matrices
        Shape kernelShape = Shape.of(inputShape.size(inputShape.numDimensions() - 1), this.units);
        Shape biasShape   = Shape.of(this.units);

        // Create dense kernel tensor
        this.kernel = this.addWeight(tf, KERNEL, tf.variable(kernelShape, dtype), KERNEL_INIT, this.kernelInitializer);

        // Create bias tensor
        this.bias = this.addWeight(tf, BIAS, tf.variable(biasShape, dtype), BIAS_INIT, this.biasInitializer);

        // Create Activation
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
        Operand<T> signal = tf.math.add(tf.linalg.matMul(input, this.kernel), this.bias);
        return this.activation.apply(tf, signal);
    }


    public static class Options {
        // Default parameters
        private Activation activation;
        private Initializer kernelInitializer;
        private Initializer biasInitializer;

        public static Options defaults() {
            return new Builder(new Options())
                    .setActivation(Activations.linear)
                    .setKernelInitializer(Initializers.randomNormal)
                    .setBiasInitializer(Initializers.zeros)
                    .build();
        }

        public <T extends TNumber> Activation<T> getActivation() {
            return (Activation<T>) activation;
        }

        public Initializer getKernelInitializer() {
            return kernelInitializer;
        }

        public Initializer getBiasInitializer() {
            return biasInitializer;
        }

        public static Builder builder() {
            return new Builder(defaults());
        }

        public static class Builder {
            private final Options options;

            public Builder(Options options) {
                this.options = options;
            }

            public Builder setActivation(Activations activation) {
                return setActivation(Activations.select(activation));
            }

            public Builder setActivation(Activation<? extends TNumber> activation) {
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


//    public static class Options<T extends Number> {
//        // Default parameters
//        private Activation<T> activation;
//        private Initializer<T> kernelInitializer;
//        private Initializer<T> biasInitializer;
//
//        public static <T extends Number> Options<T> defaults() {
//            return new Builder<T>(new Options<>())
//                    .setActivation(Activations.linear)
//                    .setKernelInitializer(Initializers.randomNormal)
//                    .setBiasInitializer(Initializers.zeros)
//                    .build();
//        }
//
//        public static <T extends Number> Builder<T> builder() {
//            return new Builder<>(defaults());
//        }
//
//        public static class Builder<T extends Number> {
//            private Options<T> options;
//
//            public Builder(Options<T> options) {
//                this.options = options;
//            }
//
//            public Builder<T> setActivation(Activations activation) {
//                return setActivation(Activations.select(activation));
//            }
//
//            public Builder<T> setActivation(Activation<T> activation) {
//                this.options.activation = activation;
//                return this;
//            }
//
//            public Builder<T> setKernelInitializer(Initializers kernelInitializer) {
//                return setKernelInitializer(Initializers.select(kernelInitializer));
//            }
//
//            public Builder<T> setKernelInitializer(Initializer<T> kernelInitializer) {
//                this.options.kernelInitializer = kernelInitializer;
//                return this;
//            }
//
//            public Builder<T> setBiasInitializer(Initializers biasInitializer) {
//                return setBiasInitializer(Initializers.select(biasInitializer));
//            }
//
//            public Builder<T> setBiasInitializer(Initializer<T> biasInitializer) {
//                this.options.biasInitializer = biasInitializer;
//                return this;
//            }
//
//            public Options<T> build() {
//                return this.options;
//            }
//        }
//    }
}