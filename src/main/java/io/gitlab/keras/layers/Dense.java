package io.gitlab.keras.layers;


import io.gitlab.keras.activations.Activation;
import io.gitlab.keras.activations.Activations;
import io.gitlab.keras.initializers.Initializer;
import io.gitlab.keras.initializers.Initializers;
import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;


public class Dense extends Layer<Float> {
    private static int DENSE_INPUT_LENGTH = 1;
    private int units;
    private Shape inputShape;

    private String KERNEL = "kernel";
    private String KERNEL_INIT = "kernelInit";
    private String BIAS = "bias";
    private String BIAS_INIT = "biasInit";

    // weight tensors
    private Variable<Float> kernel;
    private Variable<Float> bias;

    // initializers
    private Initializer<Float> kernelInitializer;
    private Initializer<Float> biasInitializer;

    // activation function
    private Activation<Float> activation;


    public Dense(int units, Shape inputShape) {
        super(DENSE_INPUT_LENGTH);
        this.inputShape = inputShape;
        this.units = units;
    }

    public Dense setActivation(String activationName) {
        return setActivation(Activations.select(activationName));
    }

    public Dense setActivation(Activations activation) {
        return setActivation(Activations.select(activation));
    }

    public Dense setActivation(Activation<Float> activation) { this.activation = activation; return this; }

    public Dense setKernelInitializer(Initializers initializer) {
        this.kernelInitializer = Initializers.select(initializer);
        return this;
    }

    public Dense setBiasInitializer(Initializers initializer) {
        this.biasInitializer = Initializers.select(initializer);
        return this;
    }


    public void build(Ops tf) {
        tf = tf.withName("Dense_Layer_" + this.id);
        Shape kernelShape = Shape.make(
                inputShape.size(inputShape.numDimensions() - 1),
                this.units
        );

        Shape biasShape = Shape.make(this.units);

        // Create dense kernel tensor
        this.kernel = addWeight(KERNEL, tf.variable(kernelShape, Float.class));
        if (this.kernelInitializer != null) {
            addInitializer(KERNEL_INIT, this.kernelInitializer);
            this.kernelInitializer.build(tf, this.kernel);
        }

        // Create bias tensor
        this.bias = addWeight(BIAS, tf.variable(biasShape, Float.class));
        if (this.biasInitializer != null) {
            addInitializer(BIAS_INIT, this.biasInitializer);
            this.biasInitializer.build(tf, this.bias);
        }

        this.built = true;
    }

    @SafeVarargs
    public final Operand<Float> call(Ops tf, Operand<Float>... inputs) {
        return this.call(tf, inputs[0]);
    }

    private Operand<Float> call(Ops tf, Operand<Float> input) {
        Operand<Float> signal = tf.math.add(tf.linalg.matMul(input, this.kernel), this.bias);
        return this.activation.call(tf, signal);
    }
}

