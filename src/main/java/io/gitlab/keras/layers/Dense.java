package io.gitlab.keras.layers;


import io.gitlab.keras.activations.Activation;
import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;


public class Dense extends Layer<Float> {
    int units;
    Shape inputShape;

    // kernel matrix
    Variable<Float> kernel;
    Assign<Float> kernelInit;

    // bias
    Variable<Float> bias;
    Assign<Float> biasInit;

    // softmax
    Sigmoid<Float> softmax;
    Activation<Float> activation;


    public Dense(int units, Shape inputShape) {
        super();
        this.inputShape = inputShape;
        this.units = units;
        System.out.println("dense layer " + this.id + " shape" + inputShape.toString());

    }

    public Dense setActivation(String activationName) {
        return setActivation(Activation.select(activationName));
    }

    public Dense setActivation(Activation activation) {
        this.activation = activation;
        return this;
    }

    public Operand build(Ops tf, Operand iris) {
        tf = tf.withName("Dense_Layer_ID_" + this.id);
        Shape kernelShape = Shape.make(
                inputShape.size(inputShape.numDimensions() - 1),
                this.units
        );

        // Create dense kernel tensor
        this.kernel = addWeight("kernel", tf.variable(kernelShape, Float.class));
        this.kernelInit = addInitializer("kernelInit", tf.assign(this.kernel, tf.zeros(constArray(tf, kernelShape.size(0), kernelShape.size(1)), Float.class)));


        Shape biasShape = Shape.make(this.units);

        // Create bias tensor
        System.out.println("-> " + this.id + " kernelShape " + kernelShape + " biasShape " + biasShape);
        this.bias = addWeight("bias", tf.variable(biasShape, Float.class));
        this.biasInit = addInitializer("biasInit", tf.assign(this.bias, tf.zeros(constArray(tf, biasShape.size(0)), Float.class)));

        // Apply activation
        Operand<Float> signal = tf.add(tf.matMul(iris, this.kernel), this.bias);

        return activation.build(tf, signal);
    }

    private static Operand constArray(Ops tf, long... i) { return tf.constant(i); }
}

