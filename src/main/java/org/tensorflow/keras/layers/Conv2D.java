package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.keras.activations.Activation;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.op.Ops;

public class Conv2D extends Layer<Float> {

    enum Padding {valid, same}

    enum DataFormat {channelsLast, channelsFirst}

    org.tensorflow.op.core.Conv2D<Float> convOp;

    private Conv2D(long filters, long[] kernel) {
        super(1);
    }

    @Override
    public void build(Ops tf, Shape inputShape) {
        convOp = tf.conv2D()
    }

    @Override
    public Shape computeOutputShape(Shape inputShape) {
        return null;
    }

    @Override
    protected Operand<Float> call(Ops tf, Operand<Float>... inputs) {
        return null;
    }

    class Options {
        long[] strides = new long[]{1, 1};
        long[] dilationRate = new long[]{1, 1};

        Padding padding = Padding.valid;
        DataFormat dataFormat = DataFormat.channelsLast;

        Activation<Float> activation = Activations.select(Activations.linear);
        Initializer<Float> kernelInitializer = Initializers.select(Initializers.randomNormal);
        Initializer<Float> biasInitializer = Initializers.select(Initializers.zeros);
        boolean useBias = true;

        public Builder builder() {
            return new Builder();
        }

        class Builder {
            Options options = new Options();

            public Builder setStrides(long strideHeight, long strideWidth) {
                options.strides = new long[]{strideHeight, strideWidth};
                return this;
            }

            public Builder setDilationRate(long dilationHeight, long dilationWidth) {
                options.strides = new long[]{dilationHeight, dilationWidth};
                return this;
            }

          public Builder setPadding(Padding padding) {
            options.padding = padding;
            return this;
          }

          public Builder setDataFormat(DataFormat dataFormat) {
            options.strides ;
            return this;
          }
        }
    }
}
