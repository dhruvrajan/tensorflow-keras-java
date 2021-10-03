//package org.tensorflow.keras.layers;
//
//import org.tensorflow.Operand;
//import org.tensorflow.keras.activations.Activation;
//import org.tensorflow.keras.activations.Activations;
//import org.tensorflow.keras.initializers.Initializer;
//import org.tensorflow.keras.initializers.Initializers;
//import org.tensorflow.ndarray.Shape;
//import org.tensorflow.op.Ops;
//import org.tensorflow.types.TFloat32;
//
//public class Conv2D extends Layer<TFloat32> {
//
//    enum Padding {valid, same}
//
//    enum DataFormat {channelsLast, channelsFirst}
//
//    org.tensorflow.op.nn.Conv2d<TFloat32> convOp;
//
//    private Conv2D(long filters, long[] kernel) {
//        super(1);
//    }
//
//    @Override
//    public void build(Ops tf, Shape inputShape) {
//    }
//
//    @Override
//    public Shape computeOutputShape(Shape inputShape) {
//        throw new UnsupportedOperationException("Not yet implemented"); // XXX TODO
//    }
//
//    @SafeVarargs
//    @Override
//    protected final Operand<TFloat32> call(Ops tf, Operand<TFloat32>... inputs) {
//        return this.call(tf, inputs[0]);
//    }
//
//    private Operand<TFloat32> call(Ops tf, Operand<TFloat32> input) {
//        throw new UnsupportedOperationException("Not yet implemented"); // XXX TODO
//    }
//
//    static class Options {
//        long[] strides = new long[]{1, 1};
//        long[] dilationRate = new long[]{1, 1};
//
//        Padding padding = Padding.valid;
//        DataFormat dataFormat = DataFormat.channelsLast;
//
//        Activation<TFloat32> activation = Activations.select(Activations.linear);
//        Initializer/*<TFloat32>*/ kernelInitializer = Initializers.select(Initializers.randomNormal);
//        Initializer/*<TFloat32>*/ biasInitializer = Initializers.select(Initializers.zeros);
//        boolean useBias = true;
//
//        public Builder builder() {
//            return new Builder();
//        }
//
//        static class Builder {
//            Options options = new Options();
//
//            public Builder setStrides(long strideHeight, long strideWidth) {
//                options.strides = new long[]{strideHeight, strideWidth};
//                return this;
//            }
//
//            public Builder setDilationRate(long dilationHeight, long dilationWidth) {
//                options.strides = new long[]{dilationHeight, dilationWidth};
//                return this;
//            }
//
//          public Builder setPadding(Padding padding) {
//            options.padding = padding;
//            return this;
//          }
//
//          public Builder setDataFormat(DataFormat dataFormat) {
////            options.strides ;
//            return this;
//          }
//        }
//    }
//}
