package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Conv2D extends Layer<Float> {
    ConvOptions options;
    long filtersIn;
    long[] kernelIn;
    Constant<Float> filters;
    Constant<Long> strides;
    private Conv2D(long filters, long[] kernel) {
        super(1);
        this.options = ConvOptions.defaults(2, filters, kernel);
        this.filtersIn = filters;
        this.kernelIn = kernel;
    }

    @Override
    public void build(Ops tf, Shape inputShape) {
        filters = tf.constant(filtersIn, Float.class);
        strides = tf.constant(options.getStrides());
    }

    @Override
    public Shape computeOutputShape(Shape inputShape) {
        return inputShape;
    }

    @Override
    protected Operand<Float> call(Ops tf, Operand<Float>... inputs) {
        return this.call(tf, inputs[0]);
    }

    public static List<Long> toList(long[] arr) {
        return Arrays.stream(arr).boxed().collect(Collectors.toList());
    }

    private Operand<Float> call(Ops tf, Operand<Float> input) {
        return tf.conv2D(input, filters,
                toList(options.getStrides()),
                options.getPadding().name(),
                org.tensorflow.op.core.Conv2D
                        .dataFormat(options.getDataFormat().name())
                        .dilations(toList(options.getDilationRate())));
    }

}
