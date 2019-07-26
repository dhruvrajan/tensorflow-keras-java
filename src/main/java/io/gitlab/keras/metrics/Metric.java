package io.gitlab.keras.metrics;

import io.gitlab.keras.layers.Layer;
import io.gitlab.keras.mixin.MetricFunction;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public abstract class Metric extends Layer<Float> implements MetricFunction {

    public abstract Operand<Float> build(Ops tf, Operand<Float> output, Placeholder<Float> label) throws Exception;

    public static Metric select(String s) { return select(MetricType.valueOf(s)); }

    private static Metric select(MetricType metricType) {
        return MetricType.select(metricType);
    }

    @Override
    public Operand<Float> apply(Ops tf, Operand<Float> output, Placeholder<Float> label) throws Exception {
        return build(tf, output, label);
    }

    public enum MetricType {
        accuracy,
        binary_accuracy,
        categorical_accuracy,
        sparse_categorical_accuracy,
        top_k_categorical_accuracy,
        sparse_top_k_categorical_accuracy;

         static Metric select(MetricType metricType) {
            switch (metricType) {
                case accuracy:
                    return new Accuracy();
                case binary_accuracy:
                    return new BinaryAccuracy();
                case categorical_accuracy:
                    return new CategoricalAccuracy();
                case sparse_categorical_accuracy:
                    return new SparseCategoricalAccuracy();
                case top_k_categorical_accuracy:
                    return new TopKCategoricalAccuracy();
                case sparse_top_k_categorical_accuracy:
                    return new SparseTopKCategoricalAccuracy();
                default:
                    throw new IllegalArgumentException("Invalid metric type");
            }
        }
    }
}
