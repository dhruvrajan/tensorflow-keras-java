package io.gitlab.tensorflow.keras.metrics;

public enum Metrics {
    accuracy,
    binary_accuracy,
    categorical_accuracy,
    sparse_categorical_accuracy,
    top_k_categorical_accuracy,
    sparse_top_k_categorical_accuracy;

    public static Metric select(Metrics metricType) {
        switch (metricType) {
            case accuracy:
                return new Accuracy();
//            case binary_accuracy:
//                return new BinaryAccuracy();
            case categorical_accuracy:
                return new CategoricalAccuracy();
            case sparse_categorical_accuracy:
                return new SparseCategoricalAccuracy();
//            case top_k_categorical_accuracy:
//                return new TopKCategoricalAccuracy();
            case sparse_top_k_categorical_accuracy:
                return new SparseTopKCategoricalAccuracy();
            default:
                throw new IllegalArgumentException("Invalid metric type");
        }
    }
}