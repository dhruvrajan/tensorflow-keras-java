package org.tensorflow.keras.models;

import org.tensorflow.Shape;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.losses.Loss;
import org.tensorflow.keras.losses.Losses;
import org.tensorflow.keras.metrics.Metric;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.keras.optimizers.Optimizer;
import org.tensorflow.keras.optimizers.Optimizers;
import org.tensorflow.op.Ops;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.keras.callbacks.Callback;
import org.tensorflow.keras.callbacks.Callbacks;

public abstract class Model<T extends Number> extends Layer<T> {
    public Model(Class<T> dtype) {
        // TODO:  For now, models take in only 1 input
        super(1);
        this.dtype = dtype;
        this.built = true;
    }

    public abstract void compile(Ops tf, Optimizer<T> optimizer, Loss loss, List<Metric> metric)
            throws Exception;

    public void compile(Ops tf, CompileOptions compilerBuilder) throws Exception {
        compile(tf, (Optimizer<T>) compilerBuilder.optimizer, compilerBuilder.loss,compilerBuilder.metrics);
    }

    public abstract void fit(Ops tf, GraphLoader<T> train, GraphLoader<T> test, int epochs, int batchSize, List<Callback> callbacks);

    public void fit(Ops tf, GraphLoader<T> train, GraphLoader<T> test, FitOptions fitOptions) {
        fit(tf, train, test, fitOptions.epochs, fitOptions.batchSize, fitOptions.callbacks);
    }

    public static class CompileOptions {
        private List<Metric> metrics;
        private Optimizer optimizer;

        private Loss loss;

        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private CompileOptions options;

            public Builder() {
                this.options = new CompileOptions();
            }

            public Builder setLoss(Losses lossType) {
                return setLoss(Losses.select(lossType));
            }

            public Builder setLoss(Loss loss) {
                options.loss = loss;
                return this;
            }

            public Builder setOptimizer(Optimizer optimizer) {
                options.optimizer = optimizer;
                return this;
            }

            public Builder setOptimizer(Optimizers optimizerType) {
                return setOptimizer(Optimizers.select(optimizerType));
            }

            public Builder addMetric(Metrics metric) {
                return addMetric(Metrics.select(metric));
            }

            public Builder addMetric(Metric metric) {
                if (options.metrics == null) {
                    options.metrics = new ArrayList<>();
                }
                options.metrics.add(metric);
                return this;
            }

            public CompileOptions build() {
                return options;
            }
        }


        public List<Metric> getMetrics() {
            return this.metrics;
        }

        public Optimizer getOptimizer() {
            return this.optimizer;
        }

        public Loss getLoss() {
            return this.loss;
        }
    }

    public static class FitOptions {
        private int epochs;
        private int batchSize;
        private List<Callback> callbacks;

    
        public int getEpochs() {
            return epochs;
        }

        public int getBatchSize() {
            return batchSize;
        }

        public static FitOptions defaults() {
            return new Builder()
                    .setEpochs(1)
                    .setBatchSize(32)
                    .build();
        }

        public static Builder builder() {
            return new Builder(defaults());
        }

        public static class Builder {
            private FitOptions options;

            public Builder() { options = new FitOptions(); options.callbacks = new ArrayList<>();}
            private Builder(FitOptions options) {
                this.options = options;
            }

            public Builder setEpochs(int epochs) {
                options.epochs = epochs;
                return this;
            }

            public Builder setBatchSize(int batchSize) {
                options.batchSize = batchSize;
                return this;
            }

            public Builder addCallback(Callback callback) {
                options.callbacks.add(callback);
                return this;
            }

            public Builder addCallback(Callbacks callbackType) {
                return addCallback(Callbacks.select(callbackType));
            }



            public FitOptions build() {
                return options;
            }

        }
    }
}
