package org.tensorflow.keras.models;

import org.tensorflow.data.GraphLoader;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.losses.Loss;
import org.tensorflow.keras.losses.Losses;
import org.tensorflow.keras.metrics.Metric;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.keras.optimizers.Optimizer;
import org.tensorflow.keras.optimizers.Optimizers;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

import java.util.ArrayList;
import java.util.List;

public abstract class Model<T extends TNumber> extends Layer<T> {
    public Model(Class<T> dtype) {
        // TODO:  For now, models take in only 1 input
        super(1);
        this.dtype = dtype;
        this.built = true;
    }

    public abstract void compile(Ops tf, Optimizer<T> optimizer, Loss loss, List<Metric> metric)
            throws Exception;

    public void compile(Ops tf, CompileOptions<T> compilerBuilder) throws Exception {
        compile(tf, compilerBuilder.optimizer, compilerBuilder.loss,compilerBuilder.metrics);
    }

    public abstract void fit(Ops tf, GraphLoader<T> train, GraphLoader<T> test, int epochs, int batchSize);

    public void fit(Ops tf, GraphLoader<T> train, GraphLoader<T> test, FitOptions fitOptions) {
        fit(tf, train, test, fitOptions.epochs, fitOptions.batchSize);
    }

    public static class CompileOptions<T extends TNumber> {
        private List<Metric> metrics;
        private Optimizer<T> optimizer;

        private Loss loss;

        public static <T extends TNumber> Builder<T> builder() {
            return new Builder<>();
        }

        public static class Builder<T extends TNumber> {
            private final CompileOptions<T> options;

            public Builder() {
                this.options = new CompileOptions<>();
            }

            public Builder<T> setLoss(Losses lossType) {
                return setLoss(Losses.select(lossType));
            }

            public Builder<T> setLoss(Loss loss) {
                options.loss = loss;
                return this;
            }

            public Builder<T> setOptimizer(Optimizer<T> optimizer) {
                options.optimizer = optimizer;
                return this;
            }

            public Builder<T> setOptimizer(Optimizers optimizerType) {
                return setOptimizer(Optimizers.select(optimizerType));
            }

            public Builder<T> addMetric(Metrics metric) {
                return addMetric(Metrics.select(metric));
            }

            public Builder<T> addMetric(Metric metric) {
                if (options.metrics == null) {
                    options.metrics = new ArrayList<>();
                }
                options.metrics.add(metric);
                return this;
            }

            public CompileOptions<T> build() {
                return options;
            }
        }


        public List<Metric> getMetrics() {
            return this.metrics;
        }

        public Optimizer<T> getOptimizer() {
            return this.optimizer;
        }

        public Loss getLoss() {
            return this.loss;
        }
    }

    public static class FitOptions {
        private int epochs;
        private int batchSize;

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
            private final FitOptions options;

            public Builder() { options = new FitOptions(); }
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

            public FitOptions build() {
                return options;
            }

        }
    }
}
