package org.tensorflow.keras.models;

import org.tensorflow.Graph;
import org.tensorflow.Shape;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.data.TensorFrame;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.losses.Loss;
import org.tensorflow.keras.losses.Losses;
import org.tensorflow.keras.metrics.Metric;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.keras.mixin.MetricFunction;
import org.tensorflow.keras.optimizers.Optimizer;
import org.tensorflow.keras.optimizers.Optimizers;
import org.tensorflow.op.Ops;

import java.util.ArrayList;
import java.util.List;

public abstract class Model<T extends Number> extends Layer<T> {
    Class<T> dtype;

    public Model(Class<T> dtype) {
        // TODO:  For now, models take in only 1 input
        super(1);
        this.dtype = dtype;
    }

    public abstract void compile(Ops tf, Optimizer<T> optimizer, Loss<T> loss, List<Metric<T>> metric)
            throws Exception;

    public void compile(Ops tf, CompileOptions<T> compilerBuilder) throws Exception {
        compile(tf, compilerBuilder.optimizer, compilerBuilder.loss, compilerBuilder.metrics);
    }

    public abstract void fit(Ops tf, GraphLoader<T> train, GraphLoader<T> test, int epochs, int batchSize);

    public void fit(Ops tf, GraphLoader<T> train, GraphLoader<T> test, FitOptions fitOptions) {
        fit(tf, train, test, fitOptions.epochs, fitOptions.batchSize);
    }

    @Override
    public final void build(Ops tf, Shape inputShape, Class<T> dtype) {
        throw new UnsupportedOperationException("Cannot create a Sequential model with inputShape");
    }

    public FitOptions fitOptions() {
        return new FitOptions();
    }

    public CompileOptions compileOptions() {
        return new CompileOptions();
    }


    public static class CompileOptions<T extends Number> {
        private List<Metric<T>> metrics;
        private Optimizer<T> optimizer;

        private Loss<T> loss;

        public static Builder builder() {
            return new Builder();
        }

        public static class Builder<T extends Number> {
            private CompileOptions<T> options;

            public Builder() {
                this.options = new CompileOptions();
            }

            public Builder setLoss(Losses lossType) {
                return setLoss(Losses.select(lossType));
            }

            public Builder setLoss(Loss<T>loss) {
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

            public Builder addMetric(Metric<T> metric) {
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


        public List<Metric<T>> getMetrics() {
            return this.metrics;
        }

        public Optimizer getOptimizer() {
            return this.optimizer;
        }

        public Loss<T> getLoss() {
            return this.loss;
        }
    }

    public static class FitOptions {
        public static int DEFAULT_EPOCHS = 1;
        public static int DEFAULT_BATCH_SIZE = 32;

        private int epochs = DEFAULT_EPOCHS;
        private int batchSize = DEFAULT_BATCH_SIZE;

        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private FitOptions options;

            public Builder() {
                this.options = new FitOptions();
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
