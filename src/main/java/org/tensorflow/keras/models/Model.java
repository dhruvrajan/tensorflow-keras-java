package org.tensorflow.keras.models;

import org.tensorflow.Graph;
import org.tensorflow.Shape;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.data.TensorFrame;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.losses.Loss;
import org.tensorflow.keras.losses.Losses;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.keras.mixin.MetricFunction;
import org.tensorflow.keras.optimizers.Optimizer;
import org.tensorflow.keras.optimizers.Optimizers;
import org.tensorflow.op.Ops;

import java.util.ArrayList;
import java.util.List;

public abstract class Model<T> extends Layer<T> {

    public Model() {
        // TODO:  For now, models take in only 1 input
        super(1);
    }

    public abstract void compile(Ops tf, Optimizer optimizer, Loss loss, List<MetricFunction> metric)
            throws Exception;

    public void compile(Ops tf, CompileOptions compilerBuilder) throws Exception {
        compile(tf, compilerBuilder.optimizer, compilerBuilder.loss, compilerBuilder.metrics);
    }

    public abstract void fit(Ops tf, GraphLoader<T> train, GraphLoader<T> test, int epochs, int batchSize);

    public void fit(Ops tf, GraphLoader<T> train, GraphLoader<T> test, FitOptions fitOptions) {
        fit(tf, train, test, fitOptions.epochs, fitOptions.batchSize);
    }

    @Override
    public final void build(Ops tf, Shape inputShape) {
        throw new UnsupportedOperationException("Cannot create a Sequential model with inputShape");
    }

    public FitOptions fitOptions() {
        return new FitOptions();
    }

    public CompileOptions compileOptions() {
        return new CompileOptions();
    }


    public static class CompileOptions {
        private List<MetricFunction> metrics;
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

            public Builder setOptimizer(String optimizerName) {
                return setOptimizer(Optimizer.select(optimizerName));
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

            public Builder addMetric(MetricFunction metric) {
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


        public List<MetricFunction> getMetrics() {
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
        int epochs = 10;
        int batchSize = 1;

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
