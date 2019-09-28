package org.tensorflow.keras.models;

import org.tensorflow.Graph;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
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

    public void compile(Ops tf, CompilerOptions compilerBuilder) throws Exception {
        compile(tf, compilerBuilder.optimizer, compilerBuilder.loss, compilerBuilder.metrics);
    }

    public abstract void fit(Ops tf, GraphLoader<Float> train, GraphLoader<Float> test, int epochs, int batchSize);

    public abstract void fit(Graph graph, List<float[][]> data, List<float[][]> labels, int
            epochs, int batchSize,
                             List<float[][]> validationData, List<float[][]> validationLabels);

    public void fit(Ops tf, FitOptions fitOptions) {
        fit(tf, fitOptions.train, fitOptions.test, fitOptions.epochs, fitOptions.batchSize);
    }

    @Override
    public final void build(Ops tf, Shape inputShape) {
        throw new UnsupportedOperationException("Cannot create a Sequential model with inputShape");
    }

    public FitOptions fitOptions() {
      return new FitOptions();
    }

    public CompilerOptions compilerOptions() {
      return new CompilerOptions();
    }


    public static class CompilerOptions {
        private Graph graph;
        private List<MetricFunction> metrics;
        private Optimizer optimizer;

        private Loss loss;

        public CompilerOptions() {
        }

        public CompilerOptions create(Graph graph) {
            this.graph = graph;
            return this;
        }

        public CompilerOptions setLoss(Losses lossType) {
            return setLoss(Losses.select(lossType));
        }

        public CompilerOptions setLoss(Loss loss) {
            this.loss = loss;
            return this;
        }

        public CompilerOptions setOptimizer(String optimizerName) {
            return setOptimizer(Optimizer.select(optimizerName));
        }

        public CompilerOptions setOptimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public CompilerOptions setOptimizer(Optimizers optimizerType) {
            return setOptimizer(Optimizers.select(optimizerType));
        }

        public CompilerOptions addMetric(Metrics metric) {
            return addMetric(Metrics.select(metric));
        }

        public CompilerOptions addMetric(MetricFunction metric) {
            if (this.metrics == null) {
                this.metrics = new ArrayList<>();
            }
            this.metrics.add(metric);
            return this;
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

        public Graph getGraph() {
            return graph;
        }

        public void setGraph(Graph graph) {
            this.graph = graph;
        }
    }

    public static class FitOptions {
        TensorFrame<Float> train;
        TensorFrame<Float> test;
        int epochs = 10;
        int batchSize = 1;

        public void setTrain(TensorFrame<Float> train) {
            this.train = train;
        }

        public void setTest(TensorFrame<Float> test) {
            this.test = test;
        }

        public void setEpochs(int epochs) {
            this.epochs = epochs;
        }

        public void setBatchSize(int batchSize) {
            this.batchSize = batchSize;
        }

        public FitOptions() {
        }

        public FitOptions create(TensorFrame<Float> train, TensorFrame<Float> test) {
          this.train = train;
          this.test = test;
          return this;
        }

    }
}
