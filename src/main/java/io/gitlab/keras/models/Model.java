package io.gitlab.keras.models;

import io.gitlab.keras.datasets.Dataset;
import io.gitlab.keras.layers.Layer;
import io.gitlab.keras.losses.Loss;
import io.gitlab.keras.metrics.Metric;
import io.gitlab.keras.mixin.MetricFunction;
import io.gitlab.keras.optimizers.Optimizer;
import org.tensorflow.Graph;
import org.tensorflow.op.Ops;

import java.util.ArrayList;
import java.util.List;

public abstract class Model<T> extends Layer<T> {

    public abstract void compile(Ops tf, Optimizer optimizer, Loss loss, List<MetricFunction> metric) throws Exception;
    public void compile(Ops tf, CompilerBuilder compilerBuilder) throws Exception {
        compile(tf,compilerBuilder.optimizer, compilerBuilder.loss, compilerBuilder.metrics);
    }

    public abstract void fit(Graph graph, Dataset data, int epochs, int batchSize);
//    public abstract void fit(Graph graph, List<float[][]> data, List<float[][]> labels, int epochs, int batchSize,
//                             List<float[][]> validationData, List<float[][]> validationLabels);
    public void fit(Graph graph, FitBuilder fitBuilder) {
        fit(graph, fitBuilder.data, fitBuilder.epochs, fitBuilder.batchSize);
    }


    public static class CompilerBuilder {
        private Graph graph;
        private List<MetricFunction> metrics;
        private Optimizer optimizer;

        private Loss loss;

        public CompilerBuilder(Graph graph) {
            this.graph = graph;
        }

        public CompilerBuilder(Graph graph, Optimizer optimizer) {
            this.optimizer = optimizer;
        }

        public CompilerBuilder(Graph graph, Optimizer optimizer, Loss loss) {
            this.optimizer = optimizer;
            this.loss = loss;
        }

        public CompilerBuilder(Graph graph, Optimizer optimizer, Loss loss, List<MetricFunction> metrics) {
            this.optimizer = optimizer;
            this.loss = loss;
            this.metrics = metrics;
        }

        public CompilerBuilder setLoss(String lossName) {
            return setLoss(Loss.select(lossName));
        }

        public CompilerBuilder setLoss(Loss loss) {
            this.loss = loss;
            return this;
        }

        public CompilerBuilder setOptimizer(String optimizerName) {
            return setOptimizer(Optimizer.select(optimizerName));
        }

        public CompilerBuilder setOptimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public CompilerBuilder addMetric(String metricName) {
            return addMetric(Metric.select(metricName));
        }

        public CompilerBuilder addMetric(MetricFunction metric) {
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

    public static class FitBuilder {
        Dataset data;
        int epochs = 10;
        int batchSize = 1;

        public FitBuilder() {

        }

        public FitBuilder(Dataset data) {
            this.data = data;
        }


        public Dataset getData() {
            return data;
        }

        public FitBuilder setData(Dataset data) {
            this.data = data;
            return this;
        }

        public int getEpochs() {
            return epochs;
        }

        public FitBuilder setEpochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public int getBatchSize() {
            return batchSize;
        }

        public FitBuilder setBatchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }
    }
}
