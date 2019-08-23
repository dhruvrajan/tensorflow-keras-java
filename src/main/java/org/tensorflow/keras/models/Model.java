package org.tensorflow.keras.models;

import org.tensorflow.Graph;
import org.tensorflow.Shape;
import org.tensorflow.keras.data.Dataset;
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

  //    public abstract void fit(Graph graph, Dataset data, int epochs, int batchSize);
  //    public abstract void fit(Graph graph, List<float[][]> data, List<float[][]> labels, int
  // epochs, int batchSize,
  //                             List<float[][]> validationData, List<float[][]> validationLabels);
  //    public void fit(Graph graph, FitOptions fitBuilder) {
  //        fit(graph, fitBuilder.data, fitBuilder.epochs, fitBuilder.batchSize);
  //    }

  @Override
  public final void build(Ops tf, Shape inputShape) {
    throw new UnsupportedOperationException("Cannot create a Sequential model with inputShape");
  }

  public static class CompilerOptions {
    private Graph graph;
    private List<MetricFunction> metrics;
    private Optimizer optimizer;

    private Loss loss;

    public CompilerOptions(Graph graph) {
      this.graph = graph;
    }

    public CompilerOptions(Graph graph, Optimizer optimizer) {
      this.optimizer = optimizer;
    }

    public CompilerOptions(Graph graph, Optimizer optimizer, Loss loss) {
      this.optimizer = optimizer;
      this.loss = loss;
    }

    public CompilerOptions(
        Graph graph, Optimizer optimizer, Loss loss, List<MetricFunction> metrics) {
      this.optimizer = optimizer;
      this.loss = loss;
      this.metrics = metrics;
    }

    public CompilerOptions setLoss(Losses lossType) {
      return setLoss(Losses.select(lossType));
    }
    //        public CompilerOptions setLoss(String lossName) {
    //            return setLoss(Losses.select(lossName));
    //        }

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

    public CompilerOptions addMetric(String metricName) {
      return addMetric(Metric.select(metricName));
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
    Dataset data;
    int epochs = 10;
    int batchSize = 1;

    public FitOptions() {}

    public FitOptions(Dataset data) {
      this.data = data;
    }

    public Dataset getData() {
      return data;
    }

    public FitOptions setData(Dataset data) {
      this.data = data;
      return this;
    }

    public int getEpochs() {
      return epochs;
    }

    public FitOptions setEpochs(int epochs) {
      this.epochs = epochs;
      return this;
    }

    public int getBatchSize() {
      return batchSize;
    }

    public FitOptions setBatchSize(int batchSize) {
      this.batchSize = batchSize;
      return this;
    }
  }
}
