/**
 * A single-layer perceptron with two inputs and one bias node,
 * using a step activation function and the perceptron learning rule.
 * <p>
 * Weight update rule: delta_w = learningRate * (target - output) * input
 */
public class Perceptron {

    private double[] weights;       // [W0_bias, W1, W2]
    private double learningRate;
    private boolean verbose;

    /**
     * Constructs a perceptron with the given initial weights and learning rate.
     *
     * @param initialWeights array of three weights [W0_bias, W1, W2]
     * @param learningRate   the learning rate applied to each weight update
     * @param verbose        if true, prints per-pass training details to stdout
     */
    public Perceptron(double[] initialWeights, double learningRate, boolean verbose) {
        this.weights = initialWeights.clone();
        this.learningRate = learningRate;
        this.verbose = verbose;
    }

    /**
     * Applies the step activation function.
     *
     * @param net the net input value
     * @return    1 if net >= 0.0, otherwise 0
     */
    private int activate(double net) {
        return (net >= 0.0) ? 1 : 0;
    }

    /**
     * Computes the net input: W0*1 + W1*x1 + W2*x2.
     * Result is rounded to 10 decimal places to suppress floating-point noise
     * (e.g. -0.3 + 0.3 yielding -1e-17 instead of 0).
     *
     * @param inputs array of two input values [x1, x2]
     * @return       the rounded net input value
     */
    private double computeNet(double[] inputs) {
        double net = weights[0] + weights[1] * inputs[0] + weights[2] * inputs[1];
        return Math.round(net * 1e10) / 1e10;
    }

    /**
     * Trains the perceptron until all examples are correctly classified.
     * After each wrong prediction, weights are updated and training restarts
     * from the first example (online learning with immediate restart).
     *
     * @param inputs   2D array of training inputs; each row is [x1, x2]
     * @param targets  expected binary outputs (0 or 1) for each input row
     * @return         the total number of passes through the training set
     */
    public int train(double[][] inputs, int[] targets) {
        int passes = 0;

        while (true) {
            passes++;
            boolean errorFound = false;

            if (verbose) {
                System.out.printf("%n=== Pass %d ===%n", passes);
                System.out.printf("Weights: W0=%.4f  W1=%.4f  W2=%.4f%n",
                        weights[0], weights[1], weights[2]);
            }

            for (int i = 0; i < inputs.length; i++) {
                double net    = computeNet(inputs[i]);
                int    output = activate(net);
                int    target = targets[i];

                if (verbose) {
                    System.out.printf("  (x1=%.1f, x2=%.1f): net=%.4f  output=%d  target=%d",
                            inputs[i][0], inputs[i][1], net, output, target);
                }

                if (output != target) {
                    double error = target - output;
                    double oldW0 = weights[0], oldW1 = weights[1], oldW2 = weights[2];

                    weights[0] += learningRate * error * 1.0;
                    weights[1] += learningRate * error * inputs[i][0];
                    weights[2] += learningRate * error * inputs[i][1];

                    if (verbose) {
                        System.out.printf("  <- WRONG%n");
                        System.out.printf("     dW0=%.4f  dW1=%.4f  dW2=%.4f%n",
                                weights[0]-oldW0, weights[1]-oldW1, weights[2]-oldW2);
                        System.out.printf("     W0: %.4f->%.4f  W1: %.4f->%.4f  W2: %.4f->%.4f%n",
                                oldW0, weights[0], oldW1, weights[1], oldW2, weights[2]);
                    }

                    errorFound = true;
                    break;
                } else {
                    if (verbose) System.out.printf("  <- correct%n");
                }
            }

            if (!errorFound) break;
        }

        return passes;
    }

    /**
     * Returns a copy of the current weight vector.
     *
     * @return a cloned array of the current weights [W0_bias, W1, W2]
     */
    public double[] getWeights() {
        return weights.clone();
    }
}