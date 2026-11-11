import java.util.Random;

/**
 * Some very basic stuff to get you started. It shows basically how each
 * chromosome is built.
 *
 * @author Jo Stevens
 * @version 1.0, 14 Nov 2008
 *
 * @author Alard Roebroeck
 * @version 1.1, 12 Dec 2012
 *
 */

public class PracticalGAs {

    static final String TARGET = "HELLO WORLD";
    static char[] alphabet = new char[27];
    static Random generator = new Random(System.currentTimeMillis());
    static Individual[] population;

    /**
     * Runs all four GA experiments (population size, mutation rate,
     * no crossover, no mutation) and prints average generations to stdout.
     *
     * @param args command-line arguments (not used)
     */
    public static void main(String[] args) {
        for (char c = 'A'; c <= 'Z'; c++) {
            alphabet[c - 'A'] = c;
        }
        alphabet[26] = ' ';

        int[] popSizes = {10, 25, 50, 100, 200, 500};
        double[] mutationRates = {0.001, 0.005, 0.01, 0.05, 0.1, 0.2};  // all are valid according to TU Delft paper

        System.out.println("Q1: Population Size");
        for (int popSize : popSizes) {
            int total = 0;
            int eliteCount = 10;
            int elitePoolSize = 10;

            if (eliteCount >= popSize) {
                eliteCount = eliteCount / 2;
            }
            if (elitePoolSize >= popSize) {
                elitePoolSize = elitePoolSize / 2;
            }

            for (int i = 0; i < 10; i++) {
                total += runGA(popSize, 0.1, eliteCount, elitePoolSize, true, true);
            }
            System.out.println("popSize=" + popSize + " avg generations=" + (total / 10.0));
        }

        System.out.println("Q2: Mutation Rate");
        for (double mutationRate : mutationRates) {
            int total = 0;
            for (int j = 0; j < 10; j++) {
                total += runGA(100, mutationRate, 10, 10, true, true);
            }
            System.out.println("mutationRate=" + mutationRate + " avg generations=" + (total / 10.0));
        }

        System.out.println("Q3: No Crossover");
        for (int popSize : popSizes) {
            int total = 0;
            for (int i = 0; i < 10; i++) {
                total += runGA(popSize, 0.1, 10, 10, false, true);
            }
            System.out.println("popSize=" + popSize + " avg generations=" + (total / 10.0));
        }

        System.out.println("Q4: No Mutation");
        for (int popSize : popSizes) {
            int total = 0;
            for (int i = 0; i < 10; i++) {
                total += runGA(popSize, 0.1, 10, 10, true, false);
            }
            System.out.println("popSize=" + popSize + " avg generations=" + (total / 10.0));
        }
    }

    /**
     * Runs one complete GA trial and returns the number of generations to convergence or 10000 if DNF
     * @param popSize       number of individuals in the population
     * @param mutationRate  probability of mutating each gene per generation
     * @param eliteCount    number of top individuals carried over unchanged each generation
     * @param elitePoolSize number of top-ranked individuals eligible as parents
     * @param useCrossover  whether to apply two-point crossover
     * @param useMutation   whether to apply mutation
     * @return generations to convergence, or 10000 if (DNF)
     */
    static int runGA(int popSize, double mutationRate, int eliteCount, int elitePoolSize,
                     boolean useCrossover, boolean useMutation) {
        int generation = 0;
        population = new Individual[popSize];
        for (int i = 0; i < popSize; i++) {
            char[] tempChromosome = new char[TARGET.length()];
            for (int j = 0; j < TARGET.length(); j++) {
                tempChromosome[j] = alphabet[generator.nextInt(alphabet.length)];
            }
            population[i] = new Individual(tempChromosome);
        }
        // Set initial fitness
        for (int i = 0; i < popSize; i++) {
            population[i].setFitness(evaluateFitness(population[i]));
        }

        while (true) {
            for (int i = 0; i < popSize; i++) {
                population[i].setFitness(evaluateFitness(population[i]));
            }
            // HeapSort sorts descending: best individual is at index 0
            HeapSort.sort(population);
            if (population[0].getFitness() == 11) {
                break;
            }
            if (generation >= 10000) {
                return 10000;
            }

            Individual[] newPopulation = new Individual[popSize];

            // Carry over the top eliteCount individuals unchanged
            for (int i = 0; i < eliteCount; i++) {
                newPopulation[i] = population[i].clone();
            }

            for (int i = eliteCount; i < popSize; i++) {
                // Elitist selection: select two parents from the top elitePoolSize individuals
                int individualAIndex = generator.nextInt(elitePoolSize);
                int individualBIndex = generator.nextInt(elitePoolSize);

                char[] childChromosome = new char[TARGET.length()];
                if (useCrossover) {
                    // Two-point crossover: swap the segment between two cut points
                    int crossPoint1 = generator.nextInt(TARGET.length());
                    int crossPoint2 = generator.nextInt(TARGET.length());
                    if (crossPoint1 > crossPoint2) {
                        int temp = crossPoint1;
                        crossPoint1 = crossPoint2;
                        crossPoint2 = temp;
                    }
                    for (int j = 0; j < TARGET.length(); j++) {
                        if (j < crossPoint1 || j > crossPoint2) {
                            childChromosome[j] = population[individualAIndex].getChromosome()[j];
                        } else {
                            childChromosome[j] = population[individualBIndex].getChromosome()[j];
                        }
                    }
                } else {
                    // No crossover: child is a copy of parent A
                    for (int j = 0; j < TARGET.length(); j++) {
                        childChromosome[j] = population[individualAIndex].getChromosome()[j];
                    }
                }

                if (useMutation) {
                    for (int j = 0; j < TARGET.length(); j++) {
                        if (generator.nextDouble() < mutationRate) {
                            childChromosome[j] = alphabet[generator.nextInt(alphabet.length)];
                        }
                    }
                }
                newPopulation[i] = new Individual(childChromosome);
            }
            population = newPopulation;
            generation++;
        }
        return generation;
    }

    /**
     * Evaluates fitness as the number of characters matching the target
     * at the correct position. Maximum fitness is TARGET.length().
     *
     * @param individual the individual to evaluate
     * @return fitness score between 0 and TARGET.length() inclusive
     */
    static double evaluateFitness(Individual individual) {
        double fitness = 0;
        for (int i = 0; i < TARGET.length(); i++) {
            if (individual.getChromosome()[i] == TARGET.charAt(i)) {
                fitness++;
            }
        }
        return fitness;
    }
}