package com.prgmtrouble.ml.prgmML.examples;

import java.util.concurrent.ThreadLocalRandom;

import com.prgmtrouble.ml.prgmML.fcl.FCL;
import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;
import com.prgmtrouble.ml.prgmML.math.FunctionTypes;

/**
 * An example of how an FCL may be constructed and used with
 * an objective function.
 * 
 * @author prgmTrouble
 */
public class FCLExample {
	private static final int N_EXAMPLES = 2;
	private static final int IN_SIZE = 3;
	private static final int OUT_SIZE = 2;
	private static final byte DEPTH = 2;
	private static final double LEARNING_RATE = 1;
	private static final double PRUNE = -1.0;
	private static final double LR_CONSTANT = 1.0 + 1e-2;
	private static final long ITR_COUNT = 1000000L;
	private static final FunctionTypes FUNCTION = FunctionTypes.NSoftmax_DEF;
	private static final FunctionTypes OUT_FUNCTION = FunctionTypes.CrossEntropy;
	private static final Parameter<ListOfTypes> p = new Parameter<ListOfTypes>(new ListOfTypes(new Class<?>[] {double[].class}));
	private static final Parameter<ListOfTypes> q = new Parameter<ListOfTypes>(new ListOfTypes(new Class<?>[] {double[].class,Integer.class}));
	private static final Parameter<ListOfTypes> r = new Parameter<ListOfTypes>(new ListOfTypes(new Class<?>[] {double[].class,Double.class}));
	private static final Parameter<ListOfTypes> s = new Parameter<ListOfTypes>(new ListOfTypes(new Class<?>[] {null}));
	@SuppressWarnings("unchecked")
	private static final Parameter<ListOfTypes>[] H_PARAMETERS = new Parameter[] {p,p,p,p};
	@SuppressWarnings("unchecked")
	private static final Parameter<ListOfTypes>[] O_PARAMETERS = new Parameter[] {q,s,r,p};
	
	private static double[] in;
	private static double[][] in2;
	private static int exp;
	private static int[] exp2;
	
	public static void main(String[] args) {
		init(IN_SIZE);
		final double[] lr = new double[DEPTH],
					    p = new double[DEPTH];
		FunctionTypes[] layerFunctions = new FunctionTypes[DEPTH];
		@SuppressWarnings("unchecked")
		Parameter<ListOfTypes>[][] layerParams = new Parameter[DEPTH][];
		for(byte d = 0; d < DEPTH; d++) {
			lr[d] = LEARNING_RATE / (Math.pow(LR_CONSTANT, (double) (DEPTH - d - 1)));
			p[d] = PRUNE;
			layerFunctions[d] = (d < DEPTH-1)? FUNCTION:OUT_FUNCTION;
			layerParams[d] = (d < DEPTH-1)? H_PARAMETERS:O_PARAMETERS;
		}
		final FCL fcl = new FCL(IN_SIZE, DEPTH, OUT_SIZE, layerFunctions, layerParams);
		layerFunctions = null;
		layerParams = null;
		long iteration = 0L,
			 correctCount = 0L;
		double accuracy = 0.0,
			   pacc = 0.0;
		while(true) {
			generate();
			//generateOverfit(IN_SIZE);
			fcl.setExpected(exp);
			final double[] out = fcl.forward(in);
			if((iteration % ITR_COUNT) == 0L) {
				String s = "";
				for(double o : out)
					s += String.format("%.2f",o) + " ";
				String t = "";
				for(int o = 0; o < out.length; o++)
					t += ((o != exp)? "0.00":"1.00") + " ";
				String u = String.format("%.2f", (accuracy = ((double) correctCount / (double) ITR_COUNT)) * 100.0) + "% (max: "+String.format("%.2f",pacc * 100.0)+"%)";
				correctCount = 0L;
				double diff = (double) (pacc - accuracy);
				u += " Diff: "+(diff * 100.0) + "\tLR: "+lr[0];
				System.out.println("Iteration "+iteration+"| Output:[ "+s+"] | Expected:[ "+t+"] "+(exp)+" | Accuracy: "+u);
				if(accuracy > pacc) {
					pacc = accuracy;
					for(byte i = 0; i < DEPTH; i++)
						lr[i] /= LR_CONSTANT;
				} else if(diff > 0.15) {
					for(byte i = 0; i < DEPTH; i++)
						lr[i] += diff * LR_CONSTANT;
				}
			} else if(evaluate(out))
				correctCount++;
			iteration++;
			fcl.backward(lr, p, null);
		}
	}
	
	public static void init(int size) {
		in2 = new double[N_EXAMPLES][size];
		exp2 = new int[N_EXAMPLES];
		ThreadLocalRandom r = ThreadLocalRandom.current();
		for(int i = 0; i < N_EXAMPLES; i++) {
			final double[] x = new double[size];
			for(int j = 0; j < size; j++)
				x[j] = r.nextDouble();
			in2[i] = x;
			exp2[i] = i%2;
		}
	}
	
	public static void generate() {
		ThreadLocalRandom r = ThreadLocalRandom.current();
		final int i = r.nextInt(0, N_EXAMPLES);
		in = in2[i];
		exp = exp2[i];
	}
	
	public static void generateOverfit(int size) {
		in = new double[size];
		for(int i = 0; i < size; in[i] = (i++) % 2);
		exp = 0;
	}
	
	public static boolean evaluate(double[] out) {return out[exp] > out[1 - exp];}
}


















































