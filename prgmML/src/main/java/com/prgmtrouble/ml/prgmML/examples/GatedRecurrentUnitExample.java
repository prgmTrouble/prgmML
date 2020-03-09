package com.prgmtrouble.ml.prgmML.examples;

import java.util.concurrent.ThreadLocalRandom;

import com.prgmtrouble.ml.prgmML.fcl.FCL;
import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;
import com.prgmtrouble.ml.prgmML.math.FunctionTypes;
import com.prgmtrouble.ml.prgmML.math.ParameterTypes;
import com.prgmtrouble.ml.prgmML.math.Tensor;
import com.prgmtrouble.ml.prgmML.recurrent.GatedRecurrentUnit;

public class GatedRecurrentUnitExample {
	
	private static final int DIM1 = 3;
	private static final int DIM2 = 2;
	private static final int MAX_T = 1;
	private static final int OUT_SIZE = 3;
	private static final int FCL_DEPTH = 1;
	private static final int EXP_CLASS = 0;
	private static final long PRINT = 1000000L;
	private static final double LR = 0.01;
	private static final double FCLLR = 1;
	private static double[][] IN,OUT,LOSS;
	private static double[] FCL_IN, FCL_LOSS;
	private static final boolean fclLearnParams = false;
	private static final double fclPrune = -1.0;
	
	private static final FunctionTypes hiddenType = FunctionTypes.ReLU;
	private static final FunctionTypes outputType = FunctionTypes.CrossEntropy;
	private static final ListOfTypes[] hiddenParamTypes = ParameterTypes.ReLU.getTypes();
	private static final ListOfTypes[] outputParamTypes = ParameterTypes.CrossEntropy.getTypes();
	
	@SuppressWarnings({ "unchecked", "unused" })
	public static void main(String[] args) {
		generate();
		GatedRecurrentUnit gru = new GatedRecurrentUnit(DIM1, DIM2, MAX_T);
		
		final FunctionTypes[] types = new FunctionTypes[FCL_DEPTH];
		final Parameter<ListOfTypes>[][] params = new Parameter[FCL_DEPTH][];
		final double[] fclLR = new double[FCL_DEPTH],
					   prune = new double[FCL_DEPTH];
		final boolean[] learnParams = new boolean[FCL_DEPTH];
		if(FCL_DEPTH > 1)
			for(int i = 0; i < FCL_DEPTH - 1; i++) {
				types[i] = hiddenType;
				params[i] = new Parameter[] {new Parameter<>(hiddenParamTypes[0]), new Parameter<>(hiddenParamTypes[1])};
				fclLR[i] = FCLLR;
				prune[i] = fclPrune;
				learnParams[i] = fclLearnParams;
			}
		final int lastIDX = FCL_DEPTH - 1;
		types[lastIDX] = outputType;
		params[lastIDX] = new Parameter[] {new Parameter<>(outputParamTypes[0]), new Parameter<>(outputParamTypes[1])};
		fclLR[lastIDX] = FCLLR;
		prune[lastIDX] = fclPrune;
		learnParams[lastIDX] = fclLearnParams;
		final FCL fcl = new FCL(DIM1 * DIM2 * MAX_T, (byte) FCL_DEPTH, OUT_SIZE, types, params);
		fcl.setExpected(EXP_CLASS);
		
		long iteration = 0;
		while(true) {
			screenNaN(OUT = gru.forward(IN));
			flatten();
			final double[] fclOut;
			screenNaN(fclOut = fcl.forward(FCL_IN));
			screenNaN(FCL_LOSS = fcl.backwardNoLearning(fclLR, prune, learnParams));
			unflatten();
			gru.backward(LOSS, LR); 
			if(iteration % PRINT == 0) {
				System.out.print("itr:" + iteration + "\t|FCL Out:");
				Tensor.print(fclOut);
				System.out.print("\t|FCL Loss:");
				Tensor.print(FCL_LOSS);
				for(int i = 0; i < OUT.length; i++) {
					System.out.print("\t|t="+i+":");
					Tensor.print(OUT[i]);
				}
				System.out.println();
			}
			iteration++;
		}
	}
	
	
	
	private static void generate() {
		LOSS = new double[MAX_T][DIM1 * DIM2];
		FCL_IN = new double[MAX_T * DIM1 * DIM2];
		FCL_LOSS = new double[MAX_T * DIM1 * DIM2];
		IN = new double[MAX_T][];
		final ThreadLocalRandom r = ThreadLocalRandom.current();
		for(int t = 0; t < MAX_T; t++) {
			final double[] T = new double[DIM1 * DIM2],
						   U = new double[DIM1 * DIM2];
			for(int i = 0; i < DIM1 * DIM2; i++) {
				T[i] = r.nextGaussian();
				U[i] = r.nextGaussian();
			}
			IN[t] = T;
		}
	}
	
	private static void flatten() {
		for(int t = 1; t <= MAX_T; t++) {
			final double[] in = OUT[t];
			final int tidx = (t - 1) * DIM1 * DIM2;
			for(int i = 0; i < DIM1 * DIM2; i++)
				FCL_IN[tidx + i] = in[i];
		}
	}
	
	private static void unflatten() {
		for(int t = 0; t < MAX_T; t++) {
			final double[] l = new double[DIM1 * DIM2];
			final int tidx = t * DIM1 * DIM2;
			for(int i = 0; i < DIM1 * DIM2; i++)
				l[i] = FCL_LOSS[tidx + i];
			LOSS[t] = l;
		}
	}
	
	private static void screenNaN(double[] in) {
		for(double i : in)
			if(Double.isNaN(i) || Double.isInfinite(i))
				System.out.print("NaN Detected");
	}
	
	private static void screenNaN(double[][] in) {
		for(double[] i : in)
			screenNaN(i);
	}
}















































