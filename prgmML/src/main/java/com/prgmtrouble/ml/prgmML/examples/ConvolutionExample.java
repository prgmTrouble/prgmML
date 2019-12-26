package com.prgmtrouble.ml.prgmML.examples;

import java.util.concurrent.ThreadLocalRandom;

import com.prgmtrouble.ml.prgmML.convolution.Convolution;
import com.prgmtrouble.ml.prgmML.convolution.Convolution.LayerTypes;
import com.prgmtrouble.ml.prgmML.convolution.Pool.PoolingTypes;
import com.prgmtrouble.ml.prgmML.fcl.FCL;
import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;
import com.prgmtrouble.ml.prgmML.math.FunctionTypes;

public class ConvolutionExample {
	
	private static final int[] layerSizes = new int[] {10,10,2};
	private static final LayerTypes[] layerTypes = new LayerTypes[] {LayerTypes.Activation,LayerTypes.Pool};
	private static final FunctionTypes[] activationTypes = new FunctionTypes[] {FunctionTypes.ReLU};
	private static final int[][] activationDimensions = new int[][] {{4,2,3}};
	private static final PoolingTypes[] poolTypes = new PoolingTypes[] {PoolingTypes.Max};
	private static final int[] poolingFactors = new int[] {5};
	private static final int inChannels = 3;
	private static final int fclInSize = 3*3*2*2;
	private static final byte fclDepth = 1;
	private static final int fclOutSize = 2;
	private static final FunctionTypes[] fclFunctions = new FunctionTypes[] {FunctionTypes.CrossEntropy};
	private static final double learningRate = 0.1;
	private static final double[] lr = new double[] {learningRate};
	private static final double[] prune = new double[] {-1.0};
	private static final boolean[] learnParameters = new boolean[] {false};
	
	private static final Parameter<ListOfTypes> p = new Parameter<ListOfTypes>(new ListOfTypes(new Class<?>[] {double[].class}));
	private static final Parameter<ListOfTypes> q = new Parameter<ListOfTypes>(new ListOfTypes(new Class<?>[] {double[].class,Integer.class}));
	private static final Parameter<ListOfTypes> r = new Parameter<ListOfTypes>(new ListOfTypes(new Class<?>[] {double[].class,Double.class}));
	private static final Parameter<ListOfTypes> s = new Parameter<ListOfTypes>(new ListOfTypes(new Class<?>[] {null}));
	
	@SuppressWarnings("unchecked")
	private static final Parameter<ListOfTypes>[][] O_PARAMETERS = new Parameter[][] {{q,s,r,p}};
	private static final long ITR_COUNT = 1;
	
	private static final int inLength = layerSizes[0] * layerSizes[0] * inChannels;
	
	private static double[] in = new double[inLength];
	private static int exp = 0;
	
	public static void main(String[] args) {
		final Convolution conv = new Convolution(layerSizes, layerTypes, activationTypes, activationDimensions, poolTypes, poolingFactors, inChannels, learningRate);
		final FCL fcl = new FCL(fclInSize, fclDepth, fclOutSize, fclFunctions, O_PARAMETERS);
		generate();
		final Parameter<ListOfTypes> inParam = new Parameter<ListOfTypes>(new ListOfTypes(new Class<?>[] {double[].class}),new Object[] {in}),
									 backParam = new Parameter<ListOfTypes>(new ListOfTypes(new Class<?>[] {double[].class}));
		long itr = 0;
		while(true) {
			final double[] correct = new double[fclOutSize];
			correct[exp] = 1;
			fcl.setExpected(exp);
			final double[] out = fcl.forward((double[]) conv.forward(inParam).getValues()[0]);
			if(itr % ITR_COUNT == 0) {
				System.out.print("Iteration " + itr + "| ");
				String s = "[ ";
				for(double o : out)
					s += String.format("%.2f",o) + " ";
				s += "]";
				System.out.print("Out: " + s + "| ");
				s = "[ ";
				for(double o : correct)
					s += String.format("%.2f",o) + " ";
				s += "]";
				System.out.print("Exp: " + s + "| \n");
			}
			backParam.setValue(fcl.backward(lr, prune, learnParameters), 0);
			conv.backward(backParam);
			itr++;
			if(itr == 100)
				System.exit(0);
		}
	}
	
	private static void generate() {
		final ThreadLocalRandom r = ThreadLocalRandom.current();
		for(int i = 0; i < inLength; i++)
			in[i] = r.nextDouble();
		exp = r.nextInt(0, fclOutSize);
	}
	
}











































