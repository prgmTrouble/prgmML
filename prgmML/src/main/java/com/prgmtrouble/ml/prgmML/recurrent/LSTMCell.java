package com.prgmtrouble.ml.prgmML.recurrent;

import java.io.Serializable;
import java.util.concurrent.ThreadLocalRandom;

import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;
import com.prgmtrouble.ml.prgmML.math.FunctionTypes;
import com.prgmtrouble.ml.prgmML.math.Tensor;
import com.prgmtrouble.ml.prgmML.math.Vector;

public class LSTMCell implements Serializable {
	/***/
	private static final long serialVersionUID = 1L;
	
	private static final int nGates = 4;
	private static final int nGateParams = 3;
	
	private static final int GateActivation = 0;
	private static final int GateIn     	= 1;
	private static final int GateForget 	= 2;
	private static final int GateOut    	= 3;
	//private static final int GateState  = 3;
	private static final int paramW = 0;
	private static final int paramU = 1;
	private static final int paramB = 2;
	
	private final Vector activation;
	private final Vector gateIn;
	private final Vector gateForget;
	private final Vector gateOut;
	private final Vector gateState;
	private final Parameter<ListOfTypes> activationParams;
	private final Parameter<ListOfTypes> 	 inGateParams;
	private final Parameter<ListOfTypes> forgetGateParams;
	private final Parameter<ListOfTypes>    outGateParams;
	private final Parameter<ListOfTypes>  stateGateParams;
	private boolean forwardExecuted = false;
	private boolean initialized = false;
	
	private static final ListOfTypes fwdType = new ListOfTypes(new Class<?>[] {double[].class});
	
	private final double[][][] params;
	private double[][][] timeCache;
	private double[][] inCache;
	private double[][] state;
	private int maxT = 0;
	
	public LSTMCell(FunctionTypes activationType,
					FunctionTypes     inGateType,
					FunctionTypes forgetGateType,
					FunctionTypes    outGateType,
					FunctionTypes  stateGateType,
					Parameter<ListOfTypes> activationParams,
					Parameter<ListOfTypes> 	   inGateParams,
					Parameter<ListOfTypes> forgetGateParams,
					Parameter<ListOfTypes> 	  outGateParams,
					Parameter<ListOfTypes>  stateGateParams) {
		
		activation   = new Vector(activationType);
		gateIn 		 = new Vector(	  inGateType);
		gateForget   = new Vector(forgetGateType);
		gateOut 	 = new Vector(   outGateType);
		gateState 	 = new Vector( stateGateType);
		
		this.activationParams = activationParams;
		this.    inGateParams =     inGateParams;
		this.forgetGateParams = forgetGateParams;
		this.   outGateParams =    outGateParams;
		this. stateGateParams =  stateGateParams;
		
		params = new double[nGateParams][][];
	}
	
	public double[][] forward(double[][] in) {
		inCache = in;
		maxT = in.length;
		if(!initialized) {
			initialized = true;
			final double v = Math.sqrt(2.0 / (double) maxT);
			final ThreadLocalRandom r = ThreadLocalRandom.current();
			for(int i = 0; i < nGateParams; i++) {
				final double[][] a = new double[nGates][];
				for(int j = 0; j < nGates; j++) {
					final double[] b = new double[maxT];
					for(int k = 0; k < maxT; k++)
						b[k] = r.nextGaussian() * v;
					a[j] = b;
				}
				params[i] = a;
			}
		}
		
		final int vLength = in[0].length;
		state = new double[maxT + 1][];
		{
			final ThreadLocalRandom r = ThreadLocalRandom.current();
			final double[] s = new double[vLength];
			for(int v = 0; v < vLength; v++)
				s[v] = r.nextGaussian();
			state[0] = s;
		}
		final double[][] out = new double[maxT][],
						 W = params[paramW],
						 U = params[paramU],
						 B = params[paramB]; //ba lubba dub dub.
		timeCache = new double[maxT][][];
		for(int t = 0; t < maxT; t++) {
			final double[][] cache = new double[nGates + 1][];
			final double[] Ot = (t > 0)? out[t - 1]:new double[vLength];
			setParameter(activationParams, in[t], W[GateActivation], Ot, U[GateActivation], B[GateActivation]);
			setParameter(    inGateParams, in[t], W[GateIn],         Ot, U[GateIn],         B[GateIn]);
			final double[] a = Tensor.product(cache[0] = (double[]) activation.forward(activationParams).getValues()[0], cache[1] = (double[]) gateIn.forward(inGateParams).getValues()[0]);
			setParameter(forgetGateParams, in[t], W[GateForget],     Ot, U[GateForget], 	B[GateForget]);
			final double[] b = Tensor.product((double[]) gateForget.forward(forgetGateParams).getValues()[0], state[t]);
			state[t + 1] = Tensor.sum(Tensor.dupe(a), b);
			stateGateParams.setValue(state[t], 0);
			setParameter(   outGateParams, in[t], W[GateOut], 		 Ot, U[GateOut], 	    B[GateForget]);
			out[t] = Tensor.product(cache[2] = (double[]) gateState.forward(stateGateParams).getValues()[0], cache[3] = (double[]) gateOut.forward(outGateParams).getValues()[0]);
			cache[4] = Ot;
			timeCache[t] = cache;
		}
		return out;
	}
	
	public double[][] backward(double[][] loss, double learningRate) {
		final double[][] W = params[paramW],
						 U = params[paramU],
						 B = params[paramB],
						dX = new double[maxT][];
		final double[] dW = new double[nGates],
					   dU = new double[nGates];
		double[] dO;
		
		for(int t = maxT - 1; t >= 0; t++) {
			final double[][] cache = timeCache[t];
			final double[] A = cache[0],
						   I = cache[1],
						   S = cache[2],
						   O = cache[3],
						  Ot = cache[4],
						   l = loss[t];
			final double[][] gradIn = inGradient(W, U, A, I, S, O, state[t], activation, gateIn, gateOut, gateForget, gateState);
			dX[t] = Tensor.product(gradIn[0], l);
			dO = Tensor.product(gradIn[1], l);
			if(t > 0)
				Tensor.sum(loss[t - 1], dO); //TODO
		}
		
		return null;
	}
	
	private static void setParameter(Parameter<ListOfTypes> gateParam, double[] in, double[] W, double[] pOut, double[] U, double[] B) {
		gateParam.setValue(Tensor.sum(Tensor.sum(Tensor.product(Tensor.dupe(in), W), Tensor.product(Tensor.dupe(pOut), U)), B), 0);
	}
	
	private static double[][] inGradient(double[][] W, double[][] U, double[] A, double[] I, double[] S, double[] O, double[] state,
								   Vector gateActivation, Vector gateIn, Vector gateOut, Vector gateForget, Vector gateState) {
		final ListOfTypes T = new ListOfTypes(new Class<?>[] {double[].class});
		final double[][] out = new double[2][];
		out[0] = Tensor.sum(
					Tensor.product(
						(double[]) gateState.backward(new Parameter<ListOfTypes>(T, new Object[] {
							Tensor.sum(
								Tensor.sum(
									Tensor.product(
										(double[]) gateActivation.backward(new Parameter<ListOfTypes>(T, new Object[] {W[GateActivation]})).getValues()[0],
										I),
									Tensor.product(
										(double[]) gateIn.backward(new Parameter<ListOfTypes>(T, new Object[] {W[GateIn]})).getValues()[0],
										A)),
								Tensor.product(
									(double[]) gateForget.backward(new Parameter<ListOfTypes>(T, new Object[] {W[GateForget]})).getValues()[0],
									state))})).getValues()[0],
						O),
					Tensor.product(
						(double[]) gateOut.backward(new Parameter<ListOfTypes>(T, new Object[] {W[GateOut]})).getValues()[0],
						S));
		out[1] = Tensor.sum(
					Tensor.product(
						(double[]) gateState.backward(new Parameter<ListOfTypes>(T, new Object[] {
							Tensor.sum(
								Tensor.sum(
									Tensor.product(
										(double[]) gateActivation.backward(new Parameter<ListOfTypes>(T, new Object[] {U[GateActivation]})).getValues()[0],
										I),
									Tensor.product(
										(double[]) gateIn.backward(new Parameter<ListOfTypes>(T, new Object[] {U[GateIn]})).getValues()[0],
										A)),
								Tensor.product(
									(double[]) gateForget.backward(new Parameter<ListOfTypes>(T, new Object[] {U[GateForget]})).getValues()[0],
									state))})).getValues()[0],
						O),
					Tensor.product(
						(double[]) gateOut.backward(new Parameter<ListOfTypes>(T, new Object[] {U[GateOut]})).getValues()[0],
						S));
		return out;
	}
	
	private static void weightGradient(double[] x, double[] Ot, double[] A, double[] I, double[] S, double[] O, double[] state,
									   Vector activation, Vector gateIn, Vector gateOut, Vector gateForget, Vector gateState,
									   double[] dW, double[] dU, double[] dB) {
		
	}
}























































