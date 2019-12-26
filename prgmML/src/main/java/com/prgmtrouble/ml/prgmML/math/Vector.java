package com.prgmtrouble.ml.prgmML.math;

import java.io.Serializable;
import java.util.Arrays;

import com.prgmtrouble.ml.prgmML.generic.Function;
import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;

/**
 * An object that stores and executes operations on an array of data.
 * This object does not rely on an internal container to manage data;
 * it relies on the input to provide data and then stores it in order
 * to make the class more flexible.
 * 
 * @author prgmTrouble
 */
public class Vector implements Serializable {
	/***/
	private static final long serialVersionUID = 1L;
	
	/**The default value for the Leaky Rectified Linear Unit activation function hyperparameter.*/
	public static final double LEAKY_RELU_DEFAULT_W = 0.01;
	/**The default value for the Exponential Linear Unit activation function hyperparameter.*/
	public static final double ELU_DEFAULT_W = 1.0;
	/**The default value for the Inverse Square Root Linear Unit activation function hyperparameter.*/
	public static final double ISRLU_DEFAULT_A = 3.0;
	/**The default value for the Scaled Exponential Linear Unit activation function hyperparameter, <code>&#120746</code>.*/
	public static final double SELU_DEFAULT_A = 1.673263242354377284817042991671652038461155348281216270915;
	/**The default value for the Scaled Exponential Linear Unit activation function hyperparameter, <code>&#955</code>.*/
	public static final double SELU_DEFAULT_L = 1.050700987355480493419334985294606351958981204376404546977;
	/**The minimum value that an input to an exponential may have without the output being evaluated to zero.*/
	public static final double EXP_MIN_VALUE = Math.log(Double.MIN_VALUE);
	
	/**
	 * A blank function.
	 * 
	 * @author prgmTrouble
	 */
	private static final class Blank extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public Blank() {super(ParameterTypes.NULL.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {return null;}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {return null;}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "Blank";}
	}
	
	/**
	 * A Rectified Linear Unit activation function.
	 * 
	 * @author prgmTrouble
	 */
	private static final class ReLU extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public ReLU() {super(ParameterTypes.ReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] out = new double[il]; // O
				for(int i = 0; i < il; i++) { // O = max(I,0)
					final double t = in[i];
					if(t > 0.0)
						out[i] = t;
				}
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				final double[] in = (double[]) getForwardParameter().getValues()[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] out = new double[il]; // dL/dI
				for(int i = 0; i < il; i++)
					if(in[i] > 0.0)
						out[i] = l[i]; // dL/dI = dL/dO * dO/dI = dL/dO * [(I>0)? 1:0] = (I>0)? (dL/dO):0
				
				setGradientParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "ReLU";}
	};
	/**
	 * A Leaky Rectified Linear Unit activation function with default
	 * parameters.
	 * 
	 * @author prgmTrouble
	 */
	private static final class LeakyReLU_DEF extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public LeakyReLU_DEF() {super(ParameterTypes.ReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] out = new double[il]; // O
				System.arraycopy(in, 0, out, 0, il);
				for(int i = 0; i < il; i++)
					if(out[i] <= 0.0)
						out[i] *= LEAKY_RELU_DEFAULT_W; // O = max(I,w*I) = I * [(I>0)? 1:w]
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				final double[] in = (double[]) getForwardParameter().getValues()[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++)
					if(in[i] <= 0.0)
						out[i] *= LEAKY_RELU_DEFAULT_W; // dL/dI = dL/dO * dO/dI = dL/dO * [(I>0)? 1:w]
				
				setGradientParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "LeakyReLU_DEF";}
	};
	/**
	 * A Leaky Rectified Linear Unit activation function.
	 * 
	 * @author prgmTrouble
	 */
	private static final class LeakyReLU extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public LeakyReLU() {super(ParameterTypes.LeakyReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 2)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] w = (double[]) forwardParams[1]; // w
				if(w == null)
					error(toString()+": Null Weight.");
				if(w.length != il)
					error(toString()+": Invalid Weight.");
				
				final double[] out = new double[il]; // O
				System.arraycopy(in, 0, out, 0, il);
				for(int i = 0; i < il; i++)
					if(out[i] <= 0.0)
						out[i] *= w[i]; // O = max(I,wI) = I * [(I>0)? 1:w]
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				Object[] fwd = getForwardParameter().getValues();
				
				final double[] in = (double[]) fwd[0]; // I
				if(in == null)
					error(toString()+": Null Input");
				final int il = in.length;
				
				final double[] w = (double[]) fwd[1]; // w
				fwd = null;
				if(w == null)
					error(toString()+": Null Weight.");
				if(w.length != il)
					error(toString()+": Invalid Weight.");
				
				final double[] l = (double[]) backwardParams[0]; // O
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] dw  = new double[il]; // dL/dw
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = in[i];
					if(t <= 0.0) {
						out[i] *= w[i]; 	// dL/dI = dL/dO * dO/dI = dL/dO * [(I>0)? 1:w]
						dw[i]  = t * l[i]; //  dL/dw = dL/dO * dO/dw = dL/dO * [(I>0)? 0:I]
					}
				}
				
				setGradientParameter(new Object[] {out,dw});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {
			final double[] dw = (double[]) getGradientParameter().getValues()[1],
						    w = (double[]) getForwardParameter().getValues()[1];
			for(int i = 0; i < w.length; i++)
				w[i] -= dw[i] * learningRate;
		}
		
		@Override
		public String toString() {return "LeakyReLU";}
	};
	/**
	 * An Exponential Linear Unit activation function with default
	 * parameters.
	 * 
	 * @author prgmTrouble
	 */
	private static final class ELU_DEF extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public ELU_DEF() {super(ParameterTypes.ReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] out = new double[il]; // O
				for(int i = 0; i < il; i++) { // O = (I>0)? I:(w * ((e^I)-1))
					final double t = in[i];
					out[i] = (t > 0.0)? (t) : (ELU_DEFAULT_W * (Math.exp(t) - 1.0));
				}
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				final double[] in = (double[]) getForwardParameter().getValues()[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++)
					if(in[i] <= 0.0)
						out[i] *= pOut[i] + ELU_DEFAULT_W; // dL/dI = dL/dO * dO/dI = dL/dO * [(I>0)? 1:(O+w)]
				
				setGradientParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "ELU_DEF";}
	};
	/**
	 * An Exponential Linear Unit activation function.
	 * 
	 * @author prgmTrouble
	 */
	private static final class ELU extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public ELU() {super(ParameterTypes.LeakyReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 2)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] w = (double[]) forwardParams[1]; // w
				if(w == null)
					error(toString()+": Null Weight.");
				if(w.length != il)
					error(toString()+": Invalid Weight.");
				
				final double[] out = new double[il]; // O
				for(int i = 0; i < il; i++) {
					final double t = in[i]; // O = (I>0)? I:(w * ((e^I)-1))
					out[i] = (t > 0.0)? (t) : (w[i] * (Math.exp(t) - 1.0));
				}
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				Object[] fwd = getForwardParameter().getValues();
				
				final double[] in = (double[]) fwd[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] w = (double[]) fwd[1]; // w
				fwd = null;
				if(w == null)
					error(toString()+": Null Weight.");
				if(w.length != il)
					error(toString()+": Invalid Weight.");
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				
				final double[] dw  = new double[il]; // dL/dw
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++)
					if(in[i] <= 0.0) {
						final double t = pOut[i],
									 u = w[i];
						out[i] *= t + u;		// dL/dI = dL/dO * dO/dI = dL/dO * [(I>0)? 1:(O+w)]
						dw[i] = l[i] * t / u;  //  dL/dw = dL/dO * dO/dw = dL/dO * [(I>0)? 0:(O/w)] = (I>0)? 0:(dL/dO * O/w)
					}
				
				setGradientParameter(new Object[] {out,dw});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {
			final double[] dw = (double[]) getGradientParameter().getValues()[1],
						    w = (double[]) getForwardParameter().getValues()[1];
			for(int i = 0; i < w.length; i++)
				w[i] -= dw[i] * learningRate;
		}
		
		@Override
		public String toString() {return "ELU";}
	};
	/**
	 * An Inverse Square Root Linear Unit activation function with
	 * default parameters.
	 * 
	 * @author prgmTrouble
	 */
	private static final class ISRLU_DEF extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public ISRLU_DEF() {super(ParameterTypes.ReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] out = new double[il]; // O
				System.arraycopy(in, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = out[i];
					if(t < 0.0) // O = I / [(I<0)? sqrt(1+aI^2):1] 
						out[i] /= Math.sqrt(1.0 + ISRLU_DEFAULT_A * t * t);
				}
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				final double[] in = (double[]) getForwardParameter().getValues()[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = in[i];
					if(t < 0.0) {
						final double u = pOut[i] / t;
						out[i] *= u * u * u; // dL/dI = dL/dO * dO/dI = dL/dO * [(I<0)? ((O/I)^3):1]
					}
				}
				
				setGradientParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "ISRLU_DEF";}
	};
	/**
	 * An Inverse Square Root Linear Unit activation function.
	 * 
	 * @author prgmTrouble
	 */
	private static final class ISRLU extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public ISRLU() {super(ParameterTypes.LeakyReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 2)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] a = (double[]) forwardParams[1]; // a
				if(a == null)
					error(toString()+": Null Constant.");
				if(a.length != il)
					error(toString()+": Invalid Constant.");
				
				final double[] out = new double[il]; // O
				System.arraycopy(in, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = out[i];
					if(t < 0.0) // O = I / [(I<0)? sqrt(1+aI^2):1]
						out[i] /= Math.sqrt(1.0 + a[i] * t * t);
				}
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				Object[] fwd = getForwardParameter().getValues();
				
				final double[] in = (double[]) fwd[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] a = (double[]) fwd[1]; // a
				fwd = null;
				if(a == null)
					error(toString()+": Null Constant.");
				if(a.length != il)
					error(toString()+": Invalid Constant.");
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				final double[] da  = new double[il]; // dL/da
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = in[i];
					if(t < 0.0) {
						double u = pOut[i]; // dL/da = dL/dO * dO/da = dL/dO * [(I<0)? (-(O^3)/2):0] = (I<0)? (-dL/dO * (O^3)/2):0
						da[i] = l[i] * u * u * u / (-2.0);
						u /= t;
						out[i] *= u * u * u; // dL/dI = dL/dO * dO/dI = dL/dO * [(I<0)? ((O/I)^3):1]
					}
				}
				
				setGradientParameter(new Object[] {out,da});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {
			final double[] da = (double[]) getGradientParameter().getValues()[1],
						    a = (double[]) getForwardParameter().getValues()[1];
			for(int i = 0; i < a.length; i++)
				a[i] -= da[i] * learningRate;
		}
		
		@Override
		public String toString() {return "ISRLU";}
	};
	/**
	 * A Scaled Exponential Linear Unit activation function with
	 * default parameters.
	 * 
	 * @author prgmTrouble
	 */
	private static final class SELU_DEF extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public SELU_DEF() {super(ParameterTypes.ReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] out = new double[il]; // O
				Arrays.fill(out, SELU_DEFAULT_L);
				for(int i = 0; i < il; i++) {
					final double t = in[i]; // O = l * [(I>0)? I:(a * ((e^I)-1))]
					out[i] *= (t > 0.0)? (t) : (SELU_DEFAULT_A * (Math.exp(t) - 1.0));
				}
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				final double[] in = (double[]) getForwardParameter().getValues()[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) // dL/dI = dL/dO * dO/dI = dL/dO * [(I>0)? l:(O+l*a)]
					out[i] *= (in[i] > 0.0)? (SELU_DEFAULT_L) : (pOut[i] + SELU_DEFAULT_L * SELU_DEFAULT_A);
				
				setGradientParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "SELU_DEF";}
	};
	/**
	 * A Scaled Exponential Linear Unit activation function.
	 * 
	 * @author prgmTrouble
	 */
	private static final class SELU extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public SELU() {super(ParameterTypes.SELU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 3)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] A = (double[]) forwardParams[1]; // a
				if(A == null)
					error(toString()+": Null Constant (a).");
				if(A.length != il)
					error(toString()+": Invalid Constant (a).");
				
				final double[] L = (double[]) forwardParams[2]; // l
				if(L == null)
					error(toString()+": Null Constant (l).");
				if(L.length != il)
					error(toString()+": Invalid Constant (l).");
				
				final double[] out = new double[il]; // O
				System.arraycopy(L, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = in[i]; // O = l * [(I>0)? I:(a*((e^I)-1))]
					out[i] *= (t > 0.0)? (t) : (A[i] * (Math.exp(t) - 1.0));
				}
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				Object[] fwd = getForwardParameter().getValues();
				
				final double[] in = (double[]) fwd[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] A = (double[]) fwd[1]; // a
				if(A == null)
					error(toString()+": Null Constant (a).");
				final double[] L = (double[]) fwd[2]; // l
				fwd = null;
				if(L == null)
					error(toString()+": Null Constant (l).");
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				final double[] dl  = new double[il]; // dL/dl
				System.arraycopy(l, 0, dl, 0, il);
				final double[] da  = new double[il]; // dL/da
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = pOut[i],
								Li = L[i],
								Ai = A[i];
					dl[i] *= t / Li; // dL/dl = dL/dO * dO/dl = dL/dO * [(I>0)? I:(a*((e^I)-1))] = dL/dO * O/l
					if(in[i] > 0.0) {
						out[i] *= Li; // dL/dI = dL/dO * dO/dI = dL/dO * [(I>0)? l:(O+l*a)]
						da[i] = l[i] * t / Ai; // dL/da = dL/dO * dO/da = dL/dO * [(I>0)? (O/a):0] = (I>0)? (dL/dO * O/a):0
					} else out[i] *= t + Li * Ai;
				}
				
				setGradientParameter(new Object[] {out,da,dl});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {
			Object[] grad = getGradientParameter().getValues(),
					  fwd = getForwardParameter().getValues();
			double[] da = (double[]) grad[1],
					  a = (double[]) fwd[1];
			for(int i = 0; i < a.length; i++)
				a[i] -= da[i] * learningRate;
			a = da = null;
			
			final double[] dl = (double[]) grad[2],
							l = (double[])  fwd[2];
			grad = fwd = null;
			
			for(int i = 0; i < l.length; i++)
				l[i] -= dl[i] * learningRate;
		}
		
		@Override
		public String toString() {return "SELU";}
	};
	/**
	 * A Sigmoid Linear Unit activation function with default
	 * parameters.
	 * 
	 * @author prgmTrouble
	 */
	private static final class SiLU_DEF extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public SiLU_DEF() {super(ParameterTypes.SiLU_DEF.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] out = new double[il]; // O
				System.arraycopy(in, 0, out, 0, il);
				final double[] sOut = new double[il]; // Os
				Arrays.fill(sOut, 1.0);
				for(int i = 0; i < il; i++) {
					final double t = 1.0 + Math.exp(-in[i]);
					sOut[i] /= t; // Os = 1 / (1 + e^(-I))
					out[i] /= t; // O = I * Os
				}
				
				setOutputParameter(new Object[] {out,sOut});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				final double[] in = (double[]) getForwardParameter().getValues()[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				Object[] bkwd = getOutputParameter().getValues();
				
				final double[] pOut = (double[]) bkwd[0]; // O
				final double[] s    = (double[]) bkwd[1]; // Os
				bkwd = null;
				if(s == null)
					error(toString()+": Null Sigmoid.");
				
				final double[] out  = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = s[i]; // dL/dI = dL/dO * dO/dI = dL/dO * (Os + O * (1 - Os))
					out[i] *= t + pOut[i] * (1.0 - t);
				}
				
				setGradientParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "SiLU_DEF";}
	};
	/**
	 * A Sigmoid Linear Unit activation function.
	 * 
	 * @author prgmTrouble
	 */
	private static final class SiLU extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public SiLU() {super(ParameterTypes.SiLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 2)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] b = (double[]) forwardParams[1]; // b
				if(b == null)
					error(toString()+": Null Base.");
				if(b.length != il)
					error(toString()+": Invalid Base.");
				
				final double[] sOut = new double[il]; // Os
				Arrays.fill(sOut, 1.0);
				final double[] out = new double[il]; // O
				System.arraycopy(in, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = 1.0 + Math.pow(b[i], -in[i]);
					sOut[i] /= t; // Os = 1 / (1 + b^(-I))
					out[i] /= t; // O = I * Os
				}
				
				setOutputParameter(new Object[] {out,sOut});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				Object[] fwd = getForwardParameter().getValues();
				
				final double[] in = (double[]) fwd[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] b = (double[]) fwd[1]; // b
				fwd = null;
				if(b == null)
					error(toString()+": Null Base.");
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				Object[] bkwd = getOutputParameter().getValues();
				final double[] pOut = (double[]) bkwd[0]; // O
				final double[] s    = (double[]) bkwd[1]; // Os
				bkwd = null;
				if(s == null)
					error(toString()+": Null Sigmoid.");
				
				final double[] db   = new double[il]; // dL/db
				System.arraycopy(l, 0, db, 0, il);
				final double[] out  = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = s[i],
								 u = b[i],
								 v = pOut[i];
					out[i] *= t + v * (1.0 - t) * Math.log(u); // dL/dI = dL/dO * dO/dI = dL/dO * (Os + O * (1 - Os) * ln(b))
					db[i] *= v * v / Math.pow(u, in[i] + 1);  //  dL/db = dL/dO * dO/db = dL/dO * (O^2) / (b^(I+1))
				}
				
				setGradientParameter(new Object[] {out,db});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {
			final double[] db = (double[]) getGradientParameter().getValues()[1],
						    b = (double[]) getForwardParameter().getValues()[1];
			for(int i = 0; i < b.length; i++)
				b[i] -= db[i] * learningRate;
		}
		
		@Override
		public String toString() {return "SiLU";}
	};
	/**
	 * A Sigmoid activation function with default parameters.
	 * 
	 * @author prgmTrouble
	 */
	private static final class Sigmoid_DEF extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public Sigmoid_DEF() {super(ParameterTypes.ReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] out = new double[il]; // O
				Arrays.fill(out, 1.0);
				for(int i = 0; i < il; i++)
					out[i] /= 1.0 + Math.exp(-in[i]); // O = 1 / (1 + e^(-I))
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				final double[] in = (double[]) getForwardParameter().getValues()[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = pOut[i];
					out[i] *= t * (1.0 - t); // dL/dI = dL/dO * dO/dI = dL/dO * O * (1-O)
				}
				
				setGradientParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "Sigmoid_DEF";}
	};
	/**
	 * A Sigmoid activation function.
	 * 
	 * @author prgmTrouble
	 */
	private static final class Sigmoid extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public Sigmoid() {super(ParameterTypes.LeakyReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 2)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] b = (double[]) forwardParams[1]; // b
				if(b == null)
					error(toString()+": Null Base.");
				if(b.length != il)
					error(toString()+": Invalid Base.");
				
				final double[] out = new double[il]; // O
				Arrays.fill(out, 1.0);
				for(int i = 0; i < il; i++)
					out[i] /= 1.0 + Math.pow(b[i], -in[i]); // O = 1 / (1 + b^(-I))
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				Object[] fwd = getForwardParameter().getValues();
				
				final double[] in = (double[]) fwd[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] b = (double[]) fwd[1]; // b
				fwd = null;
				if(b == null)
					error(toString()+": Null Base.");
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				final double[] db  = new double[il]; // dL/db
				System.arraycopy(l, 0, db, 0, il);
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = pOut[i],
								 u = b[i];
					out[i] *= t * (1.0 - t) * Math.log(u); // dL/dI = dL/dO * dO/dI = dL/dO * O * (1-O) * ln(b)
					final double v = in[i];
					db[i] *= t * t * v / Math.pow(u, v + 1); // dL/db = dL/dO * dO/db = dL/dO * (O^2) * I / (b^(I+1))
				}
				
				setGradientParameter(new Object[] {out,db});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {
			final double[] db = (double[]) getGradientParameter().getValues()[1],
						    b = (double[]) getForwardParameter().getValues()[1];
			for(int i = 0; i < b.length; i++)
				b[i] -= db[i] * learningRate;
		}
		
		@Override
		public String toString() {return "Sigmoid";}
	};
	/**
	 * A Hyperbolic Tangent activation function.
	 * 
	 * @author prgmTrouble
	 */
	private static final class TanH extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public TanH() {super(ParameterTypes.ReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] out = new double[il]; // O
				for(int i = 0; i < il; i++)
					out[i] = Math.tanh(in[i]); // O = ((e^I)-(e^(-I)))/((e^I)+(e^(-I)))
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				double[] in = (double[]) getForwardParameter().getValues()[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				in = null;
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = pOut[i];
					out[i] *= 1.0 - t * t; // dL/dI = dL/dO * dO/dI = dL/dO * (1-(O^2))
				}
				
				setGradientParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "TanH";}
	};
	/**
	 * An Inverse Tangent activation function.
	 * 
	 * @author prgmTrouble
	 */
	private static final class ArcTan extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public ArcTan() {super(ParameterTypes.ReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] out = new double[il]; // O
				for(int i = 0; i < il; i++)
					out[i] = Math.atan(in[i]); // O = atan(I)
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				final double[] in = (double[]) getForwardParameter().getValues()[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = in[i];
					out[i] /= t * t + 1.0; // dL/dI = dL/dO * dO/dI = dL/dO / (1 + I^2)
				}
				
				setGradientParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "ArcTan";}
	};
	/**
	 * An Inverse Hyperbolic Sine activation function.
	 * 
	 * @author prgmTrouble
	 */
	private static final class ArcSinH extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public ArcSinH() {super(ParameterTypes.ReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] out = new double[il]; // O
				for(int i = 0; i < il; i++) {
					final double t = in[i];
					out[i] = Math.log(t + Math.sqrt(t * t + 1.0)); // O = ln(I + sqrt(1+I^2))
				}
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				final double[] in = (double[]) getForwardParameter().getValues()[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = in[i]; // dL/dI = dL/dO * dO/dI = dL/dO / sqrt(1+I^2)
					out[i] /= Math.sqrt(t * t + 1.0);
				}
				
				setGradientParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "ArcSinH";}
	};
	/**
	 * A Softmax activation function with default parameters.
	 * 
	 * @author prgmTrouble
	 */
	private static final class Softmax_DEF extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public Softmax_DEF() {super(ParameterTypes.ReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] out = new double[il]; // O
				double eSum = 0.0;
				for(int i = 0; i < il; i++)
					eSum += out[i] = Math.exp(in[i]);
				
				for(int i = 0; i < il; i++)
					out[i] /= eSum; // Oi = (e^(Ii)) / sum(e^I)
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				if(pOut == null)
					error(toString()+": Null Output.");
				final int il = pOut.length;
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = pOut[i];
					out[i] *= t * (1.0 - t); // dL/dI = dL/dO * dO/dI = dL/dO * O * (1-O)
				}
				
				setGradientParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "Softmax_DEF";}
	};
	/**
	 * A Softmax activation function.
	 * 
	 * @author prgmTrouble
	 */
	private static final class Softmax extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public Softmax() {super(ParameterTypes.LeakyReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 2)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] b = (double[]) forwardParams[1]; // b
				if(b == null)
					error(toString()+": Null Base.");
				if(b.length != il)
					error(toString()+": Invalid Base.");
				
				final double[] out = new double[il]; // O
				double eSum = 0.0;
				for(int i = 0; i < il; i++)
					eSum += out[i] = Math.pow(b[i], in[i]);
				
				for(int i = 0; i < il; i++)
					out[i] /= eSum; // Oi = (b^(Ii)) / sum(b^I)
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				Object[] fwd = getForwardParameter().getValues();
				
				final double[] in = (double[]) fwd[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] b = (double[]) fwd[1]; // b
				fwd = null;
				if(b == null)
					error(toString()+": Null Base.");
				if(b.length != il)
					error(toString()+": Invalid Base.");
				
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				if(pOut == null)
					error(toString()+": Null Output.");
				if(pOut.length != il)
					error(toString()+": Invalid Output.");
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] db  = new double[il]; // dL/db
				System.arraycopy(l, 0, db, 0, il);
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = pOut[i],
								 u = t * (1.0 - t),
								 v = b[i];
					out[i] *= Math.log(v) * u; // dL/dI = dL/dO * dO/dI = dL/dO * O * (1-O) * ln(b)
					db[i] *= in[i] * t / v; // dL/db = dL/dO * dO/db = dL/dO * I * O / b
				}
				
				setGradientParameter(new Object[] {out,db});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {
			final double[] db = (double[]) getGradientParameter().getValues()[1],
						    b = (double[]) getForwardParameter().getValues()[1];
			for(int i = 0; i < b.length; i++)
				b[i] -= db[i] * learningRate;
		}
		
		@Override
		public String toString() {return "Softmax";}
	};
	/**
	 * A Normalized Softmax activation function with default
	 * parameters.
	 * 
	 * @author prgmTrouble
	 */
	private static final class NSoftmax_DEF extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public NSoftmax_DEF() {super(ParameterTypes.ReLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				double max = in[0];
				for(double i : in)
					if(i > max)
						max = i;
				
				final double[] out = new double[il]; // O
				double eSum = 0.0;
				for(int i = 0; i < il; i++)
					eSum += out[i] = Math.exp(Math.max(in[i] - max, EXP_MIN_VALUE));
				
				for(int i = 0; i < il; i++)
					out[i] /= eSum; // Oi = (e^(Ii-max(I)) / sum(e^(I-max(I)))
				
				setOutputParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				if(pOut == null)
					error(toString()+": Null Output.");
				final int il = pOut.length;
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = pOut[i];
					out[i] *= t * (1.0 - t); // dL/dI = dL/dO * dO/dI = dL/dO * O * (1-O)
				}
				
				setGradientParameter(new Object[] {out});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "NSoftmax_DEF";}
	};
	/**
	 * A Normalized Softmax activation function.
	 * @author prgmTrouble
	 */
	private static final class NSoftmax extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return false;}
		
		public NSoftmax() {super(ParameterTypes.SiLU.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 2)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				final double[] b = (double[]) forwardParams[1]; // b
				if(b == null)
					error(toString()+": Null Base.");
				if(b.length != il)
					error(toString()+": Invalid Base.");
				
				double max = in[0];
				for(double i : in)
					if(i > max)
						max = i;
				
				final double[] nin = new double[il]; // In
				final double[] out = new double[il]; // O
				double eSum = 0.0;
				for(int i = 0; i < il; i++)
					eSum += out[i] = Math.pow(b[i], nin[i] = Math.max(in[i] - max, EXP_MIN_VALUE)); // In = I - max(I)
				
				for(int i = 0; i < il; i++)
					out[i] /= eSum; // Oi = (b^(Ini)) / sum(b^(In))
				
				setOutputParameter(new Object[] {out,nin});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams == null)
				error(toString()+": Null Parameters.");
			if(backwardParams.length != 1)
				error(toString()+": Invalid Parameters.");
			setBackwardParameter(backwardParams);
			try {
				Object[] o = getOutputParameter().getValues();
				
				final double[] pOut = (double[]) o[0]; // O
				if(pOut == null)
					error(toString()+": Null Output.");
				final int il = pOut.length;
				
				final double[] in = (double[]) o[1]; // In
				o = null;
				if(in == null)
					error(toString()+": Null Input.");
				if(in.length != il)
					error(toString()+": Invalid Input.");
				
				final double[] b = (double[]) getForwardParameter().getValues()[1]; // b
				if(b == null)
					error(toString()+": Null Base.");
				if(b.length != il)
					error(toString()+": Invalid Base.");
				
				final double[] l = (double[]) backwardParams[0]; // dL/dO
				if(l == null)
					error(toString()+": Null Loss.");
				if(l.length != il)
					error(toString()+": Invalid Loss.");
				
				final double[] db = new double[il]; // dL/db
				System.arraycopy(l, 0, db, 0, il);
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = pOut[i],
								 u = t * (1.0 - t),
								 v = b[i];
					out[i] *= u * Math.log(v); // dL/dI = dL/dO * dO/dI = dL/dO * O * (1-O) * ln(b)
					db[i] *= in[i] * u / v; // dL/db = dL/dO * dO/db = dL/dO * In * O * (1-O) / b
				}
				
				setGradientParameter(new Object[] {out,db});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {
			final double[] db = (double[]) getGradientParameter().getValues()[1],
						    b = (double[]) getForwardParameter().getValues()[1];
			for(int i = 0; i < b.length; i++)
				b[i] -= db[i] * learningRate;
		}
		
		@Override
		public String toString() {return "NSoftmax";}
	};
	/**
	 * A Normalized Softmax activation function with Cross
	 * Entropy Loss.
	 * 
	 * @author prgmTrouble
	 */
	private static final class CrossEntropy extends Function<ListOfTypes> {
		/***/
		private static final long serialVersionUID = 1L;
		
		@Override
		public boolean isOutputFunction() {return true;}
		
		public CrossEntropy() {super(ParameterTypes.CrossEntropy.getTypes());}
		
		@Override
		public Parameter<ListOfTypes> forward(Object[] forwardParams) {
			if(forwardParams == null)
				error(toString()+": Null Parameters.");
			if(forwardParams.length != 2)
				error(toString()+": Invalid Parameters.");
			setForwardParameter(forwardParams);
			try {
				final double[] in = (double[]) forwardParams[0]; // I
				if(in == null)
					error(toString()+": Null Input.");
				final int il = in.length;
				
				double max = in[0];
				for(double i : in)
					if(i > max)
						max = i;
				//max /= (double) il;
				
				final double[] out = new double[il]; // O
				double eSum = 0.0;
				for(int i = 0; i < il; i++)
					eSum += out[i] = Math.exp(Math.max(in[i] - max, EXP_MIN_VALUE)); // In = I - max(I)
				
				for(int i = 0; i < il; i++)
					out[i] /= eSum; // Oi = (b^(Ini)) / sum(b^(In))
				
				setOutputParameter(new Object[] {out,-Math.log(out[(int) forwardParams[1]])});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getOutputParameter();
		}
		
		@Override
		public Parameter<ListOfTypes> backward(Object[] backwardParams) {
			if(backwardParams != null)
				error(toString()+": Unexpected Parameters.");
			try {
				final double[] pOut = (double[]) getOutputParameter().getValues()[0]; // O
				if(pOut == null)
					error(toString()+": Null Output.");
				final int il = pOut.length;
				
				final double[] l = new double[il]; // dL/dO
				System.arraycopy(pOut, 0, l, 0, il);
				l[(int) getForwardParameter().getValues()[1]] -= 1.0; // dL/dO = O - (i=y)? 1:0 
				
				final double[] out = new double[il]; // dL/dI
				System.arraycopy(l, 0, out, 0, il);
				for(int i = 0; i < il; i++) {
					final double t = pOut[i];
					out[i] *= t * (1.0 - t); // dL/dI = dL/dO * dO/dI = dL/dO * O * (1-O)
				}
				
				setGradientParameter(new Object[] {l});
			} catch(ClassCastException e) {
				e.printStackTrace();
				error(toString()+": Could not cast parameters.");
			}
			return getGradientParameter();
		}
		
		@Override
		public void learnParameters(double learningRate) {}
		
		@Override
		public String toString() {return "CrossEntropy";}
	};
	
	/**Function which operates on all elements of an input.*/
	private final Function<ListOfTypes> f;
	
	/**True if the function for this vector is blank.*/
	private final boolean isBlank;
	
	/**True if the function for this vector calculates its own output gradient.*/
	private final boolean isOutput;
	
	/**True if <code>forward</code> has been called.*/
	private boolean forwardExecuted;
	
	/**
	 * Creates a new Vector.
	 * 
	 * @param function Type of function. See {@linkplain FunctionTypes} for
	 * 				   details about parameters.
	 */
	public Vector(FunctionTypes function) {
		forwardExecuted = false;
		switch(function) {
		case ReLU		  : f = new ReLU()		   ; break;
		case LeakyReLU_DEF: f = new LeakyReLU_DEF(); break;
		case LeakyReLU	  : f = new LeakyReLU()	   ; break;
		case ELU_DEF	  : f = new ELU_DEF()	   ; break;
		case ELU		  : f = new ELU()		   ; break;
		case ISRLU_DEF	  : f = new ISRLU_DEF()	   ; break;
		case ISRLU		  : f = new ISRLU()		   ; break;
		case SELU_DEF	  : f = new SELU_DEF()	   ; break;
		case SELU		  : f = new SELU()		   ; break;
		case SiLU_DEF	  : f = new SiLU_DEF()	   ; break;
		case SiLU		  : f = new SiLU()		   ; break;
		case Sigmoid_DEF  : f = new Sigmoid_DEF()  ; break;
		case Sigmoid	  : f = new Sigmoid()	   ; break;
		case TanH		  : f = new TanH()		   ; break;
		case ArcTan		  : f = new ArcTan()	   ; break;
		case ArcSinH	  : f = new ArcSinH()	   ; break;
		case Softmax_DEF  : f = new Softmax_DEF()  ; break;
		case Softmax	  : f = new Softmax()	   ; break;
		case NSoftmax_DEF : f = new NSoftmax_DEF() ; break;
		case NSoftmax	  : f = new NSoftmax()	   ; break;
		case CrossEntropy : f = new CrossEntropy() ; break;
		default			  : f = new Blank()		   ; break;
		}
		isBlank = (function == FunctionTypes.Blank);
		isOutput = f.isOutputFunction();
	}
	
	/**
	 * Runs the function with the given input parameters.
	 * 
	 * @param input See {@linkplain FunctionTypes} for
	 * 				details about parameter requirements.
	 * @return The output of the function.
	 */
	public Parameter<ListOfTypes> forward(Parameter<ListOfTypes> input) {
		forwardExecuted = true;
		if(isBlank)
			return input;
		else
			return f.forward(input.getValues());
	}
	
	/**
	 * Computes the gradient of the function with the given loss with
	 * respect to the function's input.
	 * 
	 * @param loss See {@linkplain FunctionTypes} for
	 * 			   details about parameter requirements.
	 * @return The gradient of the function with respect to
	 * 		   the inputs.
	 */
	public Parameter<ListOfTypes> backward(Parameter<ListOfTypes> loss) {
		if(!forwardExecuted)
			error("Forward must be run before backward.");
		forwardExecuted = false;
		if(isBlank)
			return loss;
		else
			return f.backward(isOutput? null:loss.getValues());
	}
	
	/**
	 * Updates the hyperparameters (if any) according to the gradient
	 * calculated during backpropagation.
	 * 
	 * @param learningRate Learning rate.
	 */
	public void learnParameters(double learningRate) {f.learnParameters(learningRate);}
	
	/**@return True iff the function for this vector calculates its own output gradient.*/
	public boolean isOutputVector() {return isOutput;}
	
	public ListOfTypes[] getFunctionTypes() {return f.getTypes();}
	
	/**
	 * A custom exception which indicates an error in the vector.
	 * 
	 * @author prgmTrouble
	 */
	private static class VectorException extends Exception {
		/***/
		private static final long serialVersionUID = 1L;
		private static final String prefix = "Vector Exception: ";
		
		public VectorException(String s) {super(prefix + s);}
	}
	
	/**
	 * Throws a {@linkplain VectorException} and terminates execution.
	 * 
	 * @param s Description of error.
	 */
	private static void error(String s) {
		try {
			throw new VectorException(s);
		} catch(VectorException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}


















































