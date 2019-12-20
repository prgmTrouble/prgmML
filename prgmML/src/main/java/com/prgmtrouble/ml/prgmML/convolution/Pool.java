package com.prgmtrouble.ml.prgmML.convolution;

import java.io.Serializable;

import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;

/**
 * An object that pools a flattened tensor and distributes
 * the gradient appropriately during the backward pass.
 * 
 * @author prgmTrouble
 */
public class Pool implements Serializable,ConvolutionLayer {
	/***/
	private static final long serialVersionUID = 1L;

	/**
	 * Types of pooling operations.
	 * 
	 * @author prgmTrouble
	 */
	public static enum PoolingTypes {
		/**Maximum of section.*/
		Max,
		/**Absolute maximum of section.*/
		MaxAbs,
		/**Average of section.*/
		Avg,
		/**Maximum of channel.*/
		MaxC,
		/**Absolute maximum of channel.*/
		MaxAbsC,
		/**Average of channel*/
		AvgC;
	}
	
	/**Side length of input map.*/
	private final int s;
	/**Side length of input map, squared.*/
	private final int sq;
	/**Side length of pooled map.*/
	private final int p;
	/**Side length of pooled map, squared.*/
	private final int pq;
	/**Number of channels.*/
	private final int c;
	/**Pooling constant.*/
	private final int f;
	/**Pooling constant, squared.*/
	private final int fq;
	/**Size of flattened input map.*/
	private final int is;
	/**Size of flattened output.*/
	private final int os;
	/**Pooled indices.*/
	private final int[] i;
	/**Pooled tensor.*/
	private final double[] o;
	/**Type of pooling operation.*/
	private final PoolingTypes t;
	/**True if the function pools across channels.*/
	private final boolean channelPooling;
	/**True if the function pools using averages.*/
	private final boolean avgType;
	/**Type for creating new {@linkplain Parameter} objects.*/
	private static final ListOfTypes outputType = new ListOfTypes(new Class<?>[] {double[].class});
	
	/**
	 * Creates a new pooling layer.
	 * 
	 * @param size Size of input map.
	 * @param channels Number of channels.
	 * @param factor Pooling constant.
	 * @param type Pooling operation type.
	 */
	public Pool(int size, int channels, int factor, PoolingTypes type) {
		if(size % factor != 0)
			error("Pooling constant is not a factor of the input size.");
		s = size;
		sq = s * s;
		f = factor;
		c = channels;
		is = c * sq;
		switch(t = type) {
		case MaxC	:
		case MaxAbsC:
		case AvgC	: channelPooling = true;
					  p = c / f;
					  os = p * sq;
					  pq = fq = 0;
					  break;
		default		: channelPooling = false;
					  p = s / f;
					  pq = p * p;
					  fq = f * f;
					  os = c * pq;
					  break;
		}
		o = new double[os];
		i = (avgType = (t == PoolingTypes.Avg || t == PoolingTypes.AvgC))? null:new int[os];
	}
	
	/**
	 * Performs the pooling operation on a flattened input map.
	 * 
	 * @param in Flattened input map.
	 * @return Flattened output map.
	 */
	public double[] forward(double[] in) {
		if(!channelPooling)
			for(int ch = 0; ch < c; ch++)
				for(int pr = 0; pr < p; pr++) {
					final int nr = pr * f;
					for(int pc = 0; pc < p; pc++) {
						final int nc = pc * f;
						double a = (t == PoolingTypes.Max)? -Double.MAX_VALUE:0.0;
						int x = (ch * sq) + (pr * s) + pc;
						for(int ir = 0; ir < f; ir++)
							for(int ic = 0; ic < f; ic++) {
								final int ridx = ir + nr,
										  cidx = ic + nc,
										  bidx = (ch * sq) + (ridx * s) + cidx;
								final double b = in[bidx];
								switch(t) {
								case Max   : if(b > a) {a = b; x = bidx;} 		   break;
								case MaxAbs: if(Math.abs(b) > a) {a = b; x = bidx;}break;
								case Avg   : a += b / (double) fq;				   break;
								default    : error("Invalid pooling type."); 	   break;
								}
							}
						final int oidx = (ch * pq) + (pr * p) + pc;
						o[oidx] = a;
						if(!avgType)
							i[oidx] = x;
					}
				}
		else {
			for(int ir = 0; ir < is; ir++) {
				final int ridx = ir * is;
				for(int ic = 0; ic < is; ic++) {
					final int cidx = ridx + ic;
					for(int pch = 0; pch < p; pch++) {
						final int pchf = pch * f;
						double a = (t == PoolingTypes.MaxC)? -Double.MAX_VALUE:0.0;
						int x = pch * f;
						for(int ich = 0; ich < f; ich++) {
							final int aidx = (pchf + ich) * sq + cidx;
							final double b = in[aidx];
							switch(t) {
							case MaxC: if(a < b) {a = b; x = aidx;}			   			   break;
							case MaxAbsC: if(Math.abs(a) < Math.abs(b)) {a = b; x = aidx;} break;
							case AvgC: a += b / (double) f; 				  			   break;
							default: error("Invalid pooling type."); 					   break;
							}
						}
						final int idx = pch * sq + cidx;
						o[idx] = a;
						if(!avgType)
							i[idx] = x;
					}
				}
			}
		}
		return o;
	}
	
	/**
	 * Wraps the output of {@linkplain #forward(double[])} in a {@linkplain Parameter}.
	 * 
	 * @param input A {@linkplain Parameter} containing the input tensor (<code>double[]</code>).
	 * @return The pooled output tensor (<code>double[]</code>) wrapped in a {@linkplain Parameter}.
	 */
	@Override
	public Parameter<ListOfTypes> forward(Parameter<ListOfTypes> input) {return new Parameter<ListOfTypes>(outputType, new Object[] {forward((double[]) input.getValues()[0])});}
	
	/**
	 * Feeds the gradient to the correct locations on the input map.
	 * 
	 * @param loss Flattened gradient with respect to the pooled output.
	 * @return Flattened gradient with respect to the input.
	 */
	public double[] backward(double[] loss) {
		final double[] out = new double[is];
		switch(t) {
		case Avg:
			{
				for(int ch = 0; ch < c; ch++) {
					final int csq = ch * sq,
					  	  	  cpq = ch * pq;
					for(int pr = 0; pr < p; pr++) {
						final int cpr = cpq + pr * p,
								  pfr = pr * f;
						for(int pc = 0; pc < p; pc++) {
							final int pfc = pc * f;
							final double g = loss[cpr + pc];
							for(int ir = 0; ir < f; ir++) {
								final int csr = csq + (pfr + ir) * s + pfc;
								for(int ic = 0; ic < f; ic++)
									out[csr + ic] = g;
							}
						}
					}
				}
			} break;
		case AvgC:
			{
				for(int pch = 0; pch < p; pch++) {
					final int bchidx = pch * sq,
							  pchf   = pch * f;
					for(int ich = 0; ich < f; ich++) {
						final int achidx = (pchf + ich) * sq;
						for(int ir = 0; ir < is; ir++) {
							final int iris  = ir * is,
									  aridx = achidx + iris,
									  bridx = bchidx + iris;
							for(int ic = 0; ic < is; ic++)
								out[aridx + ic] = loss[bridx + ic];
						}
					}
				}
			} break;
		default: for(int x = 0; x < os; x++) out[i[x]] = loss[x]; break;
		}
		return out;
	}
	
	/**
	 * Wraps the output of {@linkplain #backward(double[])} in a {@linkplain Parameter}.
	 * 
	 * @param loss A {@linkplain Parameter} containing the gradient tensor (<code>double[]</code>).
	 * @return The expanded gradient tensor (<code>double[]</code>) wrapped in a {@linkplain Parameter}.
	 */
	@Override
	public Parameter<ListOfTypes> backward(Parameter<ListOfTypes> loss) {return new Parameter<ListOfTypes>(outputType, new Object[] {forward((double[]) loss.getValues()[0])});}
	
	/**
	 * A custom exception which indicates an error in the pool.
	 * 
	 * @author prgmTrouble
	 */
	private static class PoolingException extends Exception {
		/***/
		private static final long serialVersionUID = 1L;
		private static final String prefix = "Pooling Exception: ";
		
		public PoolingException(String s) {super(prefix + s);}
	}
	
	/**
	 * Throws a {@linkplain PoolingException} and terminates execution.
	 * 
	 * @param s Description of error.
	 */
	protected static void error(String s) {
		try {
			throw new PoolingException(s);
		} catch(PoolingException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}






















































