package com.prgmtrouble.ml.prgmML.math;

import java.util.TreeSet;

public final class Miscellaneous {
	
	/**
	 * Returns the greatest factor of <code>n</code> which is a perfect square.
	 * @param n Input.
	 * @return The greatest value for <code>x</code> such that
	 * 		   <code>(n % x == 0) && (x % (int) Math.sqrt(x) == 0)</code>.
	 */
	public static int greatestSquareFactor(int n) {
		final TreeSet<Integer> factors = new TreeSet<Integer>(); //Use TreeSet to sort.
		final int step = n % 2 == 0 ? 1 : 2; //Skip 2 if odd.
		for(int i = 1, j = 1; i <= Math.sqrt(n); i += step, j = i * i)
			if(n % j == 0) //Number is factor iff remainder is zero.
				factors.add(j);
		return factors.last(); //Get largest square factor.
	}
}
















































