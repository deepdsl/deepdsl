package deepdsl.optimization

import deepdsl.ast._
import deepdsl.derivation.DivOp 

object Simplify {
	def apply(e: AExp) : AExp = {
		var (r, b) = simp2(e)
		while (b) {
			simp2(r) match {
				case (e1, b1) => r=e1; b=b1
			}
		}
		r
	}
}

object simp2 {
	def apply(e: AExp) : (AExp, Boolean) = 
		e match { 
			case Sum(i, Num(0)) => (Num(0), true)
			case Sum(i, Delta(il, ir, exp)) => 
			  //(if (i == il) exp else Delta(il, ir, Sum(i, exp)), true)
			  (if(il.contains(i)) exp.sub(i, il.invert(i, ir)) else Delta(il, ir, Sum(i, exp)), true)
			
			case Sum(i, Plus(e1, e2)) => (Plus(Sum(i, e1), Sum(i, e2)), true)
			case Sum(i, Times(e1, Num(x))) => (Times(Sum(i, e1), Num(x)), true)
			// TODO: not correct if indexed-exp can match the lhs of the indicator more than once within the range of i
			case Sum(i, Indicator(_, indexedExp)) => if (indexedExp.free(i)) (Num(1), true) else (e, false)
			// move delta to outside of indicator
			case Indicator(Delta(il, ir, e1), e2) =>  (Delta(il, ir, Indicator(e1, e2)), true)  
			
			case ScalarApp(fun, Delta(il, ir, e)) => (Delta(il, ir, ScalarApp(fun, e)), true)
			 
			case Log(Exp(e11)) => (e11, true)
			case Exp(Log(e11)) => (e11, true) 
			case Exp(Num(0)) => (Num(1), true) 
			case Exp(Num(1)) => (E, true)
 
			case Pow(e1, Num(0)) => (Num(1), true) 
			case Pow(e1, Num(1)) => (e1, true)

			case Pow(Times(e1, e2), x) => (Times(Pow(e1, x), Pow(e2, x)), true)  

			case Pow(Pow(e1, x1), x2) => (Pow(e1, x1 * x2), true)
			case Pow(Exp(e1), x) => (Exp(e1 * x), true)

			case Sum(i, exp) => {
				val (exp1, b) = simp2(exp)
				if (b) { 
				  (Sum(i, exp1), true)
				}
				else {
				  val ret = (e, false)
				  exp match { 
				    // TODO: a hack to simplify sum of a product with an indicator function inside 
				    //       this only works if the indicator is at the left most position of a normalized product
				    case Times(e1, e2) => if (! e2.free(i)) 
				    	(Times(Sum(i, e1), e2), true) 
				      else ret 
				    case _ => ret
				  } 
				}
			}

			case Pow(e1, x1) => { 
				val (e11, b) = simp2(e1)
				val (x11, b2) = simp2(x1)
				if (b || b2) (Pow(e11, x11), true) else (e, false)
			}
			case Exp(e1) => {
				val (e11, b) = simp2(e1)
				if (b) (Exp(e11), true) else (e, false)
			}
			case Log(e1) => {
				val (e11, b) = simp2(e1)
				if (b) (Log(e11), true) else (e, false)
			}

			case Plus(e1, e2) => simp3(Normalize(Plus(e1, e2)))
			
			case Times(e1, e2) => simp3(Normalize(Times(e1, e2)))
			
			case Delta(il, ir, e1) => {
			  val (e2, b) = simp2(e1)
			  if (b) (Delta(il, ir, e2), true) else (e, false)
			}

			case _ => (e, false)
		}
} 

object simp3 {
	def apply(e: Times) : (AExp, Boolean) = {
		e match {
  
		case Times(Num(0), e2) => (Num(0), true)
		case Times(Real(0f, _), e2) => (Num(0), true)
		case Times(e1, Num(0)) => (Num(0), true)
		case Times(e1, Real(0f, _)) => (Num(0), true)
		case Times(Num(1), e2) => (e2, true)
		case Times(Real(1f, _), e2) => (e2, true)
		case Times(e1, Num(1)) => (e1, true)
		case Times(e1, Real(1f, _)) => (e1, true)
		case Times(Num(i), Num(j)) => (Num(i*j), true)
		case Times(Num(x), Real(y, _)) => (Real(x*y, ""), true)
		case Times(Real(x, _), Num(y)) => (Real(x*y, ""), true)
		case Times(Real(x, _), Real(y, _)) => (Real(x*y, ""), true)

		// TODO: ugly hack to deal with the derivation of max function so that the max indices are not substituted away
		case Times(Indicator(Max(i, x), e1), Delta(il, ir, e2)) => 
		  if (i.exists(j => il.contains(j))) { 
		    val k = i.filter(j => il.contains(j))(0)
		    val inv = il.invert(k, ir)
		    (Delta(il, ir, Times(Indicator(Max(i, x), e1.sub(k, inv)), e2)), true) 
		  }
		  else 
		    (Delta(il, ir, Times(Indicator(Max(i, x), e1), e2)), true)
		
		case Times(e1, Delta(il, ir, exp)) => (Delta(il, ir, Times(e1, exp)), true)
		case Times(Delta(il, ir, exp), e2) => (Delta(il, ir, Times(exp, e2)), true)

		case Times(e1, Plus(e2, e3)) => (Plus (Times(e1, e2), Times(e1, e3)), true)
		case Times(Plus(e1, e2), e3) => (Plus (Times(e1, e3), Times(e2, e3)), true)

		case Times(Exp(e1), Exp(e2)) => (Exp(e1 + e2), true)
		
		case Times(Num(i), e2) => (Times(e2, Num(i)), true)

		case Times(Times(e1, e2), e3) => {
			val (e12, b) = simp3(Times(e1, e2))
			if (b) 
			  (Times(e12, e3), true)
			else {
				val (e13, b2) = simp3(Times(e1, e3))
				if (b2) (Times(e13, e2), true)
				else {
					val (e23, b3) = simp3(Times(e2, e3))
					if (b3) (Times(e1, e23), true)
					else (e, false)
				}
			}
		}
		
		case Times(e1, e2) => { 
			val (e11, b1) = simp2(e1)
			val (e21, b2) = simp2(e2)
					 
			if (b1 || b2) 
				(Times(e11, e21), true) 
			else {
				val ret = (e, false)
				e match {
					case Times(Pow(e1, x1), Pow(e2, x2)) => if (equal(e1, e2)) (Pow(e1, x1+x2), true) else ret
					case Times(e1, Pow(e2, x2)) => if (equal(e1, e2)) (Pow(e1, Num(1)+x2), true) else ret
					case Times(Pow(e1, x1), e2) => if (equal(e1, e2)) (Pow(e1, x1+Num(1)), true) else ret
							
					// TODO: a temporary hack to simplify p * 1/p
					case Times(e1, ConstExp(c1, DivOp, c2)) => if (e1 == c2) (c1, true) else ret

					case _ => ret
				} 
			}
		  }
		
		}
	}

  def apply(e: Plus): (AExp, Boolean) = {
    e match { 
		case Plus(Num(0), e2) => (e2, true)
		case Plus(e1, Num(0)) => (e1, true)
		case Plus(Num(x), Num(y)) => (Num(x+y), true)
		case Plus(Num(x), Real(y, _)) => (Real(x+y, ""), true)
		case Plus(Real(x, _), Num(y)) => (Real(x+y, ""), true)
		case Plus(Real(x, _), Real(y, _)) => (Real(x+y, ""), true)
		
		case Plus(Plus(e1, e2), e3) => {
			val (e12, b) = simp3(Plus(e1, e2))
			if (b) 
			  (Plus(e12, e3), true)
			else {
				val (e13, b2) = simp3(Plus(e1, e3))
				if (b2) (Plus(e13, e2), true)
				else {
					val (e23, b3) = simp3(Plus(e2, e3))
					if (b3) (Plus(e1, e23), true)
					else (e, false)
				}
			}
		}
		
		case Plus(e1, e2) => { 
			val (e11, b1) = simp2(e1)
			val (e21, b2) = simp2(e2)

			if (b1 || b2) 
			  (Plus(e11, e21), true) 
			else { 
			  val ret = (e, false)
			  e match {
			    case Plus(Times(e11, Num(i)), Times(e21, Num(j))) => if (equal(e11, e21)) (Times(e11, Num(i+j)), true) else ret
			    case Plus(Times(e11, Num(i)), e2) => if (equal(e11, e2)) (Times(e11, Num(i+1)), true) else ret
			    case Plus(e1, Times(e21, Num(j))) => if (equal(e1, e21)) (Times(e1, Num(1+j)), true) else ret
			    
			    case Plus(Delta(il, ir, e1), Delta(il2, ir2, e2)) => if (il == il2 && ir == ir2) (Delta(il, ir, Plus(e1, e2)), true) else ret
			    
			    case _ => if (equal(e1, e2)) (Times(e1, Num(2)), true) else ret
			  }
			}
		}

    }
  }
}

object equal {
  def apply(e1: AExp, e2: AExp): Boolean = (e1, e2) match {
      case (Times(e1, e2), Times(e11, e21)) => equal(e1, e11) && equal(e2, e21)
      case (Plus(e1, e2), Plus(e11, e21)) => equal(e1, e11) && equal(e2, e21)
      case (Sum(i1, e1), Sum(i2, e2)) => i1 == i2 && equal(e1, e2)
      case (Pow(e1, x1), Pow(e2, x2)) => equal(e1, e2) && equal(x1, x2)
      case (Exp(e1), Exp(e2)) => equal(e1, e2)
      case (Log(e1), Log(e2)) => equal(e1, e2)
      case (Num(i), Num(j)) => i == j   
      case (ScalarApp(e1, e2), ScalarApp(e11, e21)) => e1 == e11 && equal(e2, e21)
      case (ScalarDec(n1), ScalarDec(n2)) => n1 == n2
      case _ => e1 == e2 
  }
}
