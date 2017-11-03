package deepdsl.gradient

import deepdsl.derivation._ 
import deepdsl.optimization._
import deepdsl.layer._
import deepdsl.optimization.Simplify
import deepdsl.ast._

object Gradient {
	// free: the free indices are defined by "sum" functions or vector expressions
	// We invert index expressions containing free indices  
  def apply(self: AExp, v: ScalarVar) : AExp =  Simplify(app(self, v));
  
  private def app(self: AExp, v: ScalarVar) = self match {
			  case VecElem(v1, idx) => v match {
  					case VecElem(vector, indices) => 
  						if (v1 != vector || idx.size != indices.size || idx.size < 1) 
  						  Num(0)
  						else 
  						  (for ((indexExp, index) <- merge_quotient_remainder(idx zip indices)) yield indexExp.grad(index)) reduce ((c, e) => c * e) 
  					case ScalarDec(_) => Num(0)
  			}
  			case x @ ScalarDec(_) => v match { 
  				  	case ScalarDec(_) => if (x == v) Num(1) else Num(0) // Here v must be ScalarVar, which is AExp, and comparable to e
  				  	case VecElem(_,_) => Num(0) 
  		  }
				case ScalarApp(f, arg) => f.grad(arg) * apply(arg, v)
				case Vec2ScalarApp(f, arg) =>  f.grad()(arg) * apply(arg, v)   
				  
				case Sum(idx, exp) => Sum(idx, apply(exp, v))
				case Plus(e1, e2) => apply(e1, v) + apply(e2, v)
				case Times(e1, e2) => (e1 * apply(e2, v)) + (apply(e1, v) * e2)
				case Pow(e1, x) => Pow(e1, x - Num(1)) * apply(e1, v) * x
				case x @ Exp(e1) => x * apply(e1, v)
				case Log(e1) => apply(e1, v) / e1					
				case Max(i, x) => Times(Indicator(Max(i, x), x), apply(x, v))
				
				case _ => Num(0)
			} 
   	
  // List(i25, i33/|N22|/|N21|, i33/|N22|%|N21|, i33%|N22|) zip List(i88, i89, i90, i91)
  // becomes List(i25, i33) zip List(i88, ((i89*|N21|+i90)*|N22|+i91))
  private def merge_quotient_remainder(lst: List[(Index, Index)]) : List[(Index, Index)] = {
    for (x <- 0 to lst.size-2) {
      for(y <- x+1 to lst.size-1) {
        (lst(x)._1, lst(y)._1) match {
        case (IndexOpConst(i1, DivOp, c1), IndexOpConst(i2, ModOp, c2)) => if(i1 == i2 && c1 == c2) 
          return merge_quotient_remainder((i1, lst(x)._2 *+ (c1, lst(y)._2)) :: lst.filter(p => p._1 != lst(x)._1 && p._1 != lst(y)._1))  
        case (IndexOpConst(i1, ModOp, c1), IndexOpConst(i2, DivOp, c2)) => if(i1 == i2 && c1 == c2) 
          return merge_quotient_remainder((i1, lst(y)._2 *+ (c1, lst(x)._2)) :: lst.filter(p => p._1 != lst(x)._1 && p._1 != lst(y)._1))
        case _ => 
        } 
      }
    }
    lst
  }
 
	def apply(self: AExp, v: VecDec): Vec = self match {
    case Vec2ScalarApp(fun, arg) => { 
      if (arg.contains(v)) { // apply chain rule if the argument contains "v" and assuming function body is free from "v"
      	// lazy evaluation to cache intermediate result.  
    	  fun.grad()(arg) * Gradient(arg, v)   
      }
      else { // only the function body may contains "v"
    	  VecFun(fun.param, apply(fun.body, v))(arg) 
      }
      // TODO: Ignored the third possibility where the function body and argument can both contain "v"
    } 
    case Plus(e1, e2) if(e1.contains(v) && e2.contains(v)) => apply(e1, v) + apply(e2, v)
    case Plus(e1, e2) if(e1.contains(v)) => apply(e1, v)
    case Plus(e1, e2) if(e2.contains(v)) => apply(e2, v)
 
    case Times(e1, e2) if(e1.contains(v) && ! e2.contains(v)) => apply(e1, v) * e2
    case Times(e1, e2) if(! e1.contains(v) && e2.contains(v)) => apply(e2, v) * e1  
    
    case _ => {
      val i = v.getIndices
      VecExp(i, apply(self, VecElem(v, i))) 
    }
  }
  
  def apply(self: Vec, x: VecDec) : VecGradient = 
    self match { 
      case VecApp(fun, arg) => {  
        // assuming x CANNOT be in both fun and arg
    		if(arg.contains(x)) { 
    			VecGradient(y => fun.grad(arg)(y) * apply(arg, x)) // fun.grad(arg) * arg.grad(x)
    			// FIXME: free index doesn't extend beyond function scope?
    		}
    		else { 
    			VecGradient(y => VecFun(fun.param, apply(fun.body, x)(y))(arg)) // fun.grad(x)(arg)
    		}
      }
      case VecPlus(v1, v2) if(v1.contains(x) && v2.contains(x)) => {
        val g1 = apply(v1, x)
        val g2 = apply(v2, x)
        VecGradient(y => g1(y) + g2(y)) 
      }
      case VecPlus(v1, v2) if (v1.contains(x)) => apply(v1, x)
      case VecPlus(v1, v2) if (v2.contains(x)) => apply(v2, x)
      case VecTimesScalar(v1, e1) => VecGradient(y => apply(v1, x).fun(y) * e1)
      case ConcatVec(lst) => {
        if(! self.contains(x)) throw new Exception(s"gradient error: $x does not appear in $this") 
        VecGradient(y => {
    	      val grads = (0 to lst.size-1).map(n => ProjVec(y, lst, n))
    	      (grads zip lst).filter(p => p._2.contains(x)).map(p => p._1* apply(p._2, x)).reduce(_+_)
    	    }) 
      }
      case fv@FixVec(layer, param, dim) =>  {
        if(! self.contains(x)) throw new Exception(s"gradient error: $x does not appear in $this")
        val activeParam = layer match {
            case Pooling(_,_,_,_) | LRN(_,_,_)
              => fv :: param        // [y, x]  
            case Tanh() | Sigmoid() | ReLU() | ClippedReLU()
              => List(fv)           // [y] // it should be [y, x] but CUDNN doesn't really use "x"
            case Softmax() | LogSoftmax() 
              => List(fv)           // [y]
            case Convolv(_,_) => 
              if(x == param(0))       // d_data
                List(param(1))        // [w]
              else if (x == param(1)) // d_filter
                List(param(0))        // [x]
              else                    // d_bias
                Nil                   // []
            case Dropout(_) => Nil    // []
            case BatchNorm(_) 
               => List(param(0), param(1)) // [x, alpha]
          }
          VecGradient(y => FixProd(y, FixGrad(fv, activeParam, x))) 
      } 
    	  
      case _ => {
        val xIndices = x.getIndices 
        val scalarGrad = apply(self, VecElem(x, xIndices))
         
        VecGradient(y => VecExp(xIndices, y * scalarGrad))
      } 
  }
  def apply(self: Vec, x: ScalarVar) : VecExp = { 
    val i = self.getIndices 
    VecExp(i, apply(self.get(i), x)) 
  }
}
