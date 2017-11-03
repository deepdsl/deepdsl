package deepdsl.run

import deepdsl.derivation._
import deepdsl.layer._
import deepdsl.analysis.TypeException 
import deepdsl.ast._ 

class Env(dimE: Map[DimVar, Range]) {
  
  def +(d: DimVar, r: Range) = new Env(dimE + (d -> r))
  
  def apply(init: Init): Float = init match {
    case ConstInit(x) => x 
    case Xavier(fan_in, fan_out) => Math.sqrt(6.0 / (apply(fan_in).size + apply(fan_out).size)).asInstanceOf[Float] 
    case Caffe_Xavier(fan_in) => Math.sqrt(2.0 / apply(fan_in).size).asInstanceOf[Float] 
    case Gaussian(std) => std
  }
  
	def arith(n1: Int, op: Op, n2: Int): Int = {
			op match {
			case PlusOp => n1 + n2
			case TimesOp => n1 * n2
			case MinusOp => n1 - n2
			case DivOp => n1 / n2
			case ModOp => n1 % n2
			}
	}
	def apply(c: Const) : Int = {
			c match {
			case ConstExp(l, op, r) =>   arith(apply(l), op, apply(r))   
			case Num(n) => n 
			case DimSize(d) => apply(d).size 
			}
	}

	def apply(d: Dim) : Range = {
			d match {
  			case DimConst(b) => Range(0, b)
  			case x @ DimVar(_) => dimE(x)
  			case DimOpConst(d1, op, c) => {
  				val r = apply(d1)
  						val n = apply(c)
  
  						op match {
  						case PlusOp => Range(r.start, r.end + n) 
  						case MinusOp => {
  						  if(r.size < n) {
  						    throw new TypeException("Illegal dimension" + d)
  						  }
  						  Range(r.start, r.end - n)
  						}
  						case DivOp => {
  						  if(r.size < n) {
  						    throw new TypeException("Illegal dimension" + d)
  						  }
  						  Range(r.start, r.start + r.size / n)
  						}
  						case TimesOp => Range(r.start, r.start + r.size * n)
  				}
  			}
			}
	}
}

class InterpretEnv(dimE: Map[DimVar, Range], idxE: Map[IndexDec, Int]) extends Env(dimE) {
	def apply(i: Index): Int = {
		i match {
		case x @ IndexDec(_, _) => idxE(x)
		case IndexOpConst(i1, op, c) =>  arith(apply(i1), op, apply(c)) 
		case IndexOpIndex(i1, op, i2) =>  arith(apply(i1), op, apply(i2)) 
		case QuotientRemainder(q, d, r) => apply(q) * apply(d) + apply(r)
		}
	}
}

class CompileEnv(dimE: Map[DimVar, Range]) extends Env(dimE) {
	def arith(n1: String, op: Op, n2: String) = {
		"(" + (op match {
		case PlusOp => n1 + "+" + n2
		case MinusOp => n1 + "-" + n2
		case TimesOp => n1 + "*" + n2
		case DivOp => n1 + "/" + n2
		case ModOp => n1 + "%" + n2
		}) + ")"
	}
	def apply(i: Index): String = {
		(i match {
		case x @ IndexDec(_, _) => x.toString
		case IndexOpConst(i1, op, c) =>   arith(apply(i1), op, apply(c).toString) 
		case IndexOpIndex(i1, op, i2) =>   arith(apply(i1), op, apply(i2)) 
		case QuotientRemainder(q, d, r) =>  s"(${apply(q)} * ${apply(d)} + ${apply(r)})"
		}) 
	}
}