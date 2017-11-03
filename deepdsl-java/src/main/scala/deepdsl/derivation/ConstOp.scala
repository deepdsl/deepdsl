package deepdsl.derivation

import deepdsl.ast._

trait ConstOp extends AExp {
  override def toString = this match {
    case DimSize(dim) => "|" + dim + "|"
    case ConstExp(left: Const, op: Op, right: Const) => left.toString + op + right 
    case Num(i) => i.toString
  }
//	override def clone: Const = this match {  
//    case DimSize(d) => DimSize(d.clone)
//    case ConstExp(c1, op, c2) => ConstExp(c1.clone, op, c2.clone)
//    case x:Const => x
//  }
  
  override def sub(dl: Dim, dr: Dim): Const = {
    this match {
      case DimSize(dim) => DimSize(dim.sub(dl, dr))
      case ConstExp(c1, op, c2) => ConstExp(c1.sub(dl, dr), op, c2.sub(dl, dr))
      case x:Const => x
    }
  }
  override def sub(il: IndexDec, ir: Index): Const = {
    this match {
      case ConstExp(c1, op, c2) => ConstExp(c1.sub(il, ir), op, c2.sub(il, ir))
      case x:Const => x
    }
  }
  override def free(i: IndexDec): Boolean = {
    this match {
      case ConstExp(c1, op, c2) => c1.free(i) || c2.free(i)
      case _ => false
    }
  }
  def apply(c1: Const, op: Op, c2: Const): Const = {
	  val ret = ConstExp(c1, op, c2)

	  // TODO: does not simplify constant expressions with variables inside
	  (c1, c2) match {
			  case (Num(i), Num(j)) => Num(op(i, j))  
			  case (Num(0), _) => op match {
			    case PlusOp => c2
			    case TimesOp => Num(0)
			    case DivOp => Num(0)
			    case _ => ret
			  }
			  case (Num(1), _) => op match {
			    case TimesOp => c2
			    case _ => ret
			  }
			  case (_, Num(0)) => op match {
			    case PlusOp => c1
			    case TimesOp => Num(0)
			    case _ => ret
			  }
			  case (_, Num(1)) => op match {
			    case TimesOp => c1
			    case DivOp => c1
			    case _ => ret
			  } 
			  case _ => ret
	  }
  }
}