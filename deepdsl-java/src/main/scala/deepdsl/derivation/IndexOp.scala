package deepdsl.derivation

import deepdsl.ast._

// utility classes
object InverseException extends Exception("inversion error")

trait Op { 
  def flip : Op 
  def apply(x: Int, y: Int) : Int = {
    this match {
      case TimesOp => x * y
      case DivOp => x / y
      case PlusOp => x + y
      case MinusOp => x - y
      case ModOp => x % y
    }
  }
  override def toString = {
    this match {
      case TimesOp => "*"
      case DivOp => "/"
      case PlusOp => "+"
      case MinusOp => "-"
      case ModOp => "%"
    }
  }
}
case object TimesOp extends Op {   def flip = DivOp }
case object DivOp extends Op {  def flip = TimesOp  }
case object PlusOp extends Op {  def flip = MinusOp  }
case object MinusOp extends Op {  def flip = PlusOp  }
case object ModOp extends Op { def flip = this }


trait IndexOp {
  def apply(op: Op, c: Const) : Index = this match {
    case x@QuotientRemainder(quotient, divisor, remainder) =>   
      op match {
        case ModOp if (divisor == c) => remainder 
        case DivOp if (divisor == c) => quotient  
        case _ => IndexOpConst(x, op, c)
      } 
    case x@IndexOpConst(index, op1, c1) =>    
      // (index / const * const) != index
      if ((op1 == TimesOp || op1 == PlusOp || op1 == MinusOp) && (op1.flip == op) && c1 == c) 
        index 
      else 
        IndexOpConst(x, op, c) 
    case x:Index => IndexOpConst(x, op, c)
  }
  def apply(op: Op, i: Index) : Index = this match {
    case x@IndexOpIndex(indexL, op1, indexR) => {
      if (op1.flip == op && indexR == i) 
        indexL 
  		else 
  		  IndexOpIndex(x, op, i)   
    }
    case x:Index => IndexOpIndex(x, op, i)
  }

  def sub(il: IndexDec, ir: Index) : Index = {
    this match {
      case IndexOpConst(index, op, c) => index.sub(il, ir)(op, c.sub(il, ir))
      case IndexOpIndex(indexL, op, indexR) => indexL.sub(il, ir)(op, indexR.sub(il, ir)) 
      case x@IndexDec(_,_) => if (il == this) ir else x
      case QuotientRemainder(q, d, r) => QuotientRemainder(q.sub(il, ir), d, r.sub(il, ir)).simp 
    }
  }
  def sub(dl: Dim, dr: Dim): Index = {
    this match {
      case IndexOpConst(index, op, c) => index.sub(dl, dr)(op, c.sub(dl, dr))
      case IndexOpIndex(indexL, op, indexR) => indexL.sub(dl, dr)(op, indexR.sub(dl, dr)) 
      case QuotientRemainder(q, d, r) => QuotientRemainder(q.sub(dl, dr), d.sub(dl, dr), r.sub(dl, dr))
//      case _ => this
    }
  }
  
  // return some(|D|) if i/|D| exists in this and none otherwise
  def getUpSample(i: IndexDec) : Option[Const] = this match {
    case QuotientRemainder(quotient, _, remainder) => 
      (quotient.getUpSample(i), remainder.getUpSample(i)) match {
        case (None, x) => x
        case (x, None) => x
        case (Some(d1), Some(d2)) => if (d1 == d2) Some(d1) else None
      }
    case IndexDec(_,_) => None
    case IndexOpConst(index, op, c) => if (i == index && op == DivOp) Some(c) else index.getUpSample(i)
    case IndexOpIndex(indexL, op, indexR) => 
      (indexL.getUpSample(i), indexR.getUpSample(i)) match {
        case (None, x) => x
        case (x, None) => x
        case (Some(d1), Some(d2)) => if (d1 == d2) Some(d1) else None
      } 
  }
  
  def contains(i: IndexDec) : Boolean = this match {
     case QuotientRemainder(quotient, divisor, remainder) => quotient.contains(i) || remainder.contains(i) 
     case x@IndexDec(idx, dim) => i == x 
     case IndexOpConst(index, op, c) => index.contains(i)
     case IndexOpIndex(indexL, op, indexR) => indexL.contains(i) || indexR.contains(i) 
  }
  
  
  // this = f(il) = ir ==> il = f^-1(ir)  
  def invert(il: IndexDec, ir: Index) : Index  = this match {
    case QuotientRemainder(quotient, divisor, remainder) => 
      if (il == quotient) 
        ir/divisor 
      else if (il == remainder) 
        ir%divisor 
      else 
        throw InverseException
    case IndexDec(idx, dim) =>  
      if (il == this) 
        ir 
      else 
        throw InverseException
    case IndexOpConst(index, op, c) => index.invert(il, ir(op.flip, c))
    case IndexOpIndex(indexL, op, indexR) => 
      if (indexL.contains(il)) 
	  		indexL.invert(il, ir(op.flip, indexR)) 
	  	else {
	  		val x = ir(op.flip, indexL)
	  	  val y = indexL(op, ir)
	  		indexR.invert(il, op match { 
	  		case PlusOp => x  
	  		case TimesOp => x 
	  		case MinusOp => y 
	  		case DivOp => y 
	  		})
	  	}
  }
  
  override def toString = this match {
     case QuotientRemainder(quotient, divisor, remainder) => "(" + quotient + "*" + divisor + "+" + remainder + ")"
     case IndexDec(idx, dim) => idx.toString // + ":" + dim 
     case IndexOpConst(index, op, c) => {
        val ret = index.toString + op + c
        if (op == PlusOp || op == MinusOp) "(" + ret  + ")" else ret
      } 
     case IndexOpIndex(indexL, op, indexR) => {
        val ret = indexL.toString + op + indexR
        if (op == PlusOp || op == MinusOp) "(" + ret  + ")" else ret
      } 
  }
}