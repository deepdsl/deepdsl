package deepdsl.ast

import deepdsl.derivation._ 

trait Index extends IndexOp { 
  def +(c: Const) = apply(PlusOp, c)
  def -(c: Const) = apply(MinusOp, c)
  def *(c: Const) = apply(TimesOp, c)
  def /(c: Const) = apply(DivOp, c)
  def %(c: Const) = apply(ModOp, c)
  def +(i: Index) = apply(PlusOp, i)
  def -(i: Index) = apply(MinusOp, i)
  def *(i: Index) = apply(TimesOp, i)
  def /(i: Index) = apply(DivOp, i)
  def *+(d: Const, r: Index) = QuotientRemainder(this, d, r)
   
  def unflatten(flatDims: List[Dim]) = {
     val ret = flatDims.zipWithIndex.map(dim_pos => {
	    		val d = flatDims.slice(dim_pos._2 + 1, flatDims.length).foldRight[Index](this) ((i, c) =>  c/i.size)
	    		if (dim_pos._2 == 0) d else d % dim_pos._1.size
	    	})
	   ret
  }
    
  def grad(that: Index) : AExp  = {
    if (this == that) 
      Num(1)  
    else { 
      Delta(this, that, Num(1))
    }
  }
}

// quotient * divisor + remainder, where |remainder| < divisor
case class QuotientRemainder(quotient: Index, divisor: Const, remainder: Index) extends Index {  
  def simp = (quotient, remainder) match {
      case (IndexOpConst(j1, DivOp, d1), IndexOpConst(j2, ModOp, d2)) => if (j1 == j2 && d1 == d2) j1 else this
      case _ => this
    }
}

case class IndexDec(idx: IndexVar, dim: Dim) extends Index {   
  override def sub(dl: Dim, dr: Dim) = IndexDec(idx, dim.sub(dl, dr)) 
}

case class IndexVar(name: String) {
  override def toString = name
  // compare by object identity
  override def equals(that: Any) = super.equals(that)
}

case class IndexOpConst(val index: Index, val op: Op, val c: Const) extends Index 

case class IndexOpIndex(val indexL: Index, val op: Op, val indexR: Index) extends Index  

