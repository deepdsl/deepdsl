package deepdsl.ast

import deepdsl.derivation._

trait Const extends ConstOp { 
  def -(c: Const) = op(MinusOp, c)
  def +(c: Const) = op(PlusOp, c)
  def *(c: Const) = op(TimesOp, c)
  def /(c: Const) = op(DivOp, c)
  def %(c: Const) = op(ModOp, c)
  def op(op: Op, c: Const) = apply(this, op, c)
}

// size of a dimension
case class DimSize(dim: Dim) extends Const 
// left op right
case class ConstExp(left: Const, op: Op, right: Const) extends Const  
// i
case class Num(i: Int) extends Const 


