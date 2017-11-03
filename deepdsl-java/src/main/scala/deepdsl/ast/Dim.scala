package deepdsl.ast

import deepdsl.derivation._

// dimension 
trait Dim { 
  override def toString = print 
  def print: String = this match {
    case DimConst(bounds) => bounds.toString
    case DimVar(name) => name
    case DimOpConst(d, op, c) => "("+ d + op + c + ")" 
  }
  def sub(dl:Dim, dr:Dim): Dim = this match {
    case DimConst(_) => this
    case DimVar(_) => if(this == dl) dr else this
    case DimOpConst(d, op, c) => DimOpConst(d.sub(dl, dr), op, c.sub(dl, dr))
  }

  def size: Const = DimSize(this)
  def *(d: Dim) : Dim = this.*(d.size)
  def /(d: Dim) : Dim = this./(d.size)
  def -(d: Dim) : Dim = this.-(d.size)
  def +(d: Dim) : Dim = this.+(d.size)
  def *(c: Const) = DimOpConst(this, TimesOp, c)
  def /(c: Const) = DimOpConst(this, DivOp, c)
  def -(c: Const) = DimOpConst(this, MinusOp, c)
  def +(c: Const) = DimOpConst(this, PlusOp, c) 
}

case class DimConst(bound: Int) extends Dim  

case class DimVar(name: String) extends Dim {
   // compare by object identity
  override def equals(that: Any) = super.equals(that) 
} 
case class DimOpConst(d: Dim, op: Op, c: Const) extends Dim  

