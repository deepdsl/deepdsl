package deepdsl.ast

import deepdsl.gradient.Gradient 
import deepdsl.derivation._

// arithmetic expression
trait AExp extends AExpOp {  
  def +(e: AExp) = this match {
    case Num(0) => e
    case _ => e match {
      case Num(0) => this
      case _ => Plus(this, e)
    }
  }
  def -(e: AExp) = Plus(this, Times(e, Num(-1)))
  def *(e: AExp) = this match {
    case Num(0) => Num(0)
    case Num(1) => e
    case _ => e match {
      case Num(0) => Num(0)
      case Num(1) => this
      case _ => Times(this, e)
    }
  }
  def *(v: Vec) = this match {
    case Num(1) => v
    case _ => VecTimesScalar(v, this)
  }
  def /(e: AExp) = e match {
    case Pow(e1, Num(x)) => Times(this, Pow(e1, Num(x-1)))
    case Pow(e1, Real(x, _)) => Times(this, Pow(e1, Real(x-1, "")))
    case _ => Times(this, Pow(e, Num(-1)))
  }
  def ^(x: Int) = Pow(this, Num(x))
  def ^(x: Float) = Pow(this, Real(x, ""))
  def ^(x: Const) = Pow(this, x)

  def grad(v: VecDec) : Vec = Gradient(this, v)  
  def grad(v: ScalarVar) : AExp = Gradient(this, v)
}

// p: prediction matrix (N x K), y_label: label vector (N), top: top-k accuracy
case class Accuracy(p: Vec, y_label: Vec, top: Int) extends AExp
// x
case class Real(x: Float, name: String) extends AExp
// Kronecker delta => e if idx_old == idx_new and => 0 otherwise
case class Delta(idx_old: Index, idx_new: Index, e: AExp) extends AExp 
// indicator function => 1 if e == indexedExp
case class Indicator(e: AExp, indexedExp: AExp) extends AExp 
// index expression
case class IndexExp(i: Index) extends AExp 
// max of an expression (e.g. a vector element) over an index
case class Max(i: Seq[IndexDec], e: AExp) extends AExp
// e1 * e2
case class Times(e1: AExp, e2: AExp) extends AExp 
// e1 + e2
case class Plus(e1: AExp, e2: AExp) extends AExp  
// e_1 + e_2 + .. + e_i + .. + e_N, where i ranges from 1 to N
case class Sum(i: IndexDec, val e: AExp) extends AExp 
// e^x
case class Pow(e: AExp, x: AExp) extends AExp
// exp^e
case class Exp(e: AExp) extends AExp
// ln(e)
case class Log(e: AExp) extends AExp
// Math.E
case object E extends AExp // Had to do this to prevent matching error in evaluation function
// vec[idx[0], idx[1], .., idx[n]]
case class VecElem(vec: VecDec, idx: List[Index]) extends AExp with ScalarVar
// v1[0] * v2[0] + v1[1] * v2[1] + ... + v1[n] * v2[n]
case class InnerProd(v1: Vec, v2: Vec) extends AExp 
// max{v[0], v[1], ..., v[n]}
case class MaxVec(v: Vec) extends AExp 
// v[0] + v[1] + ... + v[n]
case class SumVec(v: Vec) extends AExp 
 // f_e e2
case class ScalarApp(fun: ScalarFun, arg: AExp) extends AExp  
// f_ve v => e'
case class Vec2ScalarApp(fun: Vec2ScalarFun, arg: Vec) extends AExp 

 