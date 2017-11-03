package deepdsl.ast

import deepdsl.lang.T.int2Exp
import deepdsl.gradient.Gradient
import deepdsl.lang._ 
import deepdsl.optimization.ExpLet

trait ScalarVar 

case class ScalarDec(name: String) extends AExp with ScalarVar {
  def <--(e: AExp) = ExpLet(this, e)
  // compare by object identity
  override def equals(that: Any) = super.equals(that) 
}

// f_e: AExp -> AExp
case class ScalarFun (param: ScalarDec, body: AExp) {
	override def toString = "(" + param + ") => " + body 
	
  def this(param: ScalarDec, fun: AExp => AExp) = this(param, fun(param)) 
  
  def apply(arg: AExp) = ScalarApp(this, arg)
  
  def apply(arg: Vec) = ScalarVecApp(this, arg)
  
  // TODO: add free indices?
  def grad = ScalarFun(param, body.grad(param))
  
  def o (f: ScalarFun) = ScalarFun(f.param, this(f.body))
  def sub(vl: ScalarDec, vr: AExp) = if(param == vl) this else ScalarFun(param, body.sub(vl, vr))
    
  override def equals(that: Any): Boolean =  
    if (that.isInstanceOf[ScalarFun]) {
      val thatFun = that.asInstanceOf[ScalarFun]
      body.sub(param, thatFun.param) == thatFun.body 
    }
    else
      false 
} 

object f_tanh extends ScalarFun (ScalarDec("x"), x => {val exp = Exp(-2 * x); (1 - exp) / (1 + exp)}) {
  override def toString = "tanh"
  override def grad = T.fun(x => 1 - (x^2)) o T.fun(x => f_tanh(x))
}

// f_ve: Vec -> AExp
case class Vec2ScalarFun (param: VecDec, body: AExp) {
	override def toString = "(" + param + ") => " + body 
	
  def apply(v: Vec) = {
		val dims = v.getDims
    Vec2ScalarApp((param.dim zip dims).foldLeft(this)((c, e) => c.sub(e._1, e._2)), v)
	}

  def grad() = VecFun(param, Gradient(body, param))
  
    // TODO: add free indices? 
  def grad(v: VecDec) = VecFun(param, body.grad(v)) 
  
  // this o fun = \lambda x. (this ( fun (x) )
  def o (fun: VecFun) =  { Vec2ScalarFun(fun.param, this(fun.body)) }
  
  def o (fun: ScalarFun) = { val p = T._new(param.size); Vec2ScalarFun(p, this(fun(p))) }
  
  def sub(dl: Dim, dr: Dim) = Vec2ScalarFun(param.sub(dl, dr), body.sub(dl, dr))
  def sub(il: IndexDec, ir: Index) = Vec2ScalarFun(param, body.sub(il, ir)) // TODO: what about parameter "param"
  def sub(vl: Vec, vr: Vec) = if(param == vl) Vec2ScalarFun(param, body.sub(vl, vr)) else this
  def contains(v: Vec) = if (param == v) false else body.contains(v)
  def free(i: IndexDec) = body.free(i)
  
  override def equals(that: Any) = 
    if (that.isInstanceOf[Vec2ScalarFun]) {
      val thatFun = that.asInstanceOf[Vec2ScalarFun]
      body.sub(param, thatFun.param) == thatFun.body
    }
    else 
      false
}

// f_v: Vec -> Vec
case class VecFun (param: VecDec, body: Vec) { 
	override def toString = "(" + param + ") => " + body  
  
  def apply(arg: Vec) = {
    val dims = arg.getDims
    VecApp((param.dim zip dims).foldLeft(this)((c, e) => c.sub(e._1, e._2)), arg)
  }
  // TODO: add free indices?
  def grad = VecGradientFun(param, body.grad(param)) 
  def grad(v: VecDec) = VecGradientFun(param, body.grad(v)) 
  
  // this o fun = \lambda x. (this ( fun (x) )
  def o (fun: VecFun) = { VecFun(fun.param, this(fun.body)) }
  
  def o (fun: ScalarFun) = { val p = T._new(param.size); VecFun(p, this(ScalarVecApp(fun, p))) }
  
  def sub(dl: Dim, dr: Dim) = VecFun(param.sub(dl, dr), body.sub(dl, dr))
  def sub(vl: Vec, vr: Vec) = if(param == vl) this else VecFun(param, body.sub(vl, vr))
  def sub(il: IndexDec, ir: Index) = VecFun(param, body.sub(il, ir))
  def contains(v: Vec) = param != v && body.contains(v)
  def free(i: IndexDec) = body.free(i) 
  
  override def equals(that: Any) = 
    if (that.isInstanceOf[VecFun]) {
      val thatFun = that.asInstanceOf[VecFun]
      body.sub(param, thatFun.param) == thatFun.body
    }
    else 
      false
}

// f_v2d: Vec -> VecGradient
case class VecGradientFun(param: VecDec, body: VecGradient) {
  def apply(arg: Vec) = VecGradient(y => VecFun(param, body(y))(arg)) 
  def print = "(" + param + ") => " + body
}



