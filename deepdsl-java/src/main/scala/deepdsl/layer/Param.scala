package deepdsl.layer

import deepdsl.derivation._
import deepdsl.ast.Dim
import deepdsl.ast.Gaussian
import deepdsl.ast.ConstInit
import deepdsl.ast.Caffe_Xavier

trait ParamInit
object XavierParam extends ParamInit // this refers to caffe-style Xavier initialization
case class ConstParam(x: Float) extends ParamInit
case class GaussianParam(std: Float) extends ParamInit

case class Param(init: ParamInit, lr_mul: Int, decay_mul: Int, isFixed: Boolean) {
  def toInit(fan_in: Dim) = {
    val ret = init match {
      case ConstParam(x) => ConstInit(x) 
      case GaussianParam(std) => Gaussian(std)
      case XavierParam => Caffe_Xavier(fan_in) 
    }
        
    ret.lr_mul = lr_mul
    ret.decay_mul = decay_mul
    ret.fixed = isFixed
    ret
  }
}

object Param {
  def xavier = Param(XavierParam, 1, 1, false)
  def gaussian(std: Float) = Param(GaussianParam(std), 1, 1, false)
  def const(x: Float): Param = const(x, 1, 1)
  def const(x: Float, lr_mul: Int, decay_mul: Int) = Param(ConstParam(x), lr_mul, decay_mul, false)
  def fixed(x: Float) = Param(ConstParam(x), 0, 0, true)
}