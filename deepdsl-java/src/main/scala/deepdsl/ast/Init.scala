package deepdsl.ast

trait Init {
  var fixed = false
  var lr_mul = 1
  var decay_mul = 1
  
  def sub(dl: Dim, dr: Dim): Init = this match {
    case ConstInit(x) => this
    case Xavier(fan_in, fan_out) => { 
      val ret = Xavier(fan_in.sub(dl, dr), fan_out.sub(dl, dr))
      ret.lr_mul = lr_mul
      ret.decay_mul = decay_mul
      ret.fixed = fixed
      ret
    }
    case Caffe_Xavier(fan_in) => { 
      val ret = Caffe_Xavier(fan_in.sub(dl, dr))
      ret.lr_mul = lr_mul
      ret.decay_mul = decay_mul
      ret.fixed = fixed
      ret
    }
    case Gaussian(std) => this
  }
}
object ZeroInit extends Init
case class ConstInit(x: Float) extends Init 
case class Xavier(fan_in: Dim, fan_out: Dim) extends Init 
case class Caffe_Xavier(fan_in: Dim) extends Init 
case class Gaussian(std: Float) extends Init 