package deepdsl.ast

trait Data {
  val dim : List[Int]
  override def toString = this match {
    case Mnist(_) => "MNIST"
    case Lmdb(_,_,_,_) => "LMDB"
    case Imagenet(_,_,_,_) => "Imagenet"
  }
  def load(option: (Boolean, String)) = new VecData(this, option)
}

case class Mnist(dim: List[Int]) extends Data 
case class Lmdb(dim: List[Int], train_size:Int, test_size:Int, k: Int) extends Data  
case class Imagenet(dim: List[Int], train_size:Int, test_size:Int, k: Int) extends Data 
