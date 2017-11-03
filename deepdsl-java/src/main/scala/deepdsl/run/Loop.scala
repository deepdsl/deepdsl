package deepdsl.run

import deepdsl.gradient._
import deepdsl.optimization._
import deepdsl.derivation._ 
import deepdsl.lang._
import deepdsl.ast._
import deepdsl.ast.ConstInit

// 0 momentum -> No SGD
// 0 decay -> No weight decay
// 0 clipping -> No gradient clipping
case class Train(
    name: String, 
    train_itr: Int, 
    test_itr: Int, 
    learn_rate: Float, 
    momentum: Float, 
    decay: Float, 
    clipping: Float)

case class Loop (
    cost: AExp, 
    forward: Vec, 
    data: Data, 
    input: (VecDec, VecDec), 
    param: List[VecParam], 
    solver: Train) {	
  
	def update(p: VecParam, gradient: Vec) = { 
      val upd = if (solver.clipping > 0) gradient.clip(Real(solver.clipping, "clip")) else gradient
       
      val lm = Num(p.init.lr_mul)
      val learn_rate = Real(- solver.learn_rate, "lrn_rate")
      val lr = lm * learn_rate
      val dm = Num(p.init.decay_mul)  
      val dc = if(dm == Num(0) || solver.decay == 0) Num(1) else {
        val decay = Real(solver.decay, "decay")
        Num(1) + dm * decay * lr  
      }
       
      if(solver.momentum > 0) {
        val mm = Real(solver.momentum, "momentum")
        val init = ZeroInit // right now we don't save parameters with this initialization
        val v = T._new(init, "V_"+p.v.name, upd.getDims:_* )
           
        List(v.update((upd * lr) + (v * mm)), p.update(v + (p * dc)))
      }
      else {
        List(p.update((upd * lr) + (p * dc)))
      } 
  }
  val (x,y) = input
  
  val test = SSA(false).optimize(LetVec(Nil, forward))
  
  // take gradients for all parameters simultaneously
  val train = { 
		val x = ScalarDec("_loss")
		
    val lst = IR.let(cost) >>= (c => (
         IR.write(x <-- c) >>
         IR.grad(c, param) >>= (g => 
         IR.write((param zip g).flatMap(x => update(x._1, x._2))) )))  
    
    val (s, _) = lst.execState((Nil, Map()))      
    SSA(true).optimize(LetExp(s, x))
  }  
}