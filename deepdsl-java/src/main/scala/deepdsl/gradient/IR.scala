package deepdsl.gradient

import deepdsl.optimization._
import deepdsl.derivation._
import deepdsl.layer._
import deepdsl.lang._
import deepdsl.ast._

object IR { 
  def unit[S, A] = (x: A) => MonadState((s: S) => (x, s))
	def get[S] = MonadState((s: S) => (s, s))
	def put[S] = (s: S) => MonadState((_: S) => ((), s))
	def modify[S] = (f : S => S) => MonadState((s: S) => ((), f(s))) 
	
  case class MonadState[S, A] (runState: S => (A, S)) {
    def evalState = (s: S) => runState(s)._1
    def execState = (s: S) => runState(s)._2
    
    def >>=[B] (f: A => MonadState[S, B]) = 
      MonadState ((s: S) => {
        val (a, newState) = runState(s)
        val g = f(a).runState
        g(newState)
      })
    def >>[B] (m: MonadState[S, B]) = >>= (_ => m)
  }
  
	type Env = Map[VecDec, Vec]
  val initEnv : Env = Map()
  // ([x -> de/dx], [de/dx -> v], [x -> v]) for some e 
  type Env2 = (Map[VecDec, VecDec], Env, Env)  
  type Env3 = (List[Let], Env)
  
  def let(e: AExp) : MonadState[Env3, AExp] = let(simplify(e))
  
  def let(v: Vec) : MonadState[Env3, VecDec] = let(simplify(v))

	def let[X](m: MonadState[Env, X]) : MonadState[Env3, X] = {
	  MonadState[Env3, X](s => {
	    val (a, s1) = m.runState(initEnv)  
	     
	    (a, (s._1 ++ s1.map(x => VecLet(x._1, x._2)), s._2 ++ s1))
	  })
	}	
	
	// this method recursively discover the live x = v by tracing its usage starting from the root set
	private def live(env: Env, root: Set[VecDec], set: Set[VecDec]): Set[VecDec] = {
	  root.foldLeft(set)((c, e) => 
	    if(!c.contains(e) && env.contains(e)) {
	      c + e ++ live(env, env(e).free, c)
	    }
	    else 
	      c
	  )
	}
	
	def grad(e: AExp, p: List[VecDec]) : MonadState[Env3, List[VecDec]] = grad(gradient(e), p)
	
	def grad(v: VecDec, p: List[VecDec]) : MonadState[Env3, List[VecDec]] =  grad(gradient(v), p)
	
	def grad[X](m: MonadState[Env2, X], p: List[VecDec]) : MonadState[Env3, List[VecDec]] = {
	  MonadState[Env3, List[VecDec]](s => {
	    val (a, s1, _) = m.execState((Map(), initEnv, s._2)) 
	    val dp = p.map(x => a(x))          // variables corresponding to dc/dp
	    val lv = live(s1, dp.toSet, Set()) // discover live x = v starting from parameters.
	    val s2 = s1.filter(x => lv.contains(x._1))
	    
	    (dp, (s._1 ++ s2.map(x => VecLet(x._1, x._2)), s._2))
	  })
	}
	
	def write(let: Let) : MonadState[Env3, Unit] = write(List(let))
	
	def write(lst: List[Let]) : MonadState[Env3, Unit] = {
	  modify(s => (s._1 ++ lst, s._2))
	}
	
	def read[X](m: MonadState[Env3, X]) = m.execState((Nil, initEnv))._1    
  
  def ssa(v: Vec) : MonadState[Env, VecDec] = v match {
    case x@VecDec(_,_) => unit(x)
//    case FlattenVec(_, _, _) 
//        | UnFlattenVec(_, _, _) 
//        | VecSlice(_, _) => unit(v)
    case _ => {
  	  val x = T._new(v.getDims)
  		MonadState(s => (x, s + (x -> v))) 
  	  // The order is important. "s" is the list of previous assignments, "x = v" is the next one
    }
  }
  def simplify(e: AExp) : MonadState[Env, AExp] = e match  {
    case Plus(e1, e2) => simplify(e1) >>= (x1 => simplify(e2) >>= (x2 => unit(Plus(x1, x2))))
    case Times(e1, e2) => simplify(e1) >>= (x1 => simplify(e2) >>= (x2 => unit(Times(x1, x2))))
    case Vec2ScalarApp(f, a) => simplify(a) >>= (v2 => simplify(f.body.sub(f.param, v2)))
    case _ => unit(e) 
  }
  def simplify(v: Vec) : MonadState[Env, VecDec] = v match {
    case VecApp(f, a) => simplify(a) >>= (v2 => simplify(f.body.sub(f.param, v2))) 
    case VecPlus(v1, v2) => simplify(v1) >>= (v11 => simplify(v2) >>= (v12 => ssa(VecPlus(v11, v12)))) 
    case VecTimesScalar(v1, e2) => simplify(v1) >>= (v11 => simplify(e2) >>= (e12 => ssa(VecTimesScalar(v11, e12)))) 
    case ConcatVec(lst) => lst.foldRight(unit[Env, List[Vec]](Nil))(
        (e, c) => c >>= (
            l => simplify(e) >>= (
                v => unit(v::l)))
        ) >>= (xlst => ssa(ConcatVec(xlst)))
    case VecExp(i, e) => {
//        if (i.exists(j => e.free(j))) { 
//            ssa(v)
//          }
//          else {
//            simplify(e) >>= (e1 => ssa(VecExp(i, e1))) 
//          }
      
  	  val m = simplify(e)
  	  val lst = m.execState(initEnv)
  	  
  		if(i.forall(j => lst.forall(p => p._2.free(j)))) {
  				m >>= (e1 => ssa(VecExp(i, e1)))
  		}
  		else {
  				ssa(v)
  		}
    }
    case _ => ssa(v) 
  }
  
  // e => m[env, [x, de/dx]] 
  // compute nabla of e and the gradients are stored in env
  def gradient (e: AExp) : MonadState[Env2, Unit] = { 
			val xs = e.free
			xs.foldLeft[MonadState[Env2,Unit]](unit(()))((c, x) => c >> (gradient(x) >>= 
				 (dx => modify(s => (s._1, (s._2 + (dx -> e.grad(x))), s._3)
		  	 )))) 
  }
  // extract the gradient map 
  def extract[A] (m: MonadState[Env2, A]): MonadState[Env, (Map[VecDec, VecDec], Env)] = {
      MonadState(s => {
        val (_, s1) = m.runState((Map(), initEnv, s))
        ((s1._1, s1._2), s1._3)
      })
  }
  
  // y => m[env2, dy]
  // compute de/dy and return dy, where y -> dy and dy -> de/dy are stored in env
  def gradient (y : VecDec) : MonadState[Env2, VecDec] = {
		 get >>= (s => {
			  val (g, genv, env) = s 
			  
				// if dy is not defined yet (not visited)
			  if(!g.contains(y)) {
			    val dy = T._new(y.getDims)
			    val g1 = g + (y -> dy)
			    
			    // learned parameters have no definition in env
			    // y must has a binding in env.
			    if(env.contains(y)) { 
						  val vy = env(y)
						  
							vy.free.foldLeft(put((g1, genv, env)))((c, x) => c >> (gradient(x) >>= (dx => modify(s => {
										  val vdx = dy * vy.grad(x)  
											(s._1, s._2 + (dx -> (if(s._2.contains(dx)) s._2(dx) + vdx else vdx )), s._3)   
						  })))) >> unit(dy)
					}
					else
						  put(g1, genv, env) >> unit(dy)
			  }
			  else 
			    unit(g(y))
		})
  } 
}
 

