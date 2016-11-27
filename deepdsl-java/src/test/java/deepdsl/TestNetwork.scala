package deepdsl.derivation

import deepdsl.analysis._
import deepdsl.layer._
import deepdsl.optimization._
import deepdsl.run._
import org.junit.Test

class TestNetwork {
  val K = 1000 // # of classes
  //val lmdb = Lmdb(1000000, 10000, K) // # of training images, # of test images, # of classes
  val imagenet = Imagenet(1000000, 10000, K) // # of training images, # of test images, # of classes
  val env = new Env(Map())

  @Test
  def testResidualNN {
    val N = 64; val C = 3;  val N1 = 224; val N2 = 224 // batch size, channel, and x/y size

    // Specifying train dataSet
    val y = Vec._new(imagenet, "label", "Y", N)
    val x = Vec._new(imagenet, "image", "X", N, C, N1, N2)

    val relu = CudaLayer.relu(4)
    val pool = CudaLayer.max_pool(3, 2, 0)
    val ave_pool = CudaLayer.ave_pool(7, 1, 0)
    val full = Layer.full("fc", K)

    val flat = Layer.flatten(4, 1)
    val softmax = CudaLayer.log_softmax

    def cv_norm_relu(name: String, k: Int, c: Int, s: Int, p: Int) = {
      val cv = CudaLayer.convolv(name+"_cv", k, c, s, p, Param.xavier, Param.fixed(0)) // Fixed bias -- not trained
      val bn = CudaLayer.batch_norm(name+"_bn", 1f, 0f) // gamma (or scale) = 1 and beta (or bias) = 0
      val x = Vec._new(4)
      VecFun(x, (relu o bn o cv)(x))
    }

    def triple(name: String, m: Int, stride: Int) = {
      val x = Vec._new(4)
      val a = cv_norm_relu(name + "_a", 1, m*64, stride, 0)
      val b = cv_norm_relu(name + "_b", 3, m*64, 1, 1)
      val c = cv_norm_relu(name + "_c", 1, m*256, 1, 0)
      VecFun(x, (c o b o a)(x))
    }

    def branch(name: String, m: Int, stride: Int) = {
      val b1 = cv_norm_relu(name + "1", 1, m*256, stride, 0)
      val b2 = triple(name + "2", m, stride)
      val x = Vec._new(4)
      relu o VecFun(x, b1(x) + b2(x))
    }

    def bypass(name: String, m: Int) = {
      val x = Vec._new(4)

      relu o VecFun(x, triple(name, m, 1)(x) + x)
    }

    val subnet2 =  bypass("2c", 1) o bypass("2b", 1) o branch("2a", 1, 1)
    val subnet3 =  bypass("3d", 2) o bypass("3c", 2) o bypass("3b", 2) o branch("3a", 2, 2)
    val subnet4 =  bypass("4f", 4) o bypass("4e", 4) o bypass("4d", 4) o bypass("4c", 4) o bypass("4b", 4) o branch("4a", 4, 2)
    val subnet5 =  bypass("5c", 8) o bypass("5b", 8) o branch("5a", 8, 2)

    val network = full o flat o ave_pool o subnet5 o subnet4 o subnet3 o subnet2 o pool o cv_norm_relu("1", 7, 64, 2, 3)

    println(typeof(network))

    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda

    val c = Layer.loss(y1)((softmax o network)(x1))
    val p = Accuracy(network(x1), y, 1)         // top 1 accuracy 

    val param = c.freeVar.toList.sortWith((a,b) => a.toString < b.toString)

    val solver = Train("resnet", 1000, 10, 0.01f, 0.9f, 0.0005f, 0)
    val loop = Loop(c, p, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    workspaceMemory(loop.train)
    // generate training and testing file
    cudnn_gen.print(loop)

    // generate forward inference file
    val inf = Inference("resnet", 10, network(x1), x)
    cudnn_gen.print(inf)
  }

  @Test
  def testVgg {
    val N = 64; val C = 3;  val N1 = 224; val N2 = 224 // batch size, channel, and x/y size

    // Specifying train dataSet
    val y = Vec._new(imagenet, "label", "Y", N)
    val x = Vec._new(imagenet, "image", "X", N, C, N1, N2)

    val cv11 = CudaLayer.convolv("cv11", 3, 64, 1, 1)
    val cv12 = CudaLayer.convolv("cv12", 3, 64, 1, 1)
    val cv21 = CudaLayer.convolv("cv21", 3, 128, 1, 1)
    val cv22 = CudaLayer.convolv("cv22", 3, 128, 1, 1)
    val cv31 = CudaLayer.convolv("cv31", 3, 256, 1, 1)
    val cv32 = CudaLayer.convolv("cv32", 3, 256, 1, 1)
    val cv33 = CudaLayer.convolv("cv33", 3, 256, 1, 1)
    val cv41 = CudaLayer.convolv("cv41", 3, 512, 1, 1)
    val cv42 = CudaLayer.convolv("cv42", 3, 512, 1, 1)
    val cv43 = CudaLayer.convolv("cv43", 3, 512, 1, 1)
    val cv51 = CudaLayer.convolv("cv51", 3, 512, 1, 1)
    val cv52 = CudaLayer.convolv("cv52", 3, 512, 1, 1)
    val cv53 = CudaLayer.convolv("cv53", 3, 512, 1, 1)

    val full6 = Layer.full("fc6", 4096)
    val full7 = Layer.full("fc7", 4096)
    val full8 = Layer.full("fc8", K)

    val relu = CudaLayer.relu(4)
    val relu2 = CudaLayer.relu(2)
    val pool = CudaLayer.max_pool(2)
    val drop = CudaLayer.dropout(2, 0.5f)

    val flat = Layer.flatten(4, 1)
    val softmax = CudaLayer.log_softmax

    val network =  full8 o drop o relu2 o
      full7 o drop o relu2 o
      full6 o flat o
      pool o relu o cv53 o relu o cv52 o relu o cv51 o
      pool o relu o cv43 o relu o cv42 o relu o cv41 o
      pool o relu o cv33 o relu o cv32 o relu o cv31 o
      pool o relu o cv22 o relu o cv21 o
      pool o relu o cv12 o relu o cv11

    println(typeof(network))

    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda

    val c = Layer.loss(y1)((softmax o network)(x1))
    val p = Accuracy(network(x1), y, 1)         // top 1 accuracy 

    val param = c.freeVar.toList.sortWith((a,b) => a.toString < b.toString)
    val solver = Train("vgg", 1000, 10, 0.1f, 0f, 0.0005f, 0)
    val loop = Loop(c, p, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    workspaceMemory(loop.train)
    // generate training and testing file
    cudnn_gen.print(loop)

    // generate forward inference file
    val inf = Inference("vgg", 10, network(x1), x)
    cudnn_gen.print(inf)
  }

  @Test
  def testOverfeat {
    val N = 128; val C = 3;  val N1 = 224; val N2 = 224 // batch size, channel, and x/y size

    // Specifying train dataSet
    val y = Vec._new(imagenet, "label", "Y", N)
    val x = Vec._new(imagenet, "image", "X", N, C, N1, N2)

    val relu = CudaLayer.relu(4)
    val pool = CudaLayer.max_pool(2, 2, 0)

    val w = Param.xavier; val b02 = Param.const(0.2f, 2, 0)

    val cv1 = CudaLayer.convolv("cv1", 11, 96, 4, 0, w, b02)
    val cv2 = CudaLayer.convolv("cv2", 5, 256, 1, 2, w, b02)
    val cv3 = CudaLayer.convolv("cv3", 3, 512, 1, 1, w, b02)
    val cv4 = CudaLayer.convolv("cv4", 3, 1024, 1, 1, w, b02)
    val cv5 = CudaLayer.convolv("cv5", 3, 1024, 1, 1, w, b02)

    val full6 = Layer.full("fc6", 3072)
    val full7 = Layer.full("fc7", 4096)
    val full8 = Layer.full("fc8", K)
    val flat = Layer.flatten(4, 1)
    val softmax = CudaLayer.log_softmax

    val network = full8 o full7 o full6 o
      flat o
      pool o relu o cv5 o
      relu o cv4 o
      relu o cv3 o
      pool o relu o cv2 o
      pool o relu o cv1

    println(typeof(network))

    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda

    val c = Layer.loss(y1)((softmax o network)(x1))
    val p = Accuracy(network(x1), y, 1)         // top 1 accuracy 
    val param = c.freeVar.toList.sortWith((a,b) => a.toString < b.toString)

    val solver = Train("overfeat", 1000, 10, 0.01f, 0.9f, 0.0005f, 0)
    val loop = Loop(c, p, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    workspaceMemory(loop.train)
    // generate training and testing file
    cudnn_gen.print(loop)

    // generate forward inference file
    val inf = Inference("overfeat", 10, network(x1), x)
    cudnn_gen.print(inf)
  }

  @Test
  def testGooglenet {
    val N = 128; val C = 3;  val N1 = 224; val N2 = 224 // batch size, channel, and x/y size

    // Specifying train dataSet
    val y = Vec._new(imagenet, "label", "Y", N)
    val x = Vec._new(imagenet, "image", "X", N, C, N1, N2)
    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda
    val relu = CudaLayer.relu(4)
    val relu2 = CudaLayer.relu(2)
    val lrn = CudaLayer.lrn(5, 0.0001, 0.75)
    val drop = CudaLayer.dropout(4, 0.4f)
    val drop2 = CudaLayer.dropout(2, 0.7f)
    val pool = CudaLayer.max_pool(3, 2, 1)
    val ipool = CudaLayer.max_pool(3, 1, 1)
    val pool7 = CudaLayer.ave_pool(7, 1, 0) // average pooling 
    val bpool = CudaLayer.ave_pool(5, 3, 0)
    val flat = Layer.flatten(4, 1)
    val softmax_loss = Layer.loss(y1) o CudaLayer.log_softmax

    val w = Param.xavier; val b02 = Param.const(0.2f, 2, 0); val b0 = Param.const(0, 2, 0)

    val full7 = Layer.full("fc", K, w, b0)
    val cv1 = CudaLayer.convolv("cv1", 7, 64, 2, 3, w, b02)
    val cv2 = CudaLayer.convolv("cv2", 1, 64, 1, 0, w, b02)
    val cv3 = CudaLayer.convolv("cv3", 3, 192, 1, 1, w, b02)

    def inception(n: Int) = {
      val icv1 = CudaLayer.convolv(s"cv${n}1", 1, 64, 1, 0, w, b02)
      val icv2 = CudaLayer.convolv(s"cv${n}2", 1, 96, 1, 0, w, b02)
      val icv3 = CudaLayer.convolv(s"cv${n}3", 3, 128, 1, 1, w, b02)
      val icv4 = CudaLayer.convolv(s"cv${n}4", 1, 16, 1, 0, w, b02)
      val icv5 = CudaLayer.convolv(s"cv${n}5", 5, 32, 1, 2, w, b02)
      val icv6 = CudaLayer.convolv(s"cv${n}6", 1, 32, 1, 0, w, b02)

      val p = Vec._new(4)

      VecFun(p, CudaLayer.concat( (relu o icv1)(p),
        (relu o icv3 o relu o icv2)(p),
        (relu o icv5 o relu o icv4)(p),
        (relu o icv6 o ipool)(p) )
      )
    }

    val network3 = full7 o flat o drop o pool7 o inception(9) o inception(8) o pool o inception(7)
    val network2 = inception(6) o inception(5) o inception(4)
    val network1 = inception(3) o pool o inception(2) o inception(1) o
      pool o lrn o relu o cv3 o relu o cv2 o lrn o pool o relu o cv1

    def branch(n: Int) = {
      val cv = CudaLayer.convolv(s"b${n}cv", 1, 128, 1, 0, w, b02)
      val f1 = Layer.full(s"b${n}fc1", 1024, w, b02)
      val f2 = Layer.full(s"b${n}fc2", K, w, b0)
      f2 o drop2 o relu2 o f1 o flat o relu o cv o bpool
    }
    val stage2 = {
      val p = Vec._new(4)
      Vec2ScalarFun(p, (softmax_loss o network3)(p) + (softmax_loss o branch(2))(p) * Real(0.3f, "loss2"))
    }
    val stage1 = {
      val p = Vec._new(4)
      Vec2ScalarFun(p, (stage2 o network2)(p) + (softmax_loss o branch(1))(p) * Real(0.3f, "loss1"))
    }

    val c = (stage1 o network1)(x1)
    val p = Accuracy((network3 o network2 o network1)(x1), y, 1)         // top 1 accuracy
    val param = c.freeVar.toList.sortWith((a,b) => a.toString < b.toString)

    val solver = Train("googlenet", 1000, 10, 0.01f, 0.9f, 0.0005f, 0)
    val loop = Loop(c, p, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    workspaceMemory(loop.train)
    // generate training and testing file
    cudnn_gen.print(loop)

    // generate forward inference file
    val inf = Inference("googlenet", 10, (network3 o network2 o network1)(x1), x)
    cudnn_gen.print(inf)
  }

  @Test
  def testAlexnet {
    val N = 128; val C = 3;  val N1 = 224; val N2 = 224 // batch size, channel, and x/y size

    // Specifying train dataSet
    val y = Vec._new(imagenet, "label", "Y", N)
    val x = Vec._new(imagenet, "image", "X", N, C, N1, N2)

    val relu2 = CudaLayer.relu(2)
    val relu = CudaLayer.relu(4)
    val lrn = CudaLayer.lrn(5, 0.0001, 0.75)
    val drop = CudaLayer.dropout(2, 0.5f)
    val pool = CudaLayer.max_pool(3, 2, 0)
    val flat = Layer.flatten(4, 1)
    val softmax = CudaLayer.log_softmax

    val w001 = Param.gaussian(0.01f); val w0005 = Param.gaussian(0.005f)   // std, lr_mult=1, decay_mult=1
    val b0 = Param.const(0, 2, 0); val b01 = Param.const(0.1f, 2, 0)       // value, lr_mult, decay_mult

    val cv1 = CudaLayer.convolv("cv1", 11, 96, 4, 2, w001, b0)
    val cv2 = CudaLayer.convolv("cv2", 5, 256, 1, 2, w001, b01)
    val cv3 = CudaLayer.convolv("cv3", 3, 384, 1, 1, w001, b0)
    val cv4 = CudaLayer.convolv("cv4", 3, 384, 1, 1, w001, b01)
    val cv5 = CudaLayer.convolv("cv5", 3, 256, 1, 1, w001, b01)

    val full6 = Layer.full("fc6", 4096, w0005, b01)
    val full7 = Layer.full("fc7", 4096, w0005, b01)
    val full8 = Layer.full("fc8", K, w001, b0)

    val network =       full8 o
      drop o relu2 o full7 o
      drop o relu2 o full6 o flat o
      pool o relu o cv5 o
      relu o cv4 o
      relu o cv3 o
      pool o lrn o relu o cv2 o
      pool o lrn o relu o cv1

    println(typeof(network)) // type-check the network and print out its type (a tensor function type)

    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda

    val c = Layer.loss(y1)((softmax o network)(x1))
    val p = Accuracy(network(x1), y, 1)         // top 1 accuracy
    val param = c.freeVar.toList.sortWith((a,b) => a.toString < b.toString)

    val solver = Train("alexnet", 1000, 10, 0.01f, 0.9f, 0.0005f, 0)
    val loop = Loop(c, p, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    workspaceMemory(loop.train)
    // generate training and testing file
    cudnn_gen.print(loop)

    // generate forward inference file
    val inf = Inference("alexnet", 10, network(x1), x)
    cudnn_gen.print(inf)
  }


  @Test
  def testLenet {
    val K = 10 // # of classes 
    val N = 500; val C = 1; val N1 = 28; val N2 = 28 // batch size, channel, and x/y size

    // Specifying train dataSet
    val y = Vec._new(Mnist, "label", "Y", N)
    val x = Vec._new(Mnist, "image", "X", N, C, N1, N2)

    val cv1 = CudaLayer.convolv("cv1", 5, 20)
    val cv2 = CudaLayer.convolv("cv2", 5, 50)
    val mp = CudaLayer.max_pool(2)
    val flat = Layer.flatten(4, 1)
    val f = Layer.full("fc1", 500)
    val f2 = Layer.full("fc2", K)
    val softmax = CudaLayer.softmax
    val relu = CudaLayer.relu(2)

    val network = f2 o relu o f o flat o mp o cv2 o mp o cv1

    println(typeof(network))
    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda
    val c = (Layer.log_loss(y1) o softmax o network) (x1)
    val p = Accuracy(network(x1), y, 1)         // top 1 accuracy

    val param = c.freeVar.toList
    val solver = Train("lenet", 100, 10, 0.01f, 0.9f, 0.0005f, 0)

    val loop = Loop(c, p, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    workspaceMemory(loop.train)
    // generate training and testing file
    cudnn_gen.print(loop)

    // generate forward inference file
    val inf = Inference("lenet", 10, network(x1), x)
    cudnn_gen.print(inf)
  }

  // This test follows the Lenet example in Theano
  @Test
  def testLenet_tanh {
    val K1 = DimConst(5); val K2 = DimConst(5) // two convolution kernel x/y dimensions
    val d = 2; val D = DimConst(d) // down-sample rate
    val K = DimConst(10) // # of classes
    val C = DimConst(1); val F1 = DimConst(20); val F2 = DimConst(50) // # of feature maps
    val N = DimConst(500); val N1 = DimConst(28); val N2 = DimConst(28) // # of images and their x/y size
    val M = N; val M1 = Dim._new // fully connected layer
    val C1 = Dim._new; val C2 = Dim._new

    val fan_in = List(C1 * K1 * K1, C2 * K2 * K2, M1, M)
    val fan_out = List((F1 * K1 * K1)/D/D, (F2 * K2 * K2)/D/D, M, K)
    val init = (fan_in zip fan_out).map({case (x, y) => Xavier(x, y)})

    val k1 = Vec._new(init(0), "W1", F1, C1, K1, K1)
    val k2 = Vec._new(init(1), "W2", F2, C2, K2, K2)
    val w = Vec._new(init(2), "W", M, M1)
    val b = Vec._new(ConstInit(0f), "B", M)
    val b1 = Vec._new(ConstInit(0f), "B1", F1)
    val b2 = Vec._new(ConstInit(0f), "B2", F2)
    val b3 = Vec._new(ConstInit(0f), "B3", K)
    val theta = Vec._new(ConstInit(0f), "Theta", K, M)
    // Specifying train dataSet
    val y = Vec._new(Mnist, "label", "Y", List(N))
    val x = Vec._new(Mnist, "image", "X", List(N, C, N1, N2))

    val cv1 = CudaLayer.convolv(k1, b1)
    val cv2 = CudaLayer.convolv(k2, b2)
    val mp = CudaLayer.max_pool(d)
    val flat = Layer.flatten(4, 1)
    val f = Layer.full(w, b)
    val f2 = Layer.full(theta, b3)
    val softmax = CudaLayer.softmax
    val tanh2 = CudaLayer.tanh(2)
    val tanh = CudaLayer.tanh(4)

    val network = f2 o tanh2 o f o flat o  tanh o mp o cv2 o tanh o mp o cv1

    println(typeof(network))
    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda
    val c = Layer.log_loss(y1)((softmax o network)(x1))
    val p = Accuracy(network(x1), y, 1)         // top 1 accuracy

    val param = c.freeVar.toList
    val solver = Train("lenet_tanh", 1000, 10, 0.1f, 0f, 0f, 0)

    val loop = Loop(c, p, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    workspaceMemory(loop.train)
    cudnn_gen.print(loop)
  }

  private def workspaceMemory(lst: List[Let]) = {
    val convolutions =
      lst.foldLeft[Set[FixVec]](Set())((c, let) => let match {
        case VecLet(x, v @ FixVec(Convolv(_,_),_,_)) => c + v
        case _ => c
      })

    def getWorkspace(cv: FixVec) {
      val c = cv.layer.asInstanceOf[Convolv]
      val stride = c.stride; val padding = c.padding;
      val x_dim = cv.param(0).getDims.map(d=>env(d).size).toArray
      val w_dim = cv.param(1).getDims.map(d=>env(d).size).toArray
      val b_dim = cv.param(2).getDims.map(d=>env(d).size).toArray

      new deepdsl.cudnn.JCudnnConvolution(x_dim, w_dim, b_dim, stride, padding)
    }
    convolutions.foreach(cv => getWorkspace(cv))
    val mb = 1E6f
    println("Convolution workspace: " + deepdsl.cudnn.JCudaFunction.getWorkspaceSize()/mb)
  }

  private def parameterMemory(loop: Loop) {
    val mb = 1E6f
    val param = loop.param
    val size = param.map(e => e.dim.map(d => env(d).size).reduce(_*_) * 4 / mb)

    println
    for((v, s) <- param zip size) {
      printf("%-60s%20s\n", v, s)
    }

    val total = size.reduce(_+_)

    println(s"total parameter memory: ${total}")
    if(loop.solver.momentum > 0) {
      println(s"with SGD, this doubles to ${total * 2}")
    }
  }

  private def runtimeMemory(lst: List[Let]) {
    val sizes = memory(lst)._1
    var pool: Set[Long] = Set()
    val total_sizes = sizes.foldLeft[(List[(Long, Long)], Long, Long)]((Nil, 0L, 0L))((c, e) => {
      val x = c._2 + e; 
      if (e < 0) {
        pool = pool + (-e)
      }
      val y = if(e > 0) {
        if(pool.contains(e)) {
          pool = pool - e
          c._3
        }
        else 
          c._3 + e 
      }  
      else 
        c._3; 
      
      (c._1:::List((x,y)), x, y)}
      
    )._1

    val mb = 1E6f
    printf("\n%-60s%-15s%20s%20s%20s\n\n", "Statement", "Dimensions", "Current mem.", "Total mem.", "Accumulate mem.")
    for((l, s) <- lst.zip(sizes zip total_sizes)) {
      printf("%-60s%-15s%20f%20f%20f\n", l,
        (l match {case VecLet(x,_) => x.dim.map(d=>env(d).size).mkString(" ")
        case _ => ""}), s._1/mb, s._2._1/mb, s._2._2/mb)
    }
  }

}

