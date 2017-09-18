using TensorFlow; tf = TensorFlow

sess = tf.Session()

x = tf.constant(Float64[1,2])
y = tf.Variable(Float64[3,4])
z = tf.placeholder(Float64)

w = exp(x + z + -y)

run(sess, tf.global_variables_initializer())
res = run(sess, w, Dict(z=>Float64[1,2]))

println(res[1] - exp(-1))
