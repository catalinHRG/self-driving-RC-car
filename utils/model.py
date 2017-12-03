import tensorflow as tf

def fully_connected_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.nn.relu):
 
  with tf.name_scope(layer_name):

    with tf.name_scope('weights'):

      weights = tf.Variable( tf.truncated_normal( [input_dim, output_dim], dtype = 'float'), name = 'weights')
      tf.summary.histogram('weight vector distribution', weights)

    with tf.name_scope('biases'):
      
      biases = tf.Variable( tf.truncated_normal( [output_dim], dtype = 'float'), name = 'biases')
      tf.summary.histogram('biases vector distribution', biases)

    with tf.name_scope('Wx_plus_b'):
      
      print input_tensor.shape
      print weights.shape
      print biases.shape

      preactivate = tf.add( tf.matmul( input_tensor, weights ), biases)
      tf.summary.histogram('pre_activations', preactivate)

    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    
    return activations


def build_accuracy_node(output, target, op_name) :

  with tf.name_scope(op_name) :

    accuracy_node = tf.reduce_mean( tf.cast( tf.equal( tf.argmax(output), tf.argmax(target) ), dtype = tf.int32), name = op_name )

    tf.summary.scalar('accuracy curve', accuracy_node)
    tf.summary.histogram('accuracy vector distribution', accuracy_node)

    return accuracy_node

def build_cost_node(output, target, op_name) :

  with tf.name_scope(op_name) :

    cost_node = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = target, logits = output, name = op_name) )
    tf.summary.scalar('cost curve', cost_node)
    tf.summary.histogram('cost distribution', cost_node)

  return cost_node
