class ModifiedLSTMCell(RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.

  The default non-peephole implementation is based on:

    http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

  The peephole implementation is based on:

    https://research.google.com/pubs/archive/43905.pdf

  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.

  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.
  """

  def __init__(self, initializer, use_peepholes=True, cell_clip=None, forget_bias=1.0, activation=tanh,is_training=True
               ):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The size of hidden vector in the LSTM cell
      input_size: Deprecated and unused.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_unit_shards: How to split the weight matrix.  If >1, the weight
        matrix is stored across num_unit_shards.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated.. Do we have c_state and m_state
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """


    self._forget_bias = forget_bias
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._activation = activation
    self._is_training = is_training

    #self._num_proj = num_proj
    #self._proj_clip = proj_clip
    #self._num_unit_shards = num_unit_shards
    #self._num_proj_shards = num_proj_shards
    # self._num_units = num_units
    #self._state_is_tuple = state_is_tuple



  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope=None):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
      scope: VariableScope for the created subgraph; defaults to "LSTMCell".

    Returns:
      A tuple containing:
      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    ls_internal={}
    #num_proj = self._num_units if self._num_proj is None else self._num_proj

    # if self._state_is_tuple:
    #   (c_prev, m_prev) = state
    # else:
    #   c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
    #   m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])
    # ls_internal["c_prev"]=c_prev
    # ls_internal["m_prev"]=m_prev

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    with vs.variable_scope(scope or type(self).__name__):  # "LSTMCell"

      x_size = inputs.get_shape().as_list()[1]
      W_xh = tf.get_variable('W_xh', [x_size, 4 * self._num_units],
                             initializer=self._initializer)
          # initializer=self._initializer)
      W_hh = tf.get_variable('W_hh',[self._num_units, 4 * self._num_units],
          initializer=orthogonal_initializer())
      bias = tf.get_variable('bias', [4 * self._num_units])

      xh = tf.matmul(inputs, W_xh)
      hh = tf.matmul(m_prev, W_hh)

      # bn_xh = batch_norm(xh, 'xh', self._is_training)
      # bn_hh = batch_norm(hh, 'hh', self._is_training)
      # hidden = bn_xh + bn_hh + bias
      hidden = xh + hh + bias

      i, j, f, o = array_ops.split(hidden,4, 1)
      # ls_internal["concat_w"]=concat_w
      # ls_internal["b"]=b
      # Diagonal connections

      f_s=sigmoid(f + self._forget_bias)
      i_s=sigmoid(i)
      j_t=tf.tanh(j)
      c = (f_s * c_prev + i_s * j_t)
      # bn_new_c = batch_norm(c, 'c', self._is_training)
      o_s=sigmoid(o)
      # act_c=tf.tanh(bn_new_c)
      act_c=tf.tanh(c)
      m = o_s *act_c
      ls_internal["i"]=i_s
      ls_internal["j"]=j_t
      ls_internal["f"]=f_s
      # ls_internal["c"]=bn_new_c
      ls_internal["m"]=m
      ls_internal["act_c"]=act_c
      ls_internal["o"]=o_s

    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple
                 else array_ops.concat(1, [c, m]))
    return m, new_state,ls_internal