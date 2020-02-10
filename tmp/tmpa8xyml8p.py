# coding=utf-8
from __future__ import absolute_import, division, print_function



def create_converted_entity_factory():

  def create_converted_entity(ag__, ag_source_map__, ag_module__):

    def tf___apply_output_shape_if_set(self, inputs, outputs):
      do_return = False
      retval_ = ag__.UndefinedReturnValue()
      with ag__.FunctionScope('_apply_output_shape_if_set', 'fscope', ag__.STD) as fscope:

        def get_state():
          return ()

        def set_state(_):
          pass

        def if_true():
          do_return = True
          retval_ = fscope.mark_return_value(outputs)
          return do_return, retval_

        def if_false():
          output_shape = ag__.converted_call(getattr, (self, '_output_shape'), None, fscope)
          batch_size = ag__.converted_call(tf.nest.flatten, (inputs,), None, fscope)[0].shape[0]

          @ag__.do_not_convert_internal
          def _inplace_set_shape(tensor, shape):
            with ag__.FunctionScope('_inplace_set_shape', 'fscope_1', ag__.STD) as fscope_1:
              ag__.converted_call(tensor.set_shape, (ag__.converted_call(ag__.converted_call(tf.TensorShape, (batch_size,), None, fscope_1).concatenate, (shape,), None, fscope_1),), None, fscope_1)
          ag__.converted_call(tf.nest.map_structure, (_inplace_set_shape, outputs, output_shape), None, fscope)
          do_return = True
          retval_ = fscope.mark_return_value(outputs)
          return do_return, retval_
        cond = ag__.not_(ag__.converted_call(hasattr, (self, '_output_shape'), None, fscope))
        do_return, retval_ = ag__.if_stmt(cond, if_true, if_false, get_state, set_state, ('do_return', 'retval_'), ())
      do_return,
      return ag__.retval(retval_)
    tf___apply_output_shape_if_set.ag_source_map = ag_source_map__
    tf___apply_output_shape_if_set.ag_module = ag_module__
    tf___apply_output_shape_if_set.autograph_info__ = {}
    return tf___apply_output_shape_if_set
  return create_converted_entity
