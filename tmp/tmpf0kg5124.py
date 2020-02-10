# coding=utf-8
from __future__ import absolute_import, division, print_function



def create_converted_entity_factory():
  name = None
  regularizer = None

  def create_converted_entity(ag__, ag_source_map__, ag_module__):

    def tf___loss_for_variable(v):
      """Creates a regularization loss `Tensor` for variable `v`."""
      do_return = False
      retval_ = ag__.UndefinedReturnValue()
      with ag__.FunctionScope('_loss_for_variable', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
        with backend.name_scope(name + '/Regularizer'):
          regularization = ag__.converted_call(regularizer, (v,), None, fscope)
        do_return = True
        retval_ = fscope.mark_return_value(regularization)
      do_return,
      return ag__.retval(retval_)
    tf___loss_for_variable.ag_source_map = ag_source_map__
    tf___loss_for_variable.ag_module = ag_module__
    tf___loss_for_variable.autograph_info__ = {}
    return tf___loss_for_variable
  return create_converted_entity
