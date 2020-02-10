# coding=utf-8
from __future__ import absolute_import, division, print_function



def create_converted_entity_factory():
  signature_function = None
  signature_key = None

  def create_converted_entity(ag__, ag_source_map__, ag_module__):

    def tf__signature_wrapper(**kwargs):
      do_return = False
      retval_ = ag__.UndefinedReturnValue()
      with ag__.FunctionScope('signature_wrapper', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
        structured_outputs = ag__.converted_call(signature_function, (), dict(**kwargs), fscope)
        do_return = True
        retval_ = fscope.mark_return_value(ag__.converted_call(_normalize_outputs, (structured_outputs, signature_function.name, signature_key), None, fscope))
      do_return,
      return ag__.retval(retval_)
    tf__signature_wrapper.ag_source_map = ag_source_map__
    tf__signature_wrapper.ag_module = ag_module__
    tf__signature_wrapper.autograph_info__ = {}
    return tf__signature_wrapper
  return create_converted_entity
