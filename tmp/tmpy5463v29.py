# coding=utf-8
from __future__ import absolute_import, division, print_function



def create_converted_entity_factory():
  loss = None
  self = None

  def create_converted_entity(ag__, ag_source_map__, ag_module__):
    tf__lambda = lambda : ag__.with_function_scope(lambda lscope: ag__.if_stmt(self.trainable, lambda : ag__.converted_call(loss, (), None, lscope), lambda : 0.0, lambda : (), lambda _: None), 'lscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True))
    tf__lambda.ag_source_map = ag_source_map__
    tf__lambda.ag_module = ag_module__
    tf__lambda.autograph_info__ = {}
    return tf__lambda
  return create_converted_entity
