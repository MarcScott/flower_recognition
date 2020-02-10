# coding=utf-8
from __future__ import absolute_import, division, print_function



def create_converted_entity_factory():

  def create_converted_entity(ag__, ag_source_map__, ag_module__):

    def tf__call(self, inputs, training=None):
      do_return = False
      retval_ = ag__.UndefinedReturnValue()
      with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
        args = []
        kwargs = ag__.converted_call(self._arguments.copy, (), None, fscope)

        def get_state():
          return ()

        def set_state(_):
          pass

        def if_true():
          ag__.converted_call(kwargs.update, (inputs,), None, fscope)
          return ag__.match_staging_level(1, cond)

        def if_false():
          ag__.converted_call(args.append, (inputs,), None, fscope)
          return ag__.match_staging_level(1, cond)
        cond = ag__.and_(lambda : self._signature, lambda : ag__.converted_call(isinstance, (inputs, dict), None, fscope))
        ag__.if_stmt(cond, if_true, if_false, get_state, set_state, (), ())
        f = ag__.converted_call(functools.partial, (self._callable,) + tuple(args), dict(**kwargs), fscope)

        def get_state_3():
          return ()

        def set_state_3(_):
          pass

        def if_true_3():
          result = ag__.converted_call(f, (), None, fscope)
          return result

        def if_false_3():
          training_2, = training,

          def get_state_2():
            return ()

          def set_state_2(_):
            pass

          def if_true_2():
            training_1, = training_2,

            def get_state_1():
              return ()

            def set_state_1(_):
              pass

            def if_true_1():
              training_1 = ag__.converted_call(tf.keras.backend.learning_phase, (), None, fscope)
              return training_1

            def if_false_1():
              return training_1
            cond_1 = training_1 is None
            training_1 = ag__.if_stmt(cond_1, if_true_1, if_false_1, get_state_1, set_state_1, ('training',), ())
            return training_1

          def if_false_2():
            training_2 = False
            return training_2
          cond_2 = self.trainable
          training_2 = ag__.if_stmt(cond_2, if_true_2, if_false_2, get_state_2, set_state_2, ('training',), ())
          result = ag__.converted_call(smart_cond.smart_cond, (training_2, lambda : ag__.converted_call(f, (), dict(training=True), fscope), lambda : ag__.converted_call(f, (), dict(training=False), fscope)), None, fscope)
          return result
        cond_3 = ag__.not_(self._has_training_argument)
        result = ag__.if_stmt(cond_3, if_true_3, if_false_3, get_state_3, set_state_3, ('result',), ())

        def get_state_6():
          return ()

        def set_state_6(_):
          pass

        def if_true_6():
          result_1, = result,

          def get_state_4():
            return ()

          def set_state_4(_):
            pass

          def if_true_4():
            raise ag__.converted_call(ValueError, ('Specifying `output_key` is forbidden if output type %s is not a dict.' % ag__.converted_call(type, (result_1,), None, fscope),), None, fscope)
            return ag__.match_staging_level(1, cond_4)

          def if_false_4():
            return ag__.match_staging_level(1, cond_4)
          cond_4 = ag__.not_(ag__.converted_call(isinstance, (result_1, dict), None, fscope))
          ag__.if_stmt(cond_4, if_true_4, if_false_4, get_state_4, set_state_4, (), ())

          def get_state_5():
            return ()

          def set_state_5(_):
            pass

          def if_true_5():
            raise ag__.converted_call(ValueError, ('KerasLayer output does not contain the output key %s (available: %s).' % (self._output_key, ag__.converted_call(result_1.keys, (), None, fscope)),), None, fscope)
            return ag__.match_staging_level(1, cond_5)

          def if_false_5():
            return ag__.match_staging_level(1, cond_5)
          cond_5 = self._output_key not in result_1
          ag__.if_stmt(cond_5, if_true_5, if_false_5, get_state_5, set_state_5, (), ())
          result_1 = result_1[self._output_key]
          return result_1

        def if_false_6():
          return result
        cond_6 = self._output_key
        result = ag__.if_stmt(cond_6, if_true_6, if_false_6, get_state_6, set_state_6, ('result',), ())
        result = ag__.converted_call(self._apply_output_shape_if_set, (inputs, result), None, fscope)
        do_return = True
        retval_ = fscope.mark_return_value(result)
      do_return,
      return ag__.retval(retval_)
    tf__call.ag_source_map = ag_source_map__
    tf__call.ag_module = ag_module__
    tf__call.autograph_info__ = {}
    return tf__call
  return create_converted_entity
