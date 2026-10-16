"""Evaluation-safe auxiliary decorators, with the caller's live relay scope."""
import functools


def guarded_call(callback, scope):
    @functools.wraps(callback)
    def wrapped(*args, **kwargs):
        from agent.evaluation_policy import refuse_effect
        refuse_effect('auxiliary generation')
        with scope(args, kwargs):
            return callback(*args, **kwargs)
    return wrapped


def guarded_call_async(callback, scope):
    @functools.wraps(callback)
    async def wrapped(*args, **kwargs):
        from agent.evaluation_policy import refuse_effect
        refuse_effect('auxiliary generation')
        with scope(args, kwargs):
            return await callback(*args, **kwargs)
    return wrapped
