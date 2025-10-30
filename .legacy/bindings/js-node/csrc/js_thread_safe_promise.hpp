#pragma once

#include <functional>
#include <memory>
#include <thread>

#include <napi.h>

#include "js_value_converters.hpp"

// promise function types
typedef std::function<void(std::shared_ptr<ailoy::value_t> val)> resolve_fn_t;
typedef std::function<void(std::shared_ptr<ailoy::value_t> val)> reject_fn_t;

// type of function type which will be passed promise functions
typedef std::function<void(resolve_fn_t, reject_fn_t)> promise_fn_t;

struct tsfn_context_t {
  Napi::Promise::Deferred deferred;
  std::shared_ptr<ailoy::value_t> data;
  bool resolve;
  bool called;
  Napi::ThreadSafeFunction tsfn;
  tsfn_context_t(Napi::Env env)
      : deferred{Napi::Promise::Deferred::New(env)}, resolve{false},
        called{false} {}
};

Napi::Promise thread_safe_promise_wrapper(const Napi::Env env,
                                          const promise_fn_t &prom_func) {
  std::shared_ptr<tsfn_context_t> context =
      std::make_shared<tsfn_context_t>(env);
  std::shared_ptr<std::mutex> mtx = std::make_shared<std::mutex>();
  context->tsfn = Napi::ThreadSafeFunction::New(
      env, Napi::Function::New(env, [](const Napi::CallbackInfo &info) {}),
      "TSFN", 0, 1, [context, mtx](Napi::Env env) {
        mtx->lock();
        if (context->resolve) {
          context->deferred.Resolve(to_napi_value(env, context->data));
        } else {
          context->deferred.Reject(to_napi_value(env, context->data));
        }
        mtx->unlock();
      });

  resolve_fn_t resolve = [context, mtx](std::shared_ptr<ailoy::value_t> val) {
    mtx->lock();
    context->data = val;
    context->resolve = true;
    if (!context->called) {
      context->called = true;
      context->tsfn.Release();
    }
    mtx->unlock();
  };

  reject_fn_t reject = [context, mtx](std::shared_ptr<ailoy::value_t> val) {
    mtx->lock();
    context->data = val;
    context->resolve = false;
    if (!context->called) {
      context->called = true;
      context->tsfn.Release();
    }
    mtx->unlock();
  };

  prom_func(resolve, reject);

  return context->deferred.Promise();
}
