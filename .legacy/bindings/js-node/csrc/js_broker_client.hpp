#pragma once

#include <napi.h>

#include "broker_client.hpp"
#include "js_value_converters.hpp"
#include "uuid.hpp"

using namespace std::chrono_literals;

std::string packet_type_to_string(const ailoy::packet_type &ptype) {
  if (ptype == ailoy::packet_type::connect)
    return "connect";
  else if (ptype == ailoy::packet_type::disconnect)
    return "disconnect";
  else if (ptype == ailoy::packet_type::subscribe)
    return "subscribe";
  else if (ptype == ailoy::packet_type::unsubscribe)
    return "unsubscribe";
  else if (ptype == ailoy::packet_type::execute)
    return "execute";
  else if (ptype == ailoy::packet_type::respond)
    return "respond";
  else if (ptype == ailoy::packet_type::respond_execute)
    return "respond_execute";
  else
    throw ailoy::exception("invalid packet type");
}

std::string instruction_type_to_string(const ailoy::instruction_type &itype) {
  if (itype == ailoy::instruction_type::call_function)
    return "call_function";
  else if (itype == ailoy::instruction_type::define_component)
    return "define_component";
  else if (itype == ailoy::instruction_type::delete_component)
    return "delete_component";
  else if (itype == ailoy::instruction_type::call_method)
    return "call_method";
  else
    throw ailoy::exception("invalid instruction type");
}

class js_broker_listener_t : public Napi::AsyncWorker {
public:
  js_broker_listener_t(std::shared_ptr<ailoy::broker_client_t> client,
                       Napi::Promise::Deferred deferred, Napi::Env env)
      : Napi::AsyncWorker(env), client_(client), deferred_(deferred) {}

  void Execute() override { resp_ = client_->listen(ailoy::timeout_default); }

  void OnOK() override {
    Napi::Env env = Env();

    if (!resp_) {
      auto ret = ailoy::create<ailoy::null_t>();
      deferred_.Resolve(to_napi_value(env, ret));
      return;
    }

    auto ret = ailoy::create<ailoy::map_t>();
    ret->insert_or_assign(
        "packet_type",
        ailoy::create<ailoy::string_t>(packet_type_to_string(resp_->ptype)));

    if (resp_->itype.has_value()) {
      ret->insert_or_assign("instruction_type", ailoy::create<ailoy::string_t>(
                                                    instruction_type_to_string(
                                                        resp_->itype.value())));
    }

    ret->insert_or_assign("headers", resp_->headers);
    ret->insert_or_assign("body", resp_->body);

    deferred_.Resolve(to_napi_value(env, ret));
  }

  void OnError(const Napi::Error &e) override { deferred_.Reject(e.Value()); }

  Napi::Promise GetPromise() const { return deferred_.Promise(); }

private:
  std::shared_ptr<ailoy::broker_client_t> client_;
  Napi::Promise::Deferred deferred_;
  std::shared_ptr<ailoy::packet_t> resp_;
};

class js_broker_client_t : public Napi::ObjectWrap<js_broker_client_t> {
public:
  static Napi::FunctionReference constructor;

  static Napi::Function initialize(Napi::Env env) {
    auto ctor = DefineClass(
        env, "BrokerClient",
        {
            InstanceMethod("send_type1", &js_broker_client_t::send_type1),
            InstanceMethod("send_type2", &js_broker_client_t::send_type2),
            InstanceMethod("send_type3", &js_broker_client_t::send_type3),
            InstanceMethod("listen", &js_broker_client_t::listen),
        });
    constructor = Napi::Persistent(ctor);
    constructor.SuppressDestruct();
    return ctor;
  }

  js_broker_client_t(const Napi::CallbackInfo &info)
      : Napi::ObjectWrap<js_broker_client_t>(info) {
    auto env = info.Env();
    std::string url;
    if (info.Length() < 1) {
      url = "inproc://";
    } else {
      url = info[0].As<Napi::String>().Utf8Value();
    }
    inner_ = ailoy::create<ailoy::broker_client_t>(url);

    // // connect immediately
    // inner_->send<ailoy::packet_type::connect>(ailoy::generate_uuid());
    // inner_->listen(10ms);
  }

  // ~js_broker_client_t() {
  //   // disconnect before destroyed
  //   inner_->send<ailoy::packet_type::disconnect>(ailoy::generate_uuid());
  //   inner_->listen(10ms);
  // }

  template <typename derived_value_t>
    requires std::is_base_of_v<Napi::Value, derived_value_t>
  derived_value_t get_from_object(Napi::Object obj, const std::string &key) {
    auto env = obj.Env();
    if (!obj.Has(key)) {
      throw ailoy::exception("key '" + key + "' not found");
    }
    return obj.Get(key).As<derived_value_t>();
  }

  Napi::Value send_type1(const Napi::CallbackInfo &info) {
    auto env = info.Env();
    auto txid = info[0].As<Napi::String>().Utf8Value();
    auto ptype = info[1].As<Napi::String>().Utf8Value();
    bool ret;
    if (ptype == "connect")
      ret = inner_->send<ailoy::packet_type::connect>(txid);
    else if (ptype == "disconnect")
      ret = inner_->send<ailoy::packet_type::disconnect>(txid);
    else {
      Napi::Error::New(info.Env(), std::format("Unknown packet type {}", ptype))
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
    return Napi::Boolean::New(env, ret);
  }

  Napi::Value send_type2(const Napi::CallbackInfo &info) {
    auto env = info.Env();
    auto txid = info[0].As<Napi::String>().Utf8Value();
    auto ptype = info[1].As<Napi::String>().Utf8Value();
    auto itype = info[2].As<Napi::String>().Utf8Value();
    bool ret;
    if (itype == "call_function") {
      auto fname = info[3].As<Napi::String>().Utf8Value();
      if (ptype == "subscribe")
        ret = inner_->send<ailoy::packet_type::subscribe,
                           ailoy::instruction_type::call_function>(txid, fname);
      else if (ptype == "unsubscribe")
        ret = inner_->send<ailoy::packet_type::unsubscribe,
                           ailoy::instruction_type::call_function>(txid, fname);
      else { // if (ptype == "execute")
        auto in = from_napi_value(env, info[4]);
        ret = inner_->send<ailoy::packet_type::execute,
                           ailoy::instruction_type::call_function>(txid, fname,
                                                                   in);
      }
    } else if (itype == "define_component") {
      auto ctname = info[3].As<Napi::String>().Utf8Value();
      if (ptype == "subscribe")
        ret = inner_->send<ailoy::packet_type::subscribe,
                           ailoy::instruction_type::define_component>(txid,
                                                                      ctname);
      else if (ptype == "unsubscribe")
        ret = inner_->send<ailoy::packet_type::unsubscribe,
                           ailoy::instruction_type::define_component>(txid,
                                                                      ctname);
      else { // if (ptype == "execute")
        auto cname = info[4].As<Napi::String>().Utf8Value();
        auto in = from_napi_value(env, info[5]);
        ret = inner_->send<ailoy::packet_type::execute,
                           ailoy::instruction_type::define_component>(
            txid, ctname, cname, in);
      }
    } else if (itype == "delete_component") {
      auto cname = info[3].As<Napi::String>().Utf8Value();
      if (ptype == "subscribe")
        ret = inner_->send<ailoy::packet_type::subscribe,
                           ailoy::instruction_type::delete_component>(txid,
                                                                      cname);
      else if (ptype == "unsubscribe")
        ret = inner_->send<ailoy::packet_type::unsubscribe,
                           ailoy::instruction_type::delete_component>(txid,
                                                                      cname);
      else { // if (ptype == "execute")
        ret = inner_->send<ailoy::packet_type::execute,
                           ailoy::instruction_type::delete_component>(txid,
                                                                      cname);
      }
    } else if (itype == "call_method") {
      auto cname = info[3].As<Napi::String>().Utf8Value();
      auto fname = info[4].As<Napi::String>().Utf8Value();
      if (ptype == "subscribe")
        ret = inner_->send<ailoy::packet_type::subscribe,
                           ailoy::instruction_type::call_method>(txid, cname,
                                                                 fname);
      else if (ptype == "unsubscribe")
        ret = inner_->send<ailoy::packet_type::unsubscribe,
                           ailoy::instruction_type::call_method>(txid, cname,
                                                                 fname);
      else { // if (ptype == "execute")
        auto in = from_napi_value(env, info[5]);
        ret = inner_->send<ailoy::packet_type::execute,
                           ailoy::instruction_type::call_method>(txid, cname,
                                                                 fname, in);
      }
    }
    return Napi::Boolean::New(env, ret);
  }

  Napi::Value send_type3(const Napi::CallbackInfo &info) {
    auto env = info.Env();
    auto txid = info[0].As<Napi::String>().Utf8Value();
    auto ptype = info[1].As<Napi::String>().Utf8Value();
    auto status = info[2].As<Napi::Boolean>().Value();
    auto sequence = info[3].As<Napi::Number>().Uint32Value();
    bool ret;
    if (status) {
      auto done = info[4].As<Napi::Boolean>().Value();
      auto out = from_napi_value(env, info[5].As<Napi::Object>());
      ret = inner_->send<ailoy::packet_type::respond_execute, true>(
          txid, sequence, done, out);
    } else {
      auto reason = info[4].As<Napi::String>().Utf8Value();
      ret = inner_->send<ailoy::packet_type::respond_execute, false>(
          txid, sequence, reason);
    }
    return Napi::Boolean::New(env, ret);
  }

  Napi::Value listen(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    auto deferred = Napi::Promise::Deferred::New(env);
    // @jhlee: We don't need to delete this object, since Napi does it
    auto *listener = new js_broker_listener_t(inner_, deferred, env);
    listener->Queue();
    return deferred.Promise();
  }

private:
  std::shared_ptr<ailoy::broker_client_t> inner_;
};

Napi::FunctionReference js_broker_client_t::constructor;