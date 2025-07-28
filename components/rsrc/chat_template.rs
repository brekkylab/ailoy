use std::sync::{Mutex, MutexGuard, OnceLock};

use minijinja::Environment;

use crate::message::Message;

static ENV: OnceLock<Mutex<Environment>> = OnceLock::new();

fn get_env<'a>() -> MutexGuard<'a, Environment<'static>> {
    ENV.get_or_init(|| Mutex::new(Environment::new()))
        .lock()
        .unwrap()
}

pub fn add_chat_template(name: String, source: String) {
    let _ = get_env().add_template_owned(name, source);
}

pub fn get_chat_template(name: &str) -> Option<String> {
    match get_env().get_template(name) {
        Ok(v) => Some(v.source().to_owned()),
        Err(_) => None,
    }
}

pub fn remove_chat_template(name: &str) {
    get_env().remove_template(name);
}

pub fn apply_chat_template(name: &str, context: &Vec<Message>) -> String {
    use minijinja::Value;

    let env = get_env();

    let template = env.get_template(name).unwrap();
    let ctx = Value::from_serialize(context);
    template.render(ctx).unwrap()
}

mod ffi {
    use std::ffi::{c_char, c_int};

    use super::*;
    use crate::ffi::{from_const_char, to_char};
    use crate::message::Message;

    #[unsafe(no_mangle)]
    pub extern "C" fn ailoy_add_chat_template(name: *const c_char, source: *const c_char) -> c_int {
        let name = match from_const_char(name) {
            Ok(v) => v,
            Err(_) => return 1,
        };
        let source = match from_const_char(source) {
            Ok(v) => v,
            Err(_) => return 1,
        };
        add_chat_template(name.to_owned(), source.to_owned());
        0
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn ailoy_remove_chat_template(name: *const c_char) -> c_int {
        let name = match from_const_char(name) {
            Ok(v) => v,
            Err(_) => return 1,
        };
        remove_chat_template(name);
        0
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn ailoy_get_chat_template(
        name: *const c_char,
        source: *mut *mut c_char,
    ) -> c_int {
        let name = match from_const_char(name) {
            Ok(v) => v,
            Err(_) => return 1,
        };
        let tmpl = get_chat_template(name);
        match tmpl {
            Some(tmpl) => {
                unsafe { *source = to_char(tmpl.as_str()) };
            }
            None => return 2,
        };
        0
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn ailoy_apply_chat_template(
        name: *const c_char,
        context: *const c_char,
        out: *mut *mut c_char,
    ) -> c_int {
        let name = match from_const_char(name) {
            Ok(v) => v,
            Err(_) => return 1,
        };
        let context = match from_const_char(context) {
            Ok(v) => v,
            Err(_) => return 1,
        };
        let context: Vec<Message> = serde_json::from_str(context).unwrap();
        let rendered = apply_chat_template(name, &context);
        unsafe { *out = to_char(rendered.as_str()) };
        0
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_apply_chat_template() {
        // apply_chat_template();
    }
}
