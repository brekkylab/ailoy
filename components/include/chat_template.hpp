#include "message.hpp"

extern "C" {

/**
 * Real implementation in rust
 */
extern int ailoy_add_chat_template(const char *name, const char *source);

/**
 * Real implementation in rust
 */
extern int ailoy_remove_chat_template(const char *name);

/**
 * Real implementation in rust
 */
extern int ailoy_get_chat_template(const char *name, char **source);

/**
 * Real implementation in rust
 * context must be JSON-serialized message_t
 */
extern int ailoy_apply_chat_template(const char *name, const char *context,
                                     char **out);
}
