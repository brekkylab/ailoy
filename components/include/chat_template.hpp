#include "message.hpp"

extern "C" {

struct ailoy_chat_template_t;

/**
 * Real implementation in rust
 */
extern int ailoy_chat_template_create(const char *, ailoy_chat_template_t **);

/**
 * Real implementation in rust
 */
extern int ailoy_chat_template_destroy(const ailoy_chat_template_t *);

/**
 * Real implementation in rust
 */
extern int ailoy_chat_template_get(const ailoy_chat_template_t *,
                                   char **source);

/**
 * Real implementation in rust
 * context must be JSON-serialized message_t
 */
extern int ailoy_chat_template_apply(const ailoy_chat_template_t *,
                                     const char *messages, char **out);
}
