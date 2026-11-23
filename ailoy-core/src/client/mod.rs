//! The two clients of an Ailoy server: [`Client`], for the code that runs an agent the server
//! keeps, and [`AdminClient`], for the code that manages which agents it keeps. Each reaches a
//! server in this process (`embed`, the `embed` feature), a child process (`spawn`) or one at a
//! URL (`connect`).

mod admin;
mod user;

pub use admin::AdminClient;
pub use user::Client;
