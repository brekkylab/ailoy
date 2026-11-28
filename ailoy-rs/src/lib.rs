//! Agents that drive a language model through tool-augmented turns, in a virtx console.
//!
//! An [`Agent`] is reached over HTTP, from a server in this process that serves the
//! [`default_provider`](agent::default_provider): its model and tools are made there, and its
//! turns run there. What is set on that provider, a tool of this process or a model of its own,
//! is what the agent gets. The data an agent is made of and says is `ailoy-core`'s, shown here
//! at the same paths as `ailoy-runner`'s modules.

pub use ailoy_core::Error;
use ailoy_core::{
    AdminClient, Client,
    agent::{AgentPutArgs, AgentSpec, SessionCreateArgs},
    message::{Message, MessageDeltaOutput, MessageOutput, into_messages},
};
pub use ailoy_runner::{agent, console, datatype, message, model, to_value, tool, tool_func};
use futures::{Stream, StreamExt as _};

/// An agent of a spec, with one session: what it says and does in each turn, and the history it
/// keeps between turns, are the server's.
pub struct Agent {
    admin: AdminClient,
    client: Client,
    /// The name the agent is kept under, made for it.
    name: String,
    /// Its session's id.
    session: String,
}

impl Agent {
    /// An agent of `spec`, whose session starts with `history` and works in `sandbox`, which
    /// stays the caller's. Refused when the spec names a model or a tool nothing provides. Must
    /// be called inside a tokio runtime.
    pub async fn from_spec(
        spec: AgentSpec,
        history: Vec<Message>,
        sandbox: Option<&virtx::Sandbox>,
    ) -> Result<Self, Error> {
        let provider = agent::default_provider().clone();
        let virtx = sandbox.map(|sandbox| sandbox.client().clone());
        let admin = AdminClient::embed(ailoy_runner::admin_router(provider.clone()));
        let client = Client::embed(ailoy_runner::router(provider, virtx));

        let name = uuid::Uuid::new_v4().to_string();
        admin.put_agent(&name, &AgentPutArgs::new(spec)).await?;
        let mut args = SessionCreateArgs::new().history(history);
        if let Some(sandbox) = sandbox {
            args = args.sandbox_id(sandbox.id());
        }
        let session = match client.create_session(&name, &args).await {
            Ok(session) => session.id,
            Err(e) => {
                admin.remove_agent(&name).await.ok();
                return Err(e);
            }
        };
        Ok(Self {
            admin,
            client,
            name,
            session,
        })
    }

    /// Run one turn on `query`, and stream what it produces: the model's output as it is
    /// generated, and each tool's result whole, one delta each. One turn at a time: another
    /// while one runs is refused.
    pub fn run_stream(
        &self,
        query: &Message,
    ) -> impl Stream<Item = Result<MessageDeltaOutput, Error>> + Send + '_ {
        self.client.run_stream(&self.name, &self.session, query)
    }

    /// Run one turn on `query`, and stream each message it produces whole, the model's and each
    /// tool's.
    pub fn run(
        &self,
        query: &Message,
    ) -> impl Stream<Item = anyhow::Result<MessageOutput>> + Send + '_ {
        into_messages(self.run_stream(query).map(|delta| Ok(delta?)))
    }

    /// The conversation so far.
    pub async fn history(&self) -> Result<Vec<Message>, Error> {
        let session = self.client.get_session(&self.name, &self.session).await?;
        Ok(session.history)
    }

    /// Let go of the agent and its session, on the server. Its sandbox stays the caller's.
    pub async fn close(&self) -> Result<(), Error> {
        self.client
            .remove_session(&self.name, &self.session)
            .await?;
        self.admin.remove_agent(&self.name).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ailoy_core::{
        message::{FinishReason, Part, Role},
        model::{InferLangModel, LangModelOptions},
        tool::ToolDesc,
    };
    use futures::{future::BoxFuture, stream::BoxStream};

    use super::*;

    /// Answers `pong` whatever it is asked.
    struct Parrot;

    impl InferLangModel for Parrot {
        fn infer(
            &self,
            _: &[Message],
            _: &[ToolDesc],
            _: &LangModelOptions,
        ) -> BoxFuture<'static, anyhow::Result<MessageOutput>> {
            unimplemented!("a turn streams")
        }

        fn infer_stream(
            &self,
            _: &[Message],
            _: &[ToolDesc],
            _: &LangModelOptions,
        ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
            let output = MessageOutput {
                message: Message::new(Role::Assistant).with_contents([Part::text("pong")]),
                finish_reason: FinishReason::Stop {},
                usage: None,
            };
            Box::pin(futures::stream::once(async move { Ok(output.into()) }))
        }
    }

    #[tokio::test]
    async fn test_an_agent_runs_on_this_process_s_provider_and_keeps_its_history() {
        agent::default_provider().write().unwrap().models.set(
            "facade-parrot",
            |_: &str, _: Option<&str>| -> anyhow::Result<Arc<dyn InferLangModel>> {
                Ok(Arc::new(Parrot))
            },
        );
        let spec = AgentSpec::new("facade-parrot/x").instruction("Be brief.");
        let agent = Agent::from_spec(spec, Vec::new(), None).await.unwrap();

        let ping = Message::new(Role::User).with_contents([Part::text("ping")]);
        let messages: Vec<_> = agent.run(&ping).collect().await;
        let messages: Vec<_> = messages.into_iter().map(Result::unwrap).collect();
        assert_eq!(messages[0].message.contents[0].as_text(), Some("pong"));

        let roles: Vec<Role> = agent
            .history()
            .await
            .unwrap()
            .into_iter()
            .map(|message| message.role)
            .collect();
        assert_eq!(roles, [Role::System, Role::User, Role::Assistant]);

        agent.close().await.unwrap();
        assert!(agent.history().await.is_err());
    }

    #[tokio::test]
    async fn test_an_agent_of_a_model_nothing_makes_is_refused() {
        let spec = AgentSpec::new("nobody/x");
        let error = Agent::from_spec(spec, Vec::new(), None)
            .await
            .err()
            .unwrap();
        assert_eq!(error.code(), Some("unknown_model"));
    }
}
