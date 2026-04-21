use std::sync::Arc;

use crate::{message::ToolDesc, tool::ToolFunc};

#[derive(Clone)]
pub struct Tool {
    desc: ToolDesc,
    f: Arc<ToolFunc>,
}

impl Tool {
    pub(crate) fn new(desc: ToolDesc, f: Arc<ToolFunc>) -> Self {
        Self { desc, f }
    }

    pub fn get_desc(&self) -> &ToolDesc {
        &self.desc
    }

    pub fn get_func(&self) -> Arc<ToolFunc> {
        self.f.clone()
    }
}

#[cfg(test)]
mod test {
    use anyhow::anyhow;
    use futures::StreamExt as _;

    use super::Tool;

    impl Tool {
        // Helper for test only
        pub(crate) async fn call(
            &self,
            args: &crate::message::Part,
        ) -> anyhow::Result<crate::message::Message> {
            Ok(self
                .get_func()
                .call(args.clone())?
                .next()
                .await
                .ok_or(anyhow!("tool function returned nothing"))?
                .message)
        }
    }
}
