-- When the tool call a `tool` row answers began: from the moment the model named it — while
-- it was still writing the arguments — or, failing that, when it started running. The row's
-- own `created_at` is when it finished, so the two give the time a call took after a reload,
-- which the live run's clock does not survive. NULL on every other row, and on tool rows
-- written before this column existed.
ALTER TABLE messages ADD COLUMN started_at INTEGER;
