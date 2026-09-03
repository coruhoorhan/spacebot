//! Set status tool for workers.

use crate::conversation::{ProcessRunLogger, WorkerLifecycle, WorkerTransitionResult};
use crate::{AgentId, ChannelId, ProcessEvent, WorkerId};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

/// Tool for setting worker status.
#[derive(Debug, Clone)]
pub struct SetStatusTool {
    agent_id: AgentId,
    worker_id: WorkerId,
    channel_id: Option<ChannelId>,
    event_tx: broadcast::Sender<ProcessEvent>,
    process_run_logger: ProcessRunLogger,
    interactive: bool,
    /// Tool secret pairs for scrubbing status text before it reaches the channel.
    tool_secret_pairs: Vec<(String, String)>,
}

impl SetStatusTool {
    /// Create a new set status tool.
    pub fn new(
        agent_id: AgentId,
        worker_id: WorkerId,
        channel_id: Option<ChannelId>,
        event_tx: broadcast::Sender<ProcessEvent>,
        process_run_logger: ProcessRunLogger,
        interactive: bool,
    ) -> Self {
        Self {
            agent_id,
            worker_id,
            channel_id,
            event_tx,
            process_run_logger,
            interactive,
            tool_secret_pairs: Vec::new(),
        }
    }

    /// Set tool secret pairs for output scrubbing.
    pub fn with_tool_secrets(mut self, pairs: Vec<(String, String)>) -> Self {
        self.tool_secret_pairs = pairs;
        self
    }
}

/// Error type for set status tool.
#[derive(Debug, thiserror::Error)]
#[error("Failed to set status: {0}")]
pub struct SetStatusError(String);

/// The kind of status update.
///
/// `progress` (default) reports intermediate progress. `outcome` signals that
/// the worker has reached a terminal result — the task is done (or failed in a
/// way the worker can describe). Workers **must** emit an `outcome` status
/// before finishing; the system will nudge them back to work if they try to
/// stop without one.
///
/// NOTE: The outcome gate only checks *whether* an outcome was signaled, not
/// *whether all task steps are actually complete*. Premature outcome signaling
/// (e.g. after 2 of 7 steps) is handled via prompt-level instructions, not
/// structural enforcement. See the worker prompt for the anti-premature-exit
/// language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StatusKind {
    /// Intermediate progress update (default).
    #[default]
    Progress,
    /// Terminal outcome — the task is complete or has a definitive result.
    Outcome,
}

/// Structured verification evidence carried by an `outcome` status.
///
/// "No test evidence, no done": an outcome is only accepted when it
/// documents what was actually run — at least one item with a command and
/// its exit code. A failed task documents its failing command; a successful
/// task documents the passing run (e.g. `cargo test --lib`, exit 0).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Evidence {
    /// The command that produced the evidence (e.g. `cargo test --lib`).
    pub command: String,
    /// The command's exit code (0 = success).
    pub exit_code: i32,
    /// Optional human-readable summary (e.g. "42 passed, 0 failed").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Arguments for set status tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetStatusArgs {
    /// The status message to report.
    pub status: String,
    /// The kind of status update: "progress" (default) for intermediate
    /// updates, "outcome" when the task has reached a terminal result.
    #[serde(default)]
    pub kind: StatusKind,
    /// Verification evidence (command + exit code) for an `outcome` status.
    /// Required when `kind` is `outcome`; ignored for `progress`.
    #[serde(default)]
    pub evidence: Option<Vec<Evidence>>,
}

/// Output from set status tool.
#[derive(Debug, Serialize)]
pub struct SetStatusOutput {
    /// Whether the status was set successfully.
    pub success: bool,
    /// The worker ID.
    pub worker_id: WorkerId,
    /// The status that was set.
    pub status: String,
    /// Full outcome text when `kind` is `outcome`, uncapped so the worker's
    /// terminal result survives into the durable completion record. Absent
    /// for progress updates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    /// The kind of status that was set.
    pub kind: StatusKind,
    /// The evidence carried by this outcome (present when `kind` is
    /// `outcome`), so callers can verify the completion gate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Vec<Evidence>>,
}

impl Tool for SetStatusTool {
    const NAME: &'static str = "set_status";

    type Error = SetStatusError;
    type Args = SetStatusArgs;
    type Output = SetStatusOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: crate::prompts::text::get("tools/set_status").to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "A concise status message describing your current progress or final result (1-2 sentences)"
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["progress", "outcome"],
                        "default": "progress",
                        "description": "Use \"progress\" for intermediate updates. Use \"outcome\" ONLY when ALL steps of the task have reached a terminal result (success or failure) and you are ready to finish. Do not signal outcome if there are remaining steps — premature outcome signaling causes the task to be incorrectly reported as complete."
                    },
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "command": { "type": "string", "description": "The command that produced the evidence" },
                                "exit_code": { "type": "integer", "description": "The command's exit code (0 = success)" },
                                "summary": { "type": "string", "description": "Optional summary, e.g. '42 tests passed'" }
                            },
                            "required": ["command", "exit_code"]
                        },
                        "description": "REQUIRED for kind=\"outcome\": at least one item documenting the command you ran and its exit code (e.g. cargo test --lib with exit_code 0). Text-only outcomes are rejected."
                    }
                },
                "required": ["status"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Scrub tool secret values before the status leaves the worker.
        // Layer 1: exact-match redaction of known secrets from the store.
        // Layer 2: regex-based redaction of unknown secret patterns.
        // Scrubbing runs on the full text so the display cap below can't
        // truncate a secret out of exact-match range.
        let scrubbed = crate::secrets::scrub::scrub_secrets(&args.status, &self.tool_secret_pairs);
        let scrubbed = crate::secrets::scrub::scrub_leaks(&scrubbed);

        // Cap status length to prevent context bloat in the status block.
        // Status is rendered into every channel turn so it should stay short.
        let status = if scrubbed.len() > 256 {
            let end = scrubbed.floor_char_boundary(256);
            let boundary = scrubbed[..end].rfind(char::is_whitespace).unwrap_or(end);
            format!("{}...", &scrubbed[..boundary])
        } else {
            scrubbed.clone()
        };

        // An outcome status is the worker's terminal result, not just a
        // progress line: the full text rides in the tool output so the
        // completion path can deliver it, while the capped form feeds the
        // live status stream.
        let outcome = (args.kind == StatusKind::Outcome).then_some(scrubbed);
        let evidence = if args.kind == StatusKind::Outcome {
            args.evidence.clone()
        } else {
            None
        };

        // "No test evidence, no done": an outcome must document what was
        // actually run. Rejecting here gives the worker immediate feedback
        // (it can retry in the same turn); the hook's gate stays as
        // defense-in-depth for anything that bypasses the tool.
        if args.kind == StatusKind::Outcome {
            let has_evidence = args
                .evidence
                .as_ref()
                .map(|items| {
                    !items.is_empty() && items.iter().all(|item| !item.command.trim().is_empty())
                })
                .unwrap_or(false);
            if !has_evidence {
                return Err(SetStatusError(
                    "outcome requires evidence: pass evidence: [{ command, exit_code }, ...] \
                     documenting the command you ran and its exit code \
                     (e.g. { command: \"cargo test --lib\", exit_code: 0 })"
                        .into(),
                ));
            }
        }

        if args.kind == StatusKind::Outcome && !self.interactive {
            match self
                .process_run_logger
                .claim_worker_completion(self.worker_id, WorkerLifecycle::Running)
                .await
                .map_err(|error| SetStatusError(error.to_string()))?
            {
                WorkerTransitionResult::Applied { .. }
                | WorkerTransitionResult::Conflict {
                    current: WorkerLifecycle::Completing,
                } => {}
                WorkerTransitionResult::Conflict { current } => {
                    return Err(SetStatusError(format!(
                        "worker lifecycle conflict: expected running, found {}",
                        current.as_str()
                    )));
                }
                WorkerTransitionResult::NotFound => {
                    return Err(SetStatusError("worker run was not found".to_string()));
                }
            }
        }

        let event = ProcessEvent::WorkerStatus {
            agent_id: self.agent_id.clone(),
            worker_id: self.worker_id,
            channel_id: self.channel_id.clone(),
            status: status.clone(),
        };

        let _ = self.event_tx.send(event);

        Ok(SetStatusOutput {
            success: true,
            worker_id: self.worker_id,
            status,
            outcome,
            kind: args.kind,
            evidence,
        })
    }
}

/// Legacy function for setting worker status.
pub fn set_status(
    agent_id: AgentId,
    worker_id: WorkerId,
    status: impl Into<String>,
    event_tx: &broadcast::Sender<ProcessEvent>,
) {
    let event = ProcessEvent::WorkerStatus {
        agent_id,
        worker_id,
        channel_id: None,
        status: status.into(),
    };

    let _ = event_tx.send(event);
}

#[cfg(test)]
mod tests {
    use super::{Evidence, SetStatusArgs, SetStatusTool, StatusKind};
    use crate::conversation::{
        ProcessRunLogger, WorkerLifecycle, WorkerOutcomeKind, WorkerTerminalOwner,
        WorkerTransitionResult,
    };
    use rig::tool::Tool as _;
    use std::sync::Arc;

    async fn setup(interactive: bool) -> (SetStatusTool, ProcessRunLogger, uuid::Uuid) {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        let logger = ProcessRunLogger::new(pool);
        let worker_id = uuid::Uuid::new_v4();
        logger
            .log_worker_started(
                None,
                worker_id,
                "task",
                "builtin",
                &Arc::from("agent"),
                interactive,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        let (event_tx, _) = tokio::sync::broadcast::channel(8);
        (
            SetStatusTool::new(
                Arc::from("agent"),
                worker_id,
                None,
                event_tx,
                logger.clone(),
                interactive,
            ),
            logger,
            worker_id,
        )
    }

    #[tokio::test]
    async fn outcome_output_carries_full_text_while_status_stays_capped() {
        let (tool, _logger, _worker_id) = setup(false).await;
        let long = format!("Verified the deployment. {}", "detail ".repeat(60));
        let output = tool
            .call(SetStatusArgs {
                status: long.clone(),
                kind: StatusKind::Outcome,
                evidence: Some(vec![Evidence {
                    command: "cargo test --lib".to_string(),
                    exit_code: 0,
                    summary: Some("42 passed".to_string()),
                }]),
            })
            .await
            .unwrap();
        assert_eq!(output.outcome.as_deref(), Some(long.as_str()));
        assert!(output.status.len() <= 260);
        assert!(output.status.ends_with("..."));
        // Evidence rides along in the output.
        assert_eq!(output.evidence.as_ref().map(|items| items.len()), Some(1));
    }

    #[tokio::test]
    async fn progress_output_has_no_outcome_text() {
        let (tool, _logger, _worker_id) = setup(false).await;
        let output = tool
            .call(SetStatusArgs {
                status: "working on it".to_string(),
                kind: StatusKind::Progress,
                evidence: None,
            })
            .await
            .unwrap();
        assert!(output.outcome.is_none());
    }

    #[tokio::test]
    async fn outcome_without_evidence_is_rejected() {
        let (tool, _logger, _worker_id) = setup(false).await;
        // Text-only outcome: no evidence -> rejected, so the hook's gate
        // never sees `outcome_signaled` and the worker gets nudged.
        let error = tool
            .call(SetStatusArgs {
                status: "all done".to_string(),
                kind: StatusKind::Outcome,
                evidence: None,
            })
            .await
            .unwrap_err();
        assert!(error.0.contains("requires evidence"));
        // Empty evidence list is also rejected.
        let error = tool
            .call(SetStatusArgs {
                status: "all done".to_string(),
                kind: StatusKind::Outcome,
                evidence: Some(vec![]),
            })
            .await
            .unwrap_err();
        assert!(error.0.contains("requires evidence"));
        // An item with a blank command is rejected too.
        let error = tool
            .call(SetStatusArgs {
                status: "all done".to_string(),
                kind: StatusKind::Outcome,
                evidence: Some(vec![Evidence {
                    command: "  ".to_string(),
                    exit_code: 0,
                    summary: None,
                }]),
            })
            .await
            .unwrap_err();
        assert!(error.0.contains("requires evidence"));
        // Progress updates never need evidence.
        let output = tool
            .call(SetStatusArgs {
                status: "still working".to_string(),
                kind: StatusKind::Progress,
                evidence: None,
            })
            .await
            .unwrap();
        assert!(output.evidence.is_none());
    }

    #[tokio::test]
    async fn failed_task_evidence_is_accepted() {
        let (tool, _logger, _worker_id) = setup(false).await;
        // A failing command is still evidence — it documents what ran.
        let output = tool
            .call(SetStatusArgs {
                status: "Build failed: 3 type errors in auth module".to_string(),
                kind: StatusKind::Outcome,
                evidence: Some(vec![Evidence {
                    command: "cargo build".to_string(),
                    exit_code: 1,
                    summary: Some("3 errors".to_string()),
                }]),
            })
            .await
            .unwrap();
        assert_eq!(
            output.evidence.as_ref().map(|items| items[0].exit_code),
            Some(1)
        );
    }

    #[tokio::test]
    async fn non_interactive_outcome_claims_completing_idempotently() {
        let (tool, logger, worker_id) = setup(false).await;
        let args = || SetStatusArgs {
            status: "finished".to_string(),
            kind: StatusKind::Outcome,
            evidence: Some(vec![Evidence {
                command: "cargo test --lib".to_string(),
                exit_code: 0,
                summary: None,
            }]),
        };
        assert!(tool.call(args()).await.is_ok());
        assert!(tool.call(args()).await.is_ok());
        assert_eq!(
            logger.read_worker_lifecycle(worker_id).await.unwrap(),
            Some(WorkerLifecycle::Completing)
        );
    }

    #[tokio::test]
    async fn outcome_claim_beats_concurrent_cancel_transition() {
        let (tool, logger, worker_id) = setup(false).await;
        tool.call(SetStatusArgs {
            status: "finished".to_string(),
            kind: StatusKind::Outcome,
            evidence: Some(vec![Evidence {
                command: "cargo test --lib".to_string(),
                exit_code: 0,
                summary: None,
            }]),
        })
        .await
        .unwrap();
        assert_eq!(
            logger
                .transition_worker(
                    worker_id,
                    WorkerLifecycle::Running,
                    WorkerLifecycle::Cancelling,
                )
                .await
                .unwrap(),
            WorkerTransitionResult::Conflict {
                current: WorkerLifecycle::Completing,
            }
        );
        logger
            .complete_worker(
                worker_id,
                WorkerLifecycle::Completing,
                WorkerOutcomeKind::Succeeded,
                Some("finished"),
                "finished",
                None,
                0,
                WorkerTerminalOwner::Worker,
            )
            .await
            .unwrap();
        assert_eq!(
            logger
                .read_worker_terminal(worker_id)
                .await
                .unwrap()
                .unwrap()
                .outcome_kind,
            WorkerOutcomeKind::Succeeded
        );
    }

    #[tokio::test]
    async fn interactive_outcome_does_not_claim_terminal_lifecycle() {
        let (tool, logger, worker_id) = setup(true).await;
        tool.call(SetStatusArgs {
            status: "turn complete".to_string(),
            kind: StatusKind::Outcome,
            evidence: Some(vec![Evidence {
                command: "cargo test --lib".to_string(),
                exit_code: 0,
                summary: None,
            }]),
        })
        .await
        .unwrap();
        assert_eq!(
            logger.read_worker_lifecycle(worker_id).await.unwrap(),
            Some(WorkerLifecycle::Running)
        );
    }
}
