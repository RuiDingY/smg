//! Message threads on the Messages route (beta): the turns of a conversation
//! kept in this process, so a client sends each turn's new messages only.
//!
//! A `thread: {"type": "create"}` request is a whole conversation; its
//! answer is kept under the answer's `id`, with the request's `system`,
//! `tools`, messages and fixed fields. A `thread: {"type": "continue",
//! "previous_message_id"}` request names an answer and carries the messages
//! since it; the gateway rebuilds the conversation from the stored turns
//! (the create turn's `system` and `tools`, every turn's messages and
//! answer, then the new messages), checks the fixed fields against the
//! create turn's, and routes the plain request this would have been. The
//! store is the Responses history's in-memory store with its record schema
//! unchanged: a turn is a [`StoredResponse`] whose `input` holds the turn's
//! request fields ([`TurnInput`]) and whose `raw_response` holds the answer,
//! chained by `previous_response_id`, so a thread holds its conversation
//! once, not once per turn. The store keeps the turns the operator's
//! `memory` bounds allow, the `max_responses` and `retention_secs` of the
//! memory history backend (unset: 10,000 turns, each for a day after it was
//! written), the oldest evicted first, on every write; a thread is read
//! back whole up to that count. A turn is written under its answer's id.
//! An id that is already a stored turn's (a repeated `rid`, a backend's
//! repeated id, a guess) would hand the client an id that continues another
//! conversation, so the answer is reissued under a fresh id, in the answer
//! the client reads and in the store.
//!
//! A streamed answer is stored as well: the thread layer reads the events
//! on their way to the client, takes the turn's place in the store at
//! `message_start` (a taken id is reissued there, in the event the client
//! reads), folds the content blocks and the final `message_delta` into the
//! answer, and writes it whole at `message_stop`; the bytes the client reads
//! are the router's own, except a reissued `message_start`.
//!
//! Best effort, as the public API's threads are: a turn that was not stored
//! (an error, an answer the gateway could not read, a write the store
//! refused, a stream cut before its `message_stop`, an evicted record) makes
//! the next continue a
//! 404 `thread_not_found`, and the
//! client replays the conversation in a new create request, as the public
//! SDKs' thread helper does on its own. An id
//! may be continued more than once (a fork); the public API's
//! `thread_already_continued` is not answered. A continue whose rebuilt
//! conversation breaks a request rule is the same 404: the public API's
//! codes name no such case, and the client's replay (the whole conversation
//! as a create request) meets the 400 that names the rule.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use axum::{
    body::{to_bytes, Body, Bytes},
    http::{header, StatusCode},
    response::Response,
};
use http_body::{Body as HttpBody, Frame, SizeHint};
use openai_protocol::messages::{
    ContentBlock, ContentBlockDelta, CreateMessageRequest, ErrorDetails, ErrorEnvelope,
    ErrorResponse, InputContent, InputContentBlock, InputMessage, Message, MessageDelta,
    MessageDeltaUsage, MessageStreamEvent, MessageThread, Role, SystemContent, Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use smg_data_connector::{
    visible_chain, MemoryConfig, MemoryResponseStorage, ResponseId, ResponseStorage,
    ResponseStorageError, StoredResponse,
};
use tracing::{debug, warn};

use crate::{
    middleware::messages_error_type, observability::metrics::Metrics, routers::error as route_error,
};

/// An answer over this cannot be read back for its record: the Messages
/// route's own answers are far smaller.
const ANSWER_LIMIT: usize = 64 << 20;

/// What a stored turn holds in its record's `input`: the fields the
/// conversation is rebuilt from. The create turn carries the thread's
/// `system`, `tools` and fixed fields; every turn carries the messages it
/// added (the create request's conversation, a continue request's new
/// messages); the answer is the record's `raw_response`.
#[derive(Deserialize)]
struct TurnInput {
    #[serde(default)]
    system: Option<SystemContent>,
    #[serde(default)]
    tools: Option<Vec<Tool>>,
    messages: Vec<InputMessage>,
    #[serde(default)]
    fixed: Map<String, Value>,
}

impl TurnInput {
    /// The record's `input` of the turn `request` sends: the create turn
    /// (`root`) with the thread's `system`, `tools` and fixed fields, any
    /// other with its messages alone.
    fn record(request: &CreateMessageRequest, root: bool) -> Value {
        if root {
            json!({
                "system": request.system,
                "tools": request.tools,
                "messages": request.messages,
                "fixed": fixed_fields(request),
            })
        } else {
            json!({ "messages": request.messages })
        }
    }
}

/// The request fields a continue request repeats as the create request sent
/// them, by name, as JSON: `model`, `tool_choice`, `thinking`,
/// `mcp_servers`, `container`, `output_config`, `output_format`, and every
/// field the typed request does not model (the beta fields in `other`). The
/// sampling fields, `max_tokens`, `stop_sequences`, `metadata`, `stream`,
/// `service_tier` and `rid` may change from turn to turn.
fn fixed_fields(request: &CreateMessageRequest) -> Map<String, Value> {
    let mut fixed = Map::new();
    let mut keep = |name: &str, value: Value| {
        if !value.is_null() {
            fixed.insert(name.to_string(), value);
        }
    };
    keep("model", Value::String(request.model.clone()));
    keep("tool_choice", json_of(&request.tool_choice));
    keep("thinking", json_of(&request.thinking));
    keep("mcp_servers", json_of(&request.mcp_servers));
    keep("container", json_of(&request.container));
    keep("output_config", json_of(&request.output_config));
    keep("output_format", json_of(&request.output_format));
    for (name, value) in &request.other {
        keep(name, value.clone());
    }
    fixed
}

fn json_of<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// The fixed fields that differ between the create turn's and a continue
/// request's, by name, sorted.
fn differing(stored: &Map<String, Value>, sent: &Map<String, Value>) -> Vec<String> {
    let mut names: Vec<&String> = stored
        .keys()
        .chain(sent.keys())
        .filter(|name| stored.get(*name) != sent.get(*name))
        .collect();
    names.sort();
    names.dedup();
    names.into_iter().cloned().collect()
}

/// The stored answer as the assistant message of the conversation: its
/// content blocks read in their wire shape as input blocks (text,
/// `tool_use`, the thinking blocks and the server-tool blocks share it).
fn answer_message(answer: &Value) -> Result<InputMessage, Refusal> {
    let content = answer
        .get("content")
        .ok_or_else(|| Refusal::NotFound("a stored answer has no content".to_string()))?;
    let blocks = Vec::<InputContentBlock>::deserialize(content).map_err(|error| {
        Refusal::NotFound(format!("a stored answer's content is no input: {error}"))
    })?;
    Ok(InputMessage {
        role: Role::Assistant,
        content: InputContent::Blocks(blocks),
    })
}

/// Why a continue request is refused before the model runs.
enum Refusal {
    /// Nothing this request can continue: the public API's 404, which the
    /// client answers with a replay. The reason goes to the log alone, as
    /// the public API's refusal does not say.
    NotFound(String),
    /// The fixed fields the request changed.
    Mismatch(Vec<String>),
}

impl Refusal {
    fn outcome(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "not_found",
            Self::Mismatch(_) => "mismatch",
        }
    }

    fn into_response(self, previous_message_id: &str) -> Response {
        match self {
            Self::NotFound(why) => {
                debug!(previous_message_id, why = %why, "message thread: continue refused");
                refused(
                    StatusCode::NOT_FOUND,
                    "thread_not_found",
                    format!(
                        "thread: no thread can be continued from `{previous_message_id}`; \
                         send the whole conversation in a new create request"
                    ),
                )
            }
            Self::Mismatch(names) => refused(
                StatusCode::BAD_REQUEST,
                "thread_fingerprint_mismatch",
                format!(
                    "thread: the request changes {}: a continue request repeats the create \
                     request's fixed fields and sends the new messages only",
                    names.join(", ")
                ),
            ),
        }
    }
}

/// A refusal in the Messages envelope with its `details.error_code`, under
/// the gateway's error-code header and metric label (`create_error`); the
/// envelope layer fills in the request id.
fn refused(status: StatusCode, code: &str, message: String) -> Response {
    let mut response = route_error::create_error(status, code, message.as_str());
    let error = ErrorResponse {
        error_type: messages_error_type(status).to_string(),
        message,
        details: Some(ErrorDetails {
            error_code: code.to_string(),
            unsupported: None,
        }),
    };
    let body = serde_json::to_vec(&ErrorEnvelope::new(error, "")).unwrap_or_default();
    *response.body_mut() = Body::from(body);
    response
}

/// How a turn's record was written.
enum Written {
    /// Under the answer's id.
    Kept,
    /// The answer's id was a stored turn's: the answer, serialized again
    /// with the fresh id it was reissued under.
    Reissued(Bytes),
    /// Not written: the answer has no id, or the reissued write was refused.
    Refused,
}

/// A fresh message id, as the request-id layer mints them.
fn fresh_message_id() -> String {
    format!(
        "msg_{}",
        uuid::Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)).simple()
    )
}

/// A thread turn in flight: what its record holds besides the answer.
pub(crate) struct ThreadTurn {
    op: &'static str,
    previous: Option<ResponseId>,
    input: Value,
    model: String,
    tenant: Option<String>,
    streamed: bool,
}

impl ThreadTurn {
    /// The store's record of this turn: `answer` under `id`, the turn's
    /// request as the record's input.
    fn record(self, id: ResponseId, answer: Value) -> StoredResponse {
        let mut record = StoredResponse::new(self.previous);
        record.id = id;
        record.input = self.input;
        record.model = Some(self.model);
        record.raw_response = answer;
        record.tenant = self.tenant;
        record
    }

    /// The turn's place in the store while its answer streams: `id` taken,
    /// the `message_start` message as the answer and no input yet (a
    /// continue meanwhile is the 404); the record replaces it at
    /// `message_stop`, a cut stream removes it.
    fn reservation(&self, id: ResponseId, message: &Message) -> StoredResponse {
        let mut record = StoredResponse::new(self.previous.clone());
        record.id = id;
        record.model = Some(self.model.clone());
        record.raw_response = serde_json::to_value(message).unwrap_or(Value::Null);
        record.tenant.clone_from(&self.tenant);
        record
    }
}

/// The threads of this gateway, in the store of their turns.
pub(crate) struct MessageThreads {
    store: MemoryResponseStorage,
    /// The store's turn count, the depth a thread is read back to.
    max_turns: usize,
}

impl MessageThreads {
    /// The threads under the default bounds ([`MemoryConfig::default`]).
    pub(crate) fn new() -> Self {
        Self::from_config(&MemoryConfig::default())
    }

    /// The threads in a store bounded as the operator's `memory` section
    /// says: at most `max_responses` turns, each kept for `retention_secs`
    /// after it was written, the oldest evicted first on every write. The
    /// count is also the depth a thread is read back to, so no kept thread
    /// is cut short of its create turn.
    pub(crate) fn from_config(config: &MemoryConfig) -> Self {
        Self {
            store: MemoryResponseStorage::with_config(config),
            max_turns: config.max_responses,
        }
    }

    /// The request to route and, for a thread turn, the turn to store with
    /// its answer: a create request as it came, its `thread` taken off; a
    /// continue request as the conversation it continues, rebuilt from the
    /// stored turns. Or the public API's refusal of a continue: a 404
    /// `thread_not_found` when nothing can be continued from the id
    /// (unknown, evicted, another tenant's, a turn the gateway cannot read,
    /// a conversation that does not take the new messages), a 400
    /// `thread_fingerprint_mismatch` naming the fixed fields the request
    /// changed.
    pub(crate) async fn resolve(
        &self,
        mut request: CreateMessageRequest,
        tenant: Option<String>,
    ) -> Result<(CreateMessageRequest, Option<ThreadTurn>), Response> {
        let Some(thread) = request.thread.take() else {
            return Ok((request, None));
        };
        let streamed = request.is_stream();
        let model = request.model.clone();
        match thread {
            MessageThread::Create => {
                let input = TurnInput::record(&request, true);
                let turn = ThreadTurn {
                    op: "create",
                    previous: None,
                    input,
                    model,
                    tenant,
                    streamed,
                };
                Ok((request, Some(turn)))
            }
            MessageThread::Continue {
                previous_message_id,
            } => {
                let input = TurnInput::record(&request, false);
                match self
                    .continue_from(&previous_message_id, tenant.as_deref(), &mut request)
                    .await
                {
                    Ok(()) => {
                        let turn = ThreadTurn {
                            op: "continue",
                            previous: Some(ResponseId::from(previous_message_id)),
                            input,
                            model,
                            tenant,
                            streamed,
                        };
                        Ok((request, Some(turn)))
                    }
                    Err(refusal) => {
                        Metrics::record_message_thread_turn("continue", refusal.outcome());
                        Err(refusal.into_response(&previous_message_id))
                    }
                }
            }
        }
    }

    /// `request` as the conversation of the thread ending at `id`, continued
    /// by its own messages.
    async fn continue_from(
        &self,
        id: &str,
        tenant: Option<&str>,
        request: &mut CreateMessageRequest,
    ) -> Result<(), Refusal> {
        let turns = self.turns(id, tenant).await?;
        let root = TurnInput::deserialize(&turns[0].input)
            .map_err(|error| Refusal::NotFound(format!("the create turn is no turn: {error}")))?;
        let changed = differing(&root.fixed, &fixed_fields(request));
        if !changed.is_empty() {
            return Err(Refusal::Mismatch(changed));
        }
        let mut messages = root.messages;
        messages.push(answer_message(&turns[0].raw_response)?);
        for turn in &turns[1..] {
            let input = TurnInput::deserialize(&turn.input)
                .map_err(|error| Refusal::NotFound(format!("a stored turn is no turn: {error}")))?;
            messages.extend(input.messages);
            messages.push(answer_message(&turn.raw_response)?);
        }
        messages.append(&mut request.messages);
        request.system = root.system;
        request.tools = root.tools;
        request.messages = messages;
        if let Some(violation) = request.contract_violation() {
            return Err(Refusal::NotFound(format!(
                "the conversation does not take the new messages: {violation}"
            )));
        }
        Ok(())
    }

    /// The stored turns of the thread ending at `id` that `tenant` may see,
    /// the create turn first.
    async fn turns(
        &self,
        id: &str,
        tenant: Option<&str>,
    ) -> Result<Vec<Arc<StoredResponse>>, Refusal> {
        let chain = match self
            .store
            .get_response_chain(&ResponseId::from(id), Some(self.max_turns))
            .await
        {
            Ok(chain) => chain.responses,
            Err(ResponseStorageError::ResponseNotFound(_)) => {
                return Err(Refusal::NotFound("no turn under the id".to_string()));
            }
            Err(error) => {
                return Err(Refusal::NotFound(format!("the store failed: {error}")));
            }
        };
        let turns = visible_chain(chain, tenant);
        match turns.first() {
            None => Err(Refusal::NotFound("no turn under the id".to_string())),
            Some(root) if root.previous_response_id.is_some() => {
                Err(Refusal::NotFound("the create turn is gone".to_string()))
            }
            Some(_) => Ok(turns),
        }
    }

    /// The answer, with the turn stored under the answer's id when it is a
    /// 200 message; an id that is already a stored turn's is reissued, in
    /// the answer and in the store. A streamed answer is stored from its
    /// events by [`StreamedTurn`] as the client reads them. An error, an
    /// answer the gateway cannot read or a write the store refused leaves
    /// the turn unstored and the answer as it is: the next continue is a
    /// 404 and the client replays.
    pub(crate) async fn stored(&self, turn: ThreadTurn, response: Response) -> Response {
        let op = turn.op;
        if response.status() != StatusCode::OK {
            Metrics::record_message_thread_turn(op, "unstored");
            return response;
        }
        if turn.streamed {
            let (parts, body) = response.into_parts();
            let streamed = StreamedTurn::new(body, turn, self.store.clone());
            return Response::from_parts(parts, Body::new(streamed));
        }
        let (mut parts, body) = response.into_parts();
        let bytes = match to_bytes(body, ANSWER_LIMIT).await {
            Ok(bytes) => bytes,
            Err(error) => {
                warn!(%error, "message thread: the answer could not be read back");
                Metrics::record_message_thread_turn(op, "unstored");
                return route_error::internal_error(
                    "thread_answer_unreadable",
                    "The answer of a thread turn could not be read back",
                );
            }
        };
        let (outcome, bytes) = match serde_json::from_slice::<Value>(&bytes) {
            Ok(answer) => match self.store(turn, answer) {
                Written::Kept => ("stored", bytes),
                Written::Reissued(reissued) => {
                    parts.headers.remove(header::CONTENT_LENGTH);
                    ("reissued", reissued)
                }
                Written::Refused => ("unstored", bytes),
            },
            Err(_) => ("unstored", bytes),
        };
        Metrics::record_message_thread_turn(op, outcome);
        Response::from_parts(parts, Body::from(bytes))
    }

    /// The record of `turn` with `answer`, written under the answer's id
    /// while nothing is stored under it. An id that is already a stored
    /// turn's would hand the client an id that continues another
    /// conversation, so the answer is reissued under a fresh id.
    fn store(&self, turn: ThreadTurn, answer: Value) -> Written {
        let Some(id) = answer
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(ResponseId::from)
        else {
            return Written::Refused;
        };
        let record = turn.record(id, answer);
        let mut record = match self.store.store_response_if_absent(record) {
            Ok(()) => return Written::Kept,
            Err(record) => *record,
        };
        let fresh = fresh_message_id();
        debug!(
            taken = %record.id.0,
            fresh,
            "message thread: the answer's id is a stored turn's; reissued"
        );
        record.raw_response["id"] = Value::String(fresh.clone());
        record.id = ResponseId::from(fresh);
        let Ok(bytes) = serde_json::to_vec(&record.raw_response) else {
            return Written::Refused;
        };
        match self.store.store_response_if_absent(record) {
            Ok(()) => Written::Reissued(Bytes::from(bytes)),
            Err(_) => Written::Refused,
        }
    }
}

/// A streamed answer on its way to the client, read for its record. The turn
/// takes its place in the store at `message_start` (under a fresh id when
/// the event's id is a stored turn's, rewritten in the event the client
/// reads), the content blocks and the final `message_delta` are folded as
/// the events pass, and the whole answer replaces the place at
/// `message_stop`. A stream that ends, fails or is dropped before that
/// leaves no turn: the next continue is the 404. The bytes the client reads
/// are the router's own, except a reissued `message_start`.
struct StreamedTurn {
    inner: Body,
    store: MemoryResponseStorage,
    turn: Option<ThreadTurn>,
    op: &'static str,
    /// The bytes before the first complete event, held until it is whole: a
    /// `message_start` may be rewritten. Empty once the stream passes.
    held: Vec<u8>,
    /// The bytes of the event not yet complete, parsed when it is.
    carry: Vec<u8>,
    /// Every byte passes as it comes: `message_start` is through, or the
    /// stream opened with an error.
    passing: bool,
    /// The place taken in the store, and whether under a fresh id.
    place: Option<(ResponseId, bool)>,
    answer: Option<Assembled>,
    /// The outcome counted, once.
    outcome: Option<&'static str>,
    ended: bool,
}

/// The answer under construction: the `message_start` message, its content
/// blocks by index, and each tool call's input JSON as it streams.
struct Assembled {
    message: Message,
    blocks: Vec<ContentBlock>,
    partial_json: Vec<String>,
}

/// More content blocks than this in one answer are not followed.
const MAX_BLOCKS: usize = 1024;

impl StreamedTurn {
    fn new(inner: Body, turn: ThreadTurn, store: MemoryResponseStorage) -> Self {
        Self {
            inner,
            store,
            op: turn.op,
            turn: Some(turn),
            held: Vec::new(),
            carry: Vec::new(),
            passing: false,
            place: None,
            answer: None,
            outcome: None,
            ended: false,
        }
    }

    /// The bytes the client reads for `data`: all of them, once the first
    /// complete event is through; until then they are held.
    fn pass(&mut self, data: Bytes) -> Option<Bytes> {
        if self.passing {
            self.fold(&data);
            return Some(data);
        }
        self.held.extend_from_slice(&data);
        let mut out = Vec::new();
        while let Some(end) = event_end(&self.held) {
            let event: Vec<u8> = self.held.drain(..end).collect();
            match parse_event(&event) {
                Some(MessageStreamEvent::MessageStart { message }) => {
                    out.extend_from_slice(&self.start(message, &event));
                    self.passing = true;
                    let rest = std::mem::take(&mut self.held);
                    self.fold(&rest);
                    out.extend_from_slice(&rest);
                    break;
                }
                Some(MessageStreamEvent::Error { .. }) => {
                    out.extend_from_slice(&event);
                    self.passing = true;
                    out.append(&mut self.held);
                    break;
                }
                _ => out.extend_from_slice(&event),
            }
        }
        (!out.is_empty()).then(|| Bytes::from(out))
    }

    /// The `message_start` bytes the client reads: the event as it came, or
    /// rewritten with a fresh id when the message's id is a stored turn's.
    /// The turn's place in the store is taken here.
    fn start(&mut self, mut message: Message, event: &[u8]) -> Vec<u8> {
        let Some(turn) = &self.turn else {
            return event.to_vec();
        };
        let place = turn.reservation(ResponseId::from(message.id.as_str()), &message);
        let mut reissued = false;
        if self.store.store_response_if_absent(place).is_err() {
            let fresh = fresh_message_id();
            debug!(
                taken = %message.id,
                fresh,
                "message thread: a streamed answer's id is a stored turn's; reissued"
            );
            message.id = fresh;
            let place = turn.reservation(ResponseId::from(message.id.as_str()), &message);
            if self.store.store_response_if_absent(place).is_err() {
                return event.to_vec();
            }
            reissued = true;
        }
        let id = ResponseId::from(message.id.as_str());
        let bytes = if reissued {
            match start_event(&message) {
                Some(bytes) => bytes,
                None => {
                    self.store.remove_response(&id);
                    return event.to_vec();
                }
            }
        } else {
            event.to_vec()
        };
        self.place = Some((id, reissued));
        self.answer = Some(Assembled {
            message,
            blocks: Vec::new(),
            partial_json: Vec::new(),
        });
        bytes
    }

    /// The events in `data` read for the record, while a place is held.
    fn fold(&mut self, data: &[u8]) {
        if self.place.is_none() {
            return;
        }
        self.carry.extend_from_slice(data);
        while let Some(end) = event_end(&self.carry) {
            let event: Vec<u8> = self.carry.drain(..end).collect();
            if let Some(event) = parse_event(&event) {
                self.apply(event);
            }
            if self.place.is_none() {
                self.carry.clear();
                break;
            }
        }
    }

    fn apply(&mut self, event: MessageStreamEvent) {
        let Some(answer) = &mut self.answer else {
            return;
        };
        match event {
            MessageStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => answer.start_block(index, content_block),
            MessageStreamEvent::ContentBlockDelta { index, delta } => answer.delta(index, delta),
            MessageStreamEvent::ContentBlockStop { index } => answer.stop_block(index),
            MessageStreamEvent::MessageDelta { delta, usage } => answer.message_delta(delta, usage),
            MessageStreamEvent::MessageStop => self.finish(),
            MessageStreamEvent::Error { .. } => self.give_up(),
            MessageStreamEvent::MessageStart { .. } | MessageStreamEvent::Ping => {}
        }
    }

    /// The whole answer in the turn's place, at `message_stop`.
    fn finish(&mut self) {
        let (Some((id, reissued)), Some(turn), Some(mut answer)) =
            (self.place.take(), self.turn.take(), self.answer.take())
        else {
            return;
        };
        answer.message.content = answer.blocks;
        match serde_json::to_value(&answer.message) {
            Ok(value) => {
                self.store.replace_response(turn.record(id, value));
                self.count(if reissued { "reissued" } else { "stored" });
            }
            Err(_) => {
                self.store.remove_response(&id);
                self.count("unstored");
            }
        }
    }

    /// The place given back: an error event, a cut stream, a dropped body.
    fn give_up(&mut self) {
        if let Some((id, _)) = self.place.take() {
            self.store.remove_response(&id);
        }
        self.count("unstored");
    }

    fn count(&mut self, outcome: &'static str) {
        if self.outcome.is_none() {
            self.outcome = Some(outcome);
            Metrics::record_message_thread_turn(self.op, outcome);
        }
    }
}

impl Drop for StreamedTurn {
    fn drop(&mut self) {
        self.give_up();
    }
}

impl HttpBody for StreamedTurn {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // `Body` is `Unpin`, and so is everything else held here.
        let this = self.get_mut();
        loop {
            if this.ended {
                return Poll::Ready(None);
            }
            match Pin::new(&mut this.inner).poll_frame(cx) {
                Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                    Ok(data) => {
                        if let Some(out) = this.pass(data) {
                            return Poll::Ready(Some(Ok(Frame::data(out))));
                        }
                    }
                    Err(frame) => return Poll::Ready(Some(Ok(frame))),
                },
                Poll::Ready(None) => {
                    this.ended = true;
                    this.give_up();
                    let held = std::mem::take(&mut this.held);
                    return if held.is_empty() {
                        Poll::Ready(None)
                    } else {
                        Poll::Ready(Some(Ok(Frame::data(Bytes::from(held)))))
                    };
                }
                Poll::Ready(Some(Err(error))) => {
                    this.ended = true;
                    this.give_up();
                    return Poll::Ready(Some(Err(error)));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.ended || (self.inner.is_end_stream() && self.held.is_empty())
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

impl Assembled {
    fn start_block(&mut self, index: u32, block: ContentBlock) {
        let Some(index) = slot(index) else {
            return;
        };
        while self.blocks.len() < index {
            self.blocks.push(ContentBlock::Text {
                text: String::new(),
                citations: None,
            });
            self.partial_json.push(String::new());
        }
        if index == self.blocks.len() {
            self.blocks.push(block);
            self.partial_json.push(String::new());
        } else {
            self.blocks[index] = block;
            self.partial_json[index].clear();
        }
    }

    fn delta(&mut self, index: u32, delta: ContentBlockDelta) {
        let Some(index) = slot(index) else {
            return;
        };
        let (Some(block), Some(partial)) =
            (self.blocks.get_mut(index), self.partial_json.get_mut(index))
        else {
            return;
        };
        match (delta, block) {
            (ContentBlockDelta::TextDelta { text: more }, ContentBlock::Text { text, .. }) => {
                text.push_str(&more);
            }
            (ContentBlockDelta::InputJsonDelta { partial_json }, _) => {
                partial.push_str(&partial_json);
            }
            (
                ContentBlockDelta::ThinkingDelta { thinking: more },
                ContentBlock::Thinking { thinking, .. },
            ) => thinking.push_str(&more),
            (
                ContentBlockDelta::SignatureDelta { signature: more },
                ContentBlock::Thinking { signature, .. },
            ) => signature.push_str(&more),
            (
                ContentBlockDelta::CitationsDelta { citation },
                ContentBlock::Text { citations, .. },
            ) => citations.get_or_insert_with(Vec::new).push(citation),
            _ => {}
        }
    }

    /// A tool call's input, parsed from its deltas when the block closes.
    fn stop_block(&mut self, index: u32) {
        let Some(index) = slot(index) else {
            return;
        };
        let (Some(block), Some(partial)) =
            (self.blocks.get_mut(index), self.partial_json.get_mut(index))
        else {
            return;
        };
        if partial.is_empty() {
            return;
        }
        if let Ok(value) = serde_json::from_str::<Value>(partial) {
            match block {
                ContentBlock::ToolUse { input, .. } | ContentBlock::ServerToolUse { input, .. } => {
                    *input = value;
                }
                _ => {}
            }
        }
        partial.clear();
    }

    /// The final `message_delta`: the stop and the usage of the whole answer.
    fn message_delta(&mut self, delta: MessageDelta, usage: MessageDeltaUsage) {
        let message = &mut self.message;
        if delta.stop_reason.is_some() {
            message.stop_reason = delta.stop_reason;
        }
        if delta.stop_sequence.is_some() {
            message.stop_sequence = delta.stop_sequence;
        }
        if delta.stop_details.is_some() {
            message.stop_details = delta.stop_details;
        }
        if delta.container.is_some() {
            message.container = delta.container;
        }
        message.usage.output_tokens = usage.output_tokens;
        if let Some(input_tokens) = usage.input_tokens {
            message.usage.input_tokens = input_tokens;
        }
        if usage.cache_creation_input_tokens.is_some() {
            message.usage.cache_creation_input_tokens = usage.cache_creation_input_tokens;
        }
        if usage.cache_read_input_tokens.is_some() {
            message.usage.cache_read_input_tokens = usage.cache_read_input_tokens;
        }
        if usage.server_tool_use.is_some() {
            message.usage.server_tool_use = usage.server_tool_use;
        }
        if usage.output_tokens_details.is_some() {
            message.usage.output_tokens_details = usage.output_tokens_details;
        }
    }
}

/// A content block's slot, for an index this gateway follows.
fn slot(index: u32) -> Option<usize> {
    usize::try_from(index)
        .ok()
        .filter(|&index| index < MAX_BLOCKS)
}

/// The end of the first complete event in `buffer` (the index after its
/// blank line), when there is one.
fn event_end(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(2)
        .position(|pair| pair == b"\n\n")
        .map(|at| at + 2)
}

/// The event `block`'s data as the stream event it names; `None` for a
/// comment, a keep-alive or a shape this gateway does not know.
fn parse_event(block: &[u8]) -> Option<MessageStreamEvent> {
    let text = std::str::from_utf8(block).ok()?;
    let mut data = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    serde_json::from_str(&data).ok()
}

/// The `message_start` event with `message`, as the bytes of one event.
fn start_event(message: &Message) -> Option<Vec<u8>> {
    let data = serde_json::to_string(&MessageStreamEvent::MessageStart {
        message: message.clone(),
    })
    .ok()?;
    Some(format!("event: message_start\ndata: {data}\n\n").into_bytes())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::{body::Bytes, http::header, response::IntoResponse, Json};
    use chrono::Utc;
    use futures::{stream, StreamExt};
    use tokio::time::timeout;

    use super::*;

    fn request(body: Value) -> CreateMessageRequest {
        let mut body = body;
        let object = body.as_object_mut().expect("an object");
        object.entry("model").or_insert(json!("m"));
        object.entry("max_tokens").or_insert(json!(64));
        serde_json::from_value(body).expect("a well-formed request")
    }

    fn answer(id: &str, content: Value) -> Response {
        Json(json!({
            "id": id, "type": "message", "role": "assistant", "model": "m",
            "content": content, "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 2}
        }))
        .into_response()
    }

    async fn body_of(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("a body");
        serde_json::from_slice(&bytes).expect("a JSON body")
    }

    type Resolved = (CreateMessageRequest, Option<ThreadTurn>);

    /// The resolution of a request the thread store accepts.
    fn accepted(outcome: Result<Resolved, Response>) -> Resolved {
        match outcome {
            Ok(resolved) => resolved,
            Err(refused) => panic!("refused with {}", refused.status()),
        }
    }

    /// The refusal of a request the thread store refuses.
    fn refusal(outcome: Result<Resolved, Response>) -> Response {
        match outcome {
            Ok(_) => panic!("accepted"),
            Err(refused) => refused,
        }
    }

    /// A thread of `turns` user turns under `tenant`, each answered by a
    /// text reply under the id `msg_<tag>_<turn>`; answers the last id.
    async fn thread(
        threads: &MessageThreads,
        tenant: Option<&str>,
        turns: usize,
        tag: &str,
    ) -> String {
        let mut previous: Option<String> = None;
        for index in 1..=turns {
            let body = match &previous {
                None => json!({
                    "system": "be brief",
                    "tools": [{"name": "f", "input_schema": {"type": "object"}}],
                    "messages": [{"role": "user", "content": "turn 1"}],
                    "thread": {"type": "create"}
                }),
                Some(id) => json!({
                    "messages": [{"role": "user", "content": format!("turn {index}")}],
                    "thread": {"type": "continue", "previous_message_id": id}
                }),
            };
            let (_, turn) = accepted(
                threads
                    .resolve(request(body), tenant.map(str::to_string))
                    .await,
            );
            let id = format!("msg_{tag}_{index}");
            let reply = answer(
                &id,
                json!([{"type": "text", "text": format!("reply {index}")}]),
            );
            let response = threads.stored(turn.expect("a thread turn"), reply).await;
            assert_eq!(response.status(), StatusCode::OK);
            previous = Some(id);
        }
        previous.expect("at least one turn")
    }

    /// A request without `thread` is routed as it came.
    #[tokio::test]
    async fn a_plain_request_is_no_turn() {
        let threads = MessageThreads::new();
        let (request, turn) = accepted(
            threads
                .resolve(
                    request(json!({"messages": [{"role": "user", "content": "hi"}]})),
                    None,
                )
                .await,
        );
        assert!(turn.is_none());
        assert_eq!(request.messages.len(), 1);
    }

    /// A continue request becomes the whole conversation: the create turn's
    /// `system` and `tools`, every turn's messages and answer, then the new
    /// messages; the answer the client gets is the router's, byte for byte.
    #[tokio::test]
    async fn a_continue_rebuilds_the_conversation() {
        let threads = MessageThreads::new();
        let last = thread(&threads, None, 3, "c").await;
        let (rebuilt, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "turn 4"}],
                        "thread": {"type": "continue", "previous_message_id": last}
                    })),
                    None,
                )
                .await,
        );
        assert!(turn.is_some());
        assert!(rebuilt.thread.is_none());
        assert!(rebuilt.system.is_some());
        assert_eq!(rebuilt.tools.as_ref().map(Vec::len), Some(1));
        let texts: Vec<String> = rebuilt
            .messages
            .iter()
            .map(|message| match &message.content {
                InputContent::String(text) => text.clone(),
                InputContent::Blocks(blocks) => match &blocks[..] {
                    [InputContentBlock::Text(text)] => text.text.clone(),
                    other => panic!("one text block, not {other:?}"),
                },
            })
            .collect();
        assert_eq!(
            texts,
            ["turn 1", "reply 1", "turn 2", "reply 2", "turn 3", "reply 3", "turn 4"]
        );
        let roles: Vec<&Role> = rebuilt.messages.iter().map(|m| &m.role).collect();
        assert_eq!(roles.len(), 7);
        assert!(roles.iter().step_by(2).all(|role| **role == Role::User));
    }

    /// A continue from an id nobody issued, or whose create turn is gone, is
    /// the public API's 404 with its code in `error.details`.
    #[tokio::test]
    async fn an_unknown_id_is_not_found() {
        let threads = MessageThreads::new();
        let refused = refusal(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "again"}],
                        "thread": {"type": "continue", "previous_message_id": "msg_none"}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            refused
                .headers()
                .get(route_error::HEADER_X_SMG_ERROR_CODE)
                .and_then(|value| value.to_str().ok()),
            Some("thread_not_found")
        );
        let body = body_of(refused).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "not_found_error");
        assert_eq!(body["error"]["details"]["error_code"], "thread_not_found");
    }

    /// A continue that changes a fixed field names it, as a 400.
    #[tokio::test]
    async fn a_changed_fixed_field_is_a_mismatch() {
        let threads = MessageThreads::new();
        let last = thread(&threads, None, 1, "f").await;
        let refused = refusal(
            threads
                .resolve(
                    request(json!({
                        "model": "other", "thinking": {"type": "enabled", "budget_tokens": 1024},
                        "messages": [{"role": "user", "content": "again"}],
                        "thread": {"type": "continue", "previous_message_id": last}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
        let body = body_of(refused).await;
        assert_eq!(
            body["error"]["details"]["error_code"],
            "thread_fingerprint_mismatch"
        );
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(
            message.starts_with("thread: the request changes model, thinking:"),
            "{message}"
        );
    }

    /// A turn stored under a tenant is continued by that tenant alone; one
    /// stored under none by every caller.
    #[tokio::test]
    async fn a_thread_stays_with_its_tenant() {
        let threads = MessageThreads::new();
        let theirs = thread(&threads, Some("auth:a"), 1, "theirs").await;
        let anyones = thread(&threads, None, 1, "anyones").await;
        let cont = |id: &str| {
            request(json!({
                "messages": [{"role": "user", "content": "again"}],
                "thread": {"type": "continue", "previous_message_id": id}
            }))
        };
        for (id, tenant, continued) in [
            (&theirs, Some("auth:a"), true),
            (&theirs, Some("auth:b"), false),
            (&theirs, None, false),
            (&anyones, Some("auth:b"), true),
            (&anyones, None, true),
        ] {
            let outcome = threads.resolve(cont(id), tenant.map(str::to_string)).await;
            assert_eq!(outcome.is_ok(), continued, "{id} as {tenant:?}");
        }
    }

    /// A streamed answer with no events, an error and an answer with no id
    /// leave the turn unstored, and the answer as it was; the next continue
    /// is the 404.
    #[tokio::test]
    async fn an_unstored_turn_cannot_be_continued() {
        let threads = MessageThreads::new();
        let create = |stream: bool| {
            request(json!({
                "stream": stream,
                "messages": [{"role": "user", "content": "hi"}],
                "thread": {"type": "create"}
            }))
        };
        let cases: [(bool, Response); 3] = [
            (true, answer("msg_streamed", json!([]))),
            (false, route_error::bad_request("some_code", "refused")),
            (false, answer("", json!([{"type": "text", "text": "x"}]))),
        ];
        for (stream, response) in cases {
            let status = response.status();
            let (_, turn) = accepted(threads.resolve(create(stream), None).await);
            let answered = threads.stored(turn.expect("a turn"), response).await;
            assert_eq!(answered.status(), status);
        }
        let refused = refusal(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "again"}],
                        "thread": {"type": "continue", "previous_message_id": "msg_streamed"}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
    }

    /// The tool loop: an answer that calls a tool is continued by the
    /// result alone, paired against the stored call.
    #[tokio::test]
    async fn a_tool_result_continues_the_stored_call() {
        let threads = MessageThreads::new();
        let (_, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "tools": [{"name": "f", "input_schema": {"type": "object"}}],
                        "messages": [{"role": "user", "content": "call f"}],
                        "thread": {"type": "create"}
                    })),
                    None,
                )
                .await,
        );
        let call = answer(
            "msg_call",
            json!([{"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}]),
        );
        threads.stored(turn.expect("a turn"), call).await;
        let (rebuilt, _) = accepted(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"}
                        ]}],
                        "thread": {"type": "continue", "previous_message_id": "msg_call"}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(rebuilt.messages.len(), 3);
        let unpaired = refusal(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_other", "content": "42"}
                        ]}],
                        "thread": {"type": "continue", "previous_message_id": "msg_call"}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(unpaired.status(), StatusCode::NOT_FOUND);
    }

    /// A streamed turn goes to the client as it comes: the store reads no
    /// stream back, and the turn stays unstored.
    #[tokio::test]
    async fn a_streamed_turn_is_not_read_back() {
        let threads = MessageThreads::new();
        let (_, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "stream": true,
                        "messages": [{"role": "user", "content": "hi"}],
                        "thread": {"type": "create"}
                    })),
                    None,
                )
                .await,
        );
        let first = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_sse\"}}\n\n";
        let never_ending =
            stream::iter([Ok::<_, std::io::Error>(Bytes::from(first))]).chain(stream::pending());
        let sse = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from_stream(never_ending))
            .expect("a response");
        let answered = timeout(
            Duration::from_secs(2),
            threads.stored(turn.expect("a turn"), sse),
        )
        .await
        .expect("a streamed turn is answered before its stream ends");
        assert_eq!(answered.status(), StatusCode::OK);
        let mut events = answered.into_body().into_data_stream();
        let event = events
            .next()
            .await
            .expect("the first event")
            .expect("a readable event");
        assert_eq!(event, first.as_bytes());
        let refused = refusal(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "again"}],
                        "thread": {"type": "continue", "previous_message_id": "msg_sse"}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
    }

    /// A thread whose create turn the store let go (the count bound reached
    /// under its continues) cannot be continued: the client replays. The
    /// store itself let the turn go, not only the walk's depth.
    #[tokio::test]
    async fn an_evicted_create_turn_ends_the_thread() {
        let threads = MessageThreads::from_config(&MemoryConfig {
            max_responses: 2,
            ..MemoryConfig::default()
        });
        assert_eq!(
            threads.max_turns, 2,
            "the chain is read back to the configured count, not the default"
        );
        let last = thread(&threads, None, 3, "bounded").await;
        assert!(
            threads
                .store
                .get_response(&ResponseId::from("msg_bounded_1"))
                .await
                .expect("a store read")
                .is_none(),
            "the create turn's record is gone from the store"
        );
        let refused = refusal(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "turn 4"}],
                        "thread": {"type": "continue", "previous_message_id": last}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
        let error = body_of(refused).await;
        assert_eq!(error["error"]["details"]["error_code"], "thread_not_found");
    }

    /// A second answer under an id that is already a stored turn (another
    /// caller's guess) leaves the first turn as it was and is reissued under
    /// a fresh id: the owner continues the first, the second caller
    /// continues their own under the fresh id and gets the 404 under the
    /// guessed one.
    #[tokio::test]
    async fn an_answer_under_a_stored_id_leaves_the_first_turn() {
        let threads = MessageThreads::new();
        let create = |system: &str| {
            request(json!({
                "system": system,
                "messages": [{"role": "user", "content": "hi"}],
                "thread": {"type": "create"}
            }))
        };
        let reply = |text: &str| answer("msg_taken", json!([{"type": "text", "text": text}]));
        let mut ids = Vec::new();
        for (system, tenant) in [("first", "auth:a"), ("second", "auth:b")] {
            let (_, turn) = accepted(
                threads
                    .resolve(create(system), Some(tenant.to_string()))
                    .await,
            );
            let answered = threads.stored(turn.expect("a turn"), reply(system)).await;
            assert_eq!(answered.status(), StatusCode::OK);
            ids.push(
                body_of(answered).await["id"]
                    .as_str()
                    .expect("an id")
                    .to_string(),
            );
        }
        assert_eq!(ids[0], "msg_taken");
        assert_ne!(ids[1], "msg_taken", "the second answer is reissued");
        assert!(ids[1].starts_with("msg_"), "{}", ids[1]);
        let cont = |id: &str, tenant: &str| {
            threads.resolve(
                request(json!({
                    "messages": [{"role": "user", "content": "again"}],
                    "thread": {"type": "continue", "previous_message_id": id}
                })),
                Some(tenant.to_string()),
            )
        };
        let (rebuilt, _) = accepted(cont("msg_taken", "auth:a").await);
        assert!(
            matches!(rebuilt.system, Some(SystemContent::String(ref text)) if text == "first"),
            "the first turn's system, not {:?}",
            json_of(&rebuilt.system)
        );
        assert_eq!(rebuilt.messages.len(), 3);
        let (theirs, _) = accepted(cont(&ids[1], "auth:b").await);
        assert!(
            matches!(theirs.system, Some(SystemContent::String(ref text)) if text == "second"),
            "the second caller's own turn under the fresh id"
        );
        assert_eq!(
            refusal(cont("msg_taken", "auth:b").await).status(),
            StatusCode::NOT_FOUND
        );
    }

    /// A client that sends one `rid` on every turn: the second answer comes
    /// back under a fresh id, the turn is stored under it, and the third
    /// turn's conversation is whole. Before the reissue the second turn was
    /// unstored and the third turn silently lost it.
    #[tokio::test]
    async fn a_repeated_rid_keeps_the_thread_whole() {
        let threads = MessageThreads::new();
        let turn = |text: &str, previous: Option<&str>| match previous {
            None => request(json!({
                "messages": [{"role": "user", "content": text}],
                "thread": {"type": "create"}
            })),
            Some(id) => request(json!({
                "messages": [{"role": "user", "content": text}],
                "thread": {"type": "continue", "previous_message_id": id}
            })),
        };
        let (_, created) = accepted(threads.resolve(turn("m1", None), None).await);
        let first = threads
            .stored(
                created.expect("a turn"),
                answer("R", json!([{"type": "text", "text": "a1"}])),
            )
            .await;
        assert_eq!(body_of(first).await["id"], "R");
        let (_, continued) = accepted(threads.resolve(turn("m2", Some("R")), None).await);
        let second = threads
            .stored(
                continued.expect("a turn"),
                answer("R", json!([{"type": "text", "text": "a2"}])),
            )
            .await;
        assert_eq!(second.status(), StatusCode::OK);
        let second = body_of(second).await;
        let reissued = second["id"].as_str().expect("an id");
        assert_ne!(reissued, "R", "the colliding id is reissued");
        assert_eq!(
            second["content"][0]["text"], "a2",
            "the answer itself is unchanged"
        );
        let (rebuilt, _) = accepted(threads.resolve(turn("m3", Some(reissued)), None).await);
        let texts: Vec<String> = rebuilt
            .messages
            .iter()
            .map(|message| match &message.content {
                InputContent::String(text) => text.clone(),
                InputContent::Blocks(blocks) => match &blocks[..] {
                    [InputContentBlock::Text(text)] => text.text.clone(),
                    other => panic!("one text block, not {other:?}"),
                },
            })
            .collect();
        assert_eq!(texts, ["m1", "a1", "m2", "a2", "m3"]);
    }

    /// A turn older than the operator's retention is gone when the store
    /// next evicts: the thread ends with the public 404 and the client
    /// replays.
    #[tokio::test]
    async fn a_turn_past_the_retention_ends_the_thread() {
        let threads = MessageThreads::from_config(&MemoryConfig {
            retention_secs: Some(60),
            ..MemoryConfig::default()
        });
        let last = thread(&threads, None, 2, "aged").await;
        let cont = || {
            request(json!({
                "messages": [{"role": "user", "content": "turn 3"}],
                "thread": {"type": "continue", "previous_message_id": last}
            }))
        };
        accepted(threads.resolve(cont(), None).await);
        threads
            .store
            .evict_at(Utc::now() + chrono::Duration::seconds(61));
        let refused = refusal(threads.resolve(cont(), None).await);
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
    }

    /// A relayed answer carries its upstream's Content-Length. Reissued under
    /// a fresh id its body is new bytes, so the stale length is dropped and
    /// the server frames the body it sends.
    #[tokio::test]
    async fn a_reissued_answer_drops_the_stale_content_length() {
        let threads = MessageThreads::new();
        let relayed = || {
            let body = serde_json::to_vec(&json!({
                "id": "R", "type": "message", "role": "assistant", "model": "m",
                "content": [{"type": "text", "text": "a"}], "stop_reason": "end_turn",
                "stop_sequence": null, "usage": {"input_tokens": 3, "output_tokens": 2}
            }))
            .expect("json");
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::CONTENT_LENGTH, body.len())
                .body(Body::from(body))
                .expect("a response")
        };
        let create = || {
            request(json!({
                "messages": [{"role": "user", "content": "hi"}],
                "thread": {"type": "create"}
            }))
        };
        let (_, first) = accepted(threads.resolve(create(), None).await);
        let kept = threads.stored(first.expect("a turn"), relayed()).await;
        assert_eq!(body_of(kept).await["id"], "R");
        let (_, second) = accepted(threads.resolve(create(), None).await);
        let reissued = threads.stored(second.expect("a turn"), relayed()).await;
        let declared = reissued.headers().get(header::CONTENT_LENGTH).map(|value| {
            value
                .to_str()
                .expect("ascii")
                .parse::<usize>()
                .expect("a length")
        });
        let bytes = to_bytes(reissued.into_body(), usize::MAX)
            .await
            .expect("a body");
        assert_ne!(
            serde_json::from_slice::<Value>(&bytes).expect("json")["id"],
            "R"
        );
        assert!(
            declared.is_none_or(|length| length == bytes.len()),
            "a stale Content-Length ({declared:?}) on a body of {} bytes",
            bytes.len()
        );
    }

    /// A streamed answer as the router writes it: `events` as `event:` and
    /// `data:` lines, then cut into frames at the byte offsets `cuts`.
    fn sse(events: &[(&str, Value)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, data) in events {
            out.extend_from_slice(format!("event: {name}\ndata: {data}\n\n").as_bytes());
        }
        out
    }

    fn streamed(bytes: &[u8], cuts: &[usize]) -> Response {
        let mut frames = Vec::new();
        let mut at = 0;
        for &cut in cuts {
            frames.push(Bytes::copy_from_slice(&bytes[at..cut]));
            at = cut;
        }
        frames.push(Bytes::copy_from_slice(&bytes[at..]));
        let body = Body::from_stream(stream::iter(
            frames.into_iter().map(Ok::<_, std::io::Error>),
        ));
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(body)
            .expect("a response")
    }

    fn message_start(id: &str) -> (&'static str, Value) {
        (
            "message_start",
            json!({"type": "message_start", "message": {
                "id": id, "type": "message", "role": "assistant", "model": "m", "content": [],
                "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 7, "output_tokens": 0}
            }}),
        )
    }

    fn text_events(id: &str, text: &str) -> Vec<(&'static str, Value)> {
        vec![
            message_start(id),
            (
                "content_block_start",
                json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            ),
            (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
            ),
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0}),
            ),
            (
                "message_delta",
                json!({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 3}}),
            ),
            ("message_stop", json!({"type": "message_stop"})),
        ]
    }

    fn create(text: &str, streamed: bool) -> CreateMessageRequest {
        request(json!({
            "stream": streamed,
            "messages": [{"role": "user", "content": text}],
            "thread": {"type": "create"}
        }))
    }

    fn continue_from(id: &str, text: &str) -> CreateMessageRequest {
        request(json!({
            "messages": [{"role": "user", "content": text}],
            "thread": {"type": "continue", "previous_message_id": id}
        }))
    }

    /// The texts of a rebuilt conversation, one per message.
    fn texts(request: &CreateMessageRequest) -> Vec<String> {
        request
            .messages
            .iter()
            .map(|message| match &message.content {
                InputContent::String(text) => text.clone(),
                InputContent::Blocks(blocks) => match &blocks[..] {
                    [InputContentBlock::Text(text)] => text.text.clone(),
                    other => panic!("one text block, not {other:?}"),
                },
            })
            .collect()
    }

    /// A streamed create is stored from its events, cut across frames as
    /// they come: the client reads the router's bytes unchanged, and the
    /// continue from the `message_start` id carries the streamed text, the
    /// stop reason and the usage of the final `message_delta`.
    #[tokio::test]
    async fn a_streamed_answer_is_stored_from_its_events() {
        let threads = MessageThreads::new();
        let bytes = sse(&text_events("msg_s1", "Hello, world"));
        let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
        let cuts = [17, 230, 231, bytes.len() - 5];
        let answered = threads
            .stored(turn.expect("a turn"), streamed(&bytes, &cuts))
            .await;
        assert_eq!(answered.status(), StatusCode::OK);
        let read = to_bytes(answered.into_body(), usize::MAX)
            .await
            .expect("the stream");
        assert_eq!(
            read.as_ref(),
            bytes.as_slice(),
            "the client's bytes are the router's"
        );
        let (rebuilt, _) = accepted(
            threads
                .resolve(continue_from("msg_s1", "again"), None)
                .await,
        );
        assert_eq!(texts(&rebuilt), ["hi", "Hello, world", "again"]);
        let turns = threads
            .turns("msg_s1", None)
            .await
            .ok()
            .expect("the thread");
        let stored = &turns[0].raw_response;
        assert_eq!(stored["stop_reason"], "end_turn");
        assert_eq!(stored["usage"]["input_tokens"], 7);
        assert_eq!(stored["usage"]["output_tokens"], 3);
        assert_eq!(turns[0].input["messages"][0]["content"], "hi");
    }

    /// A streamed tool call's input is the JSON its deltas spell, parsed when
    /// the block closes.
    #[tokio::test]
    async fn a_streamed_tool_call_is_stored_whole() {
        let threads = MessageThreads::new();
        let bytes = sse(&[
            message_start("msg_t1"),
            (
                "content_block_start",
                json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}}),
            ),
            (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}),
            ),
            (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": " [1, 2]}"}}),
            ),
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0}),
            ),
            (
                "message_delta",
                json!({"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, "usage": {"output_tokens": 9}}),
            ),
            ("message_stop", json!({"type": "message_stop"})),
        ]);
        let (_, turn) = accepted(threads.resolve(create("call f", true), None).await);
        let answered = threads
            .stored(turn.expect("a turn"), streamed(&bytes, &[]))
            .await;
        to_bytes(answered.into_body(), usize::MAX)
            .await
            .expect("the stream");
        let turns = threads
            .turns("msg_t1", None)
            .await
            .ok()
            .expect("the thread");
        let stored = &turns[0].raw_response;
        assert_eq!(stored["content"][0]["type"], "tool_use");
        assert_eq!(stored["content"][0]["input"], json!({"a": [1, 2]}));
        assert_eq!(stored["stop_reason"], "tool_use");
    }

    /// A streamed answer whose `message_start` id is a stored turn's is
    /// reissued in that event: the client continues under the fresh id, the
    /// first turn keeps its own.
    #[tokio::test]
    async fn a_streamed_answer_under_a_stored_id_is_reissued() {
        let threads = MessageThreads::new();
        let (_, first) = accepted(threads.resolve(create("one", false), None).await);
        let kept = threads
            .stored(
                first.expect("a turn"),
                answer("msg_dup", json!([{"type": "text", "text": "first answer"}])),
            )
            .await;
        assert_eq!(body_of(kept).await["id"], "msg_dup");
        let bytes = sse(&text_events("msg_dup", "second answer"));
        let (_, second) = accepted(threads.resolve(create("two", true), None).await);
        let answered = threads
            .stored(second.expect("a turn"), streamed(&bytes, &[40]))
            .await;
        let read = to_bytes(answered.into_body(), usize::MAX)
            .await
            .expect("the stream");
        let text = std::str::from_utf8(&read).expect("utf-8");
        let first_event = text.split("\n\n").next().expect("an event");
        let data = first_event
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("the message_start data");
        let start: Value = serde_json::from_str(data).expect("json");
        let fresh = start["message"]["id"].as_str().expect("an id").to_string();
        assert_ne!(
            fresh, "msg_dup",
            "the colliding id is reissued in message_start"
        );
        assert!(text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
        let (rebuilt, _) = accepted(threads.resolve(continue_from(&fresh, "three"), None).await);
        assert_eq!(texts(&rebuilt), ["two", "second answer", "three"]);
        let (rebuilt, _) = accepted(
            threads
                .resolve(continue_from("msg_dup", "more"), None)
                .await,
        );
        assert_eq!(texts(&rebuilt), ["one", "first answer", "more"]);
    }

    /// A stream that ends before its `message_stop`, or that carries an error
    /// event (here one a `message_stop` still follows), leaves no turn: the
    /// continue from its id is the 404.
    #[tokio::test]
    async fn a_cut_or_failed_stream_leaves_no_turn() {
        let threads = MessageThreads::new();
        let mut cut = text_events("msg_cut", "half");
        cut.truncate(3);
        let mut failed = text_events("msg_failed", "half");
        failed.truncate(3);
        failed.push((
            "error",
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "later"}}),
        ));
        failed.extend(text_events("msg_failed", "half").into_iter().skip(4));
        for (id, events) in [("msg_cut", cut), ("msg_failed", failed)] {
            let bytes = sse(&events);
            let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
            let answered = threads
                .stored(turn.expect("a turn"), streamed(&bytes, &[]))
                .await;
            let read = to_bytes(answered.into_body(), usize::MAX)
                .await
                .expect("the stream");
            assert_eq!(read.as_ref(), bytes.as_slice());
            let refused = refusal(threads.resolve(continue_from(id, "again"), None).await);
            assert_eq!(refused.status(), StatusCode::NOT_FOUND, "{id}");
        }
    }

    /// A body dropped before its `message_stop` (the client went away) gives
    /// its place back: no record stays under the dropped id and the thread
    /// cannot be continued from it.
    #[tokio::test]
    async fn a_dropped_stream_leaves_no_turn() {
        let threads = MessageThreads::new();
        let events = text_events("msg_dropped", "half");
        let start = sse(&events[..1]).len();
        let bytes = sse(&events);
        let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
        let answered = threads
            .stored(turn.expect("a turn"), streamed(&bytes, &[start]))
            .await;
        let mut data = answered.into_body().into_data_stream();
        let first = data
            .next()
            .await
            .expect("a frame")
            .expect("the start event");
        assert!(first.starts_with(b"event: message_start"), "{first:?}");
        let reserved = threads
            .store
            .get_response(&ResponseId::from("msg_dropped"))
            .await
            .expect("the store");
        assert!(reserved.is_some(), "the place is taken at message_start");
        drop(data);
        let kept = threads
            .store
            .get_response(&ResponseId::from("msg_dropped"))
            .await
            .expect("the store");
        assert!(kept.is_none(), "the dropped place is given back");
        let refused = refusal(
            threads
                .resolve(continue_from("msg_dropped", "again"), None)
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
    }

    /// A thread longer than the history store's default chain depth of 100
    /// is read back whole: the walk goes as deep as the store keeps turns.
    #[tokio::test]
    async fn a_long_thread_is_read_back_whole() {
        let threads = MessageThreads::new();
        let last = thread(&threads, None, 120, "long").await;
        let (rebuilt, _) = accepted(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "turn 121"}],
                        "thread": {"type": "continue", "previous_message_id": last}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(rebuilt.messages.len(), 241);
    }
}
