import express, { type ErrorRequestHandler, type Request, type Response } from "express";
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { z } from "zod";
import multer from "multer";
import { Investigator, parseLanguage, snapshotFor } from "./agent.js";
import { DocumentQueue, documentsFor, documentFor, publicDocument, createTextMaterial, MAX_FILE_BYTES } from "./ingest/documents.js";
import { Store, type Review, type Run } from "./store.js";
import { reportView } from "./report.js";
import { buildQuestionChecklist } from "./export.js";
import { runV3Intake, streamV3Analysis, streamV3Claims, streamV3WhatIf } from "./v3.js";

class ApiError extends Error {
  constructor(readonly status: number, readonly code: string, message: string) { super(message); }
}

const contextSchema = z.object({
  destination: z.string().trim().max(120).optional(),
  traveller: z.string().trim().max(120).optional(),
  activities: z.array(z.string().trim().min(1).max(80)).max(20).optional(),
  concerns: z.string().trim().max(2000).optional(),
}).strict();
const runSchema = z.object({
  expectedRevision: z.number().int().positive(),
  kind: z.enum(["investigate", "scenario"]),
  goal: z.string().trim().min(1).max(2000),
  // Set to create a successor run that continues the report of an earlier run.
  resumeFromRunId: z.string().regex(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i).optional(),
}).strict();
const contextUpdateSchema = z.object({
  expectedRevision: z.number().int().positive(), context: contextSchema,
}).strict();
const v3IntakeBodySchema = z.object({
  lang: z.enum(["zh-HK", "zh-Hant", "zh-CN", "en"]).optional(),
  userNote: z.string().trim().max(800).optional(),
}).strict();
const v3StreamSchema = z.object({
  context: z.record(z.string(), z.array(z.string().trim().max(80)).max(12)).default({}),
  contextTags: z.array(z.string().trim().max(80)).max(24).default([]),
  userNote: z.string().trim().max(800).optional(),
  lang: z.enum(["zh-HK", "zh-Hant", "zh-CN", "en"]).optional(),
  resume: z.boolean().optional(),
}).strict();
const v3ClaimsSchema = z.object({
  lang: z.enum(["zh-HK", "zh-Hant", "zh-CN", "en"]).optional(),
}).strict();
const v3WhatIfSchema = z.object({
  question: z.string().trim().min(2).max(400),
  lang: z.enum(["zh-HK", "zh-Hant", "zh-CN", "en"]).optional(),
}).strict();
const normalizeV3Lang = (raw: string | undefined, fallbackHeader: string | undefined) => {
  if (raw === "en") return "en" as const;
  if (raw === "zh-CN") return "zh-CN" as const;
  if (raw === "zh-HK" || raw === "zh-Hant") return "zh-HK" as const;
  return parseLanguage(fallbackHeader);
};
const hash = (value: string) => createHash("sha256").update(value).digest("hex");
const publicReview = ({ owner: _owner, context, ...review }: Review) => {
  // Internal v3 caches stay on the server; they are large and not part of the public shape.
  const { v3Intake: _a, v3IntakeRevision: _b, v3Analysis: _c, v3AnalysisKey: _d, v3Claims: _e, v3ClaimsKey: _f, ...rest } = review as Review & Record<string, unknown>;
  return { ...rest, context: JSON.parse(context) };
};
const inputRequest = (run: Pick<Run, "id">) => ({
  id: `${run.id}:contract`, requestedRole: "contract", status: "open",
  message: "请上传保险合同，收到材料后才能开始调查。",
});
const publicRun = ({ requestHash: _hash, idempotencyKey: _key, ...run }: Run) => ({
  ...run, inputRequests: run.status === "awaiting_input" ? [inputRequest(run)] : [],
});
// Statuses after which no further event can arrive for a run.
const TERMINAL_STATUSES = new Set(["awaiting_input", "cancelled", "failed", "partial", "interrupted"]);

function cookieToken(req: Request): string | undefined {
  const value = req.headers.cookie?.split(";").map(s => s.trim()).find(s => s.startsWith("pl_session="))?.slice(11);
  return value && /^[a-f0-9]{64}$/.test(value) ? value : undefined;
}

export function createApp(store: Store) {
  const app = express();
  const documentQueue = new DocumentQueue(store);
  const investigator = new Investigator(store);
  app.locals.close = async () => { await investigator.close(); await documentQueue.close(); };
  documentQueue.wake();
  const upload = multer({ storage: multer.memoryStorage(), defParamCharset: "utf8", limits: {
    fileSize: MAX_FILE_BYTES, files: 1, fields: 1, parts: 2, fieldSize: 100,
  } }).single("file");
  app.disable("x-powered-by");
  app.use((req, res, next) => {
    res.setHeader("X-Request-Id", randomUUID());
    res.setHeader("Cache-Control", "no-store");
    // Local API: reject cross-origin writes. No public CORS access is enabled.
    if (!["GET", "HEAD", "OPTIONS"].includes(req.method) && req.headers.origin) {
      if (req.headers.origin !== `http://${req.headers.host}`) {
        return next(new ApiError(403, "ORIGIN_DENIED", "请求来源不受支持。"));
      }
    }
    next();
  });
  app.use(express.json({ limit: "1mb" }));

  function presentRun(run: Run) {
    const saved = store.db.prepare("SELECT output FROM investigations WHERE runId = ?").get(run.id);
    const report = saved?.output ? JSON.parse(String(saved.output)) : null;
    // The reason recorded with the last status event is the difference between "failed for no
    // stated reason" and "the model API was rate limited, try again later".
    const recorded = store.lastEvent(run.id, "run.status")?.data as {reason?: string} | null;
    const stopReason =
      run.status === "awaiting_input" ? (report ? "additional_information" : "missing_documents")
      : run.status === "cancelled" ? (recorded?.reason ?? "user_cancelled")
      : run.status === "partial" ? "semantic_review_pending"
      : run.status === "failed" || run.status === "interrupted" ? (recorded?.reason ?? null)
      : null;
    return {...publicRun(run), stopReason, ...(report ? {report, inputRequests:report.requests} : {})};
  }

  function requireReview(req: Request) {
    const token = cookieToken(req);
    const review = token && store.getReview(String(req.params.id), hash(token));
    if (!review) throw new ApiError(404, "REVIEW_NOT_FOUND", "找不到这次审阅，请在创建它的浏览器中打开。");
    return review;
  }
  function requireRun(req: Request) {
    const review = requireReview(req);
    const run = store.getRun(String(req.params.runId), review.id);
    if (!run) throw new ApiError(404, "RUN_NOT_FOUND", "找不到这次调查任务。");
    return run;
  }
  function assertRevision(review: Review, revision: number) {
    if (review.revision !== revision) throw new ApiError(409, "REVISION_CONFLICT", "材料或背景已更新，请重新获取审阅信息。");
  }

  app.get("/api/health", (_req, res) => {
    res.json({ ok: true, service: "Policy Lens Backend" });
  });

  app.post("/api/reviews", (req, res) => {
    let token = cookieToken(req);
    if (!token) {
      token = randomBytes(32).toString("hex");
      res.cookie("pl_session", token, { httpOnly: true, sameSite: "strict", maxAge: 7 * 86400_000, path: "/" });
    }
    res.status(201).json(publicReview(store.createReview(hash(token))));
  });

  app.get("/api/reviews/:id", (req, res) => {
    const review = requireReview(req);
    const runs = store.db.prepare("SELECT * FROM runs WHERE reviewId = ? ORDER BY rowid DESC").all(review.id) as Run[];
    res.json({ ...publicReview(review), documents: documentsFor(store, review.id).map(publicDocument), runs: runs.map(presentRun),
      currentReport: reportView(store, review.id),
      reportHistory: store.db.prepare(`SELECT r.id AS runId, r.kind, r.goal, r.inputRevision, r.parentRunId, r.createdAt
        FROM runs r JOIN investigations i ON i.runId = r.id
        WHERE r.reviewId = ? AND i.output IS NOT NULL ORDER BY r.inputRevision DESC, r.rowid DESC`).all(review.id) });
  });

  app.get("/api/reviews/:id/reports/:reportId", (req, res) => {
    const review = requireReview(req);
    const report = reportView(store, review.id, String(req.params.reportId));
    if (!report) throw new ApiError(404, "REPORT_NOT_FOUND", "找不到这份调查报告。");
    res.json(report);
  });

  app.get("/api/reviews/:id/runs/:runId/activity", (req, res) => {
    const run = requireRun(req);
    // Public activity contains only recorded actions, never model reasoning or material text.
    res.json({runId: run.id, status: run.status, events: store.events(run).filter(event =>
      ["tool.started", "run.status", "run.resumed", "report.published"].includes(event.type))});
  });

  // The checklist is a plain download: no state changes, so a link is enough.
  app.get("/api/reviews/:id/questions/export", (req, res) => {
    const review = requireReview(req);
    const format = String(req.query.format ?? "md");
    if (format !== "md") throw new ApiError(400, "UNSUPPORTED_FORMAT", "目前只支持 format=md。");
    const selected = req.query.reportId;
    if (selected !== undefined && (typeof selected !== "string" || !/^[0-9a-f-]{36}$/i.test(selected))) throw new ApiError(400, "INVALID_REPORT_ID", "报告编号格式不正确。");
    const view = reportView(store, review.id, selected);
    if (!view) throw new ApiError(404, "REPORT_NOT_FOUND", "还没有可导出的调查结果，请先开始一次调查。");
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("Content-Disposition", 'attachment; filename="policy-questions.md"');
    res.type("text/markdown; charset=utf-8").send(buildQuestionChecklist(view));
  });

  function requireDocument(req: Request) {
    const review = requireReview(req);
    const document = documentFor(store, review.id, String(req.params.docId));
    if (!document) throw new ApiError(404, "DOCUMENT_NOT_FOUND", "找不到这份材料。");
    return document;
  }

  app.post("/api/reviews/:id/documents", (req, _res, next) => {
    requireReview(req); next();
  }, upload, (req, res) => {
    const review = requireReview(req);
    const { role } = z.object({ role: z.enum(["contract", "promotion", "schedule", "endorsement", "broker_reply"]) }).strict().parse(req.body);
    const file = req.file;
    if (!file) throw new ApiError(400, "FILE_REQUIRED", "请选择一份 PDF 文件。");
    if (!file.originalname.toLowerCase().endsWith(".pdf") || file.buffer.subarray(0, 5).toString("ascii") !== "%PDF-") {
      throw new ApiError(415, "PDF_REQUIRED", "当前仅支持 PDF，请上传真正的 PDF 文件。");
    }
    const sha256 = createHash("sha256").update(file.buffer).digest("hex");
    let duplicate = false;
    const document = store.transaction(() => {
      const existing = documentsFor(store, review.id);
      const same = existing.find(doc => doc.sha256 === sha256 && doc.role === role);
      if (same) { duplicate = true; return same; }
      if (existing.length >= 8) throw new ApiError(413, "DOCUMENT_LIMIT", "一次审阅最多保留 8 份材料，请先删除不需要的文件。");
      const id = randomUUID();
      const name = file.originalname.replace(/[\\/\u0000-\u001f\u007f]/g, "_").slice(0, 240);
      store.db.prepare("INSERT INTO documents (id, reviewId, role, name, sha256, byteSize, content, status, createdAt) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?)")
        .run(id, review.id, role, name, sha256, file.size, file.buffer, new Date().toISOString());
      return documentFor(store, review.id, id)!;
    });
    res.status(duplicate ? 200 : 202).json(publicDocument(document));
    documentQueue.wake();
  });

  // Pasted chat records: same review, same source/citation rules as an uploaded PDF, but the
  // provenance is recorded instead of inferred from a file.
  app.post("/api/reviews/:id/text-materials", (req, res) => {
    const review = requireReview(req);
    const input = z.object({
      role: z.enum(["contract", "promotion", "schedule", "endorsement", "broker_reply"]),
      fromWhom: z.string().trim().min(1).max(120),
      capturedAt: z.string().trim().min(1).max(40),
      text: z.string().trim().min(1).max(20_000),
    }).strict().parse(req.body);
    const sha256 = createHash("sha256").update(input.text).digest("hex");
    const existing = documentsFor(store, review.id);
    const same = existing.find(doc => doc.sha256 === sha256 && doc.role === input.role);
    if (same) return void res.status(200).json(publicDocument(same));
    if (existing.length >= 8) throw new ApiError(413, "DOCUMENT_LIMIT", "一次审阅最多保留 8 份材料，请先删除不需要的文件。");
    res.status(201).json(publicDocument(createTextMaterial(store, { reviewId: review.id, ...input })));
  });

  app.get("/api/reviews/:id/documents/:docId", (req, res) => {
    const document = requireDocument(req);
    const pages = store.db.prepare("SELECT pageIndex, printedPageLabel, status FROM pages WHERE documentId = ? ORDER BY pageIndex").all(document.id);
    res.json({ ...publicDocument(document), pages });
  });

  app.get("/api/reviews/:id/documents/:docId/pages/:pageIndex", (req, res) => {
    const document = requireDocument(req);
    const index = String(req.params.pageIndex);
    if (!/^[1-9]\d*$/.test(index) || !Number.isSafeInteger(Number(index))) throw new ApiError(400, "INVALID_PAGE", "页码应为从 1 开始的整数。");
    const page = store.db.prepare("SELECT pageIndex, printedPageLabel, text, status FROM pages WHERE documentId = ? AND pageIndex = ?").get(document.id, Number(index));
    if (!page) throw new ApiError(404, "PAGE_NOT_FOUND", "该页不存在或尚未完成提取。");
    const sources = store.db.prepare("SELECT id AS sourceId, blockId, quote FROM sources WHERE documentId = ? AND pageIndex = ? ORDER BY rowid").all(document.id, Number(index));
    res.json({ ...page, documentId: document.id, documentVersion: document.sha256, method: "pdf_text", sources });
  });

  app.get("/api/reviews/:id/sources/:sourceId", (req, res) => {
    const review = requireReview(req);
    const source = store.db.prepare(`SELECT s.id AS sourceId, s.documentId, s.pageIndex, s.blockId, s.quote,
      d.sha256 AS documentVersion, d.name AS documentName, d.role, p.printedPageLabel
      FROM sources s JOIN documents d ON d.id = s.documentId
      JOIN pages p ON p.documentId = s.documentId AND p.pageIndex = s.pageIndex
      WHERE s.id = ? AND d.reviewId = ?`).get(String(req.params.sourceId), review.id);
    if (!source) throw new ApiError(404, "SOURCE_NOT_FOUND", "找不到这段原文。");
    res.json({ ...source, method: "pdf_text", quoteCheck: "matched", contentUrl: `/api/reviews/${review.id}/documents/${source.documentId}/content#page=${source.pageIndex}` });
  });

  app.get("/api/reviews/:id/documents/:docId/content", (req, res) => {
    const document = requireDocument(req);
    const row = store.db.prepare("SELECT content FROM documents WHERE id = ?").get(document.id) as {content: Uint8Array};
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("Content-Security-Policy", "sandbox");
    if (document.mediaType === "text/plain") {
      // A pasted record has no PDF to point at: serve the stored text itself.
      res.setHeader("Content-Disposition", 'inline; filename="pasted-text.txt"');
      return void res.type("text/plain; charset=utf-8").send(Buffer.from(row.content));
    }
    res.setHeader("Content-Disposition", 'inline; filename="policy.pdf"');
    res.type("application/pdf").send(Buffer.from(row.content));
  });

  app.delete("/api/reviews/:id/documents/:docId", (req, res) => {
    store.transaction(() => {
      const document = requireDocument(req);
      for (const run of store.db.prepare("SELECT r.* FROM runs r JOIN investigations i ON i.runId = r.id WHERE r.reviewId = ?").all(document.reviewId) as Run[]) {
        investigator.cancel(run.id);
        store.db.prepare("UPDATE runs SET status = 'cancelled' WHERE id = ?").run(run.id);
        store.addEvent(run.id, "run.status", {status:"cancelled", reason:"input_changed"});
      }
      store.db.prepare("DELETE FROM investigations WHERE runId IN (SELECT id FROM runs WHERE reviewId = ?)").run(document.reviewId);
      store.db.prepare("DELETE FROM documents WHERE id = ?").run(document.id);
      store.db.prepare("UPDATE reviews SET revision = revision + 1 WHERE id = ?").run(document.reviewId);
    });
    res.status(204).end();
  });

  app.patch("/api/reviews/:id/context", (req, res) => {
    const body = contextUpdateSchema.parse(req.body);
    const updated = store.transaction(() => {
      const review = requireReview(req);
      assertRevision(review, body.expectedRevision);
      // PATCH merges supplied fields; omitted facts remain unknown or unchanged.
      const context = { ...JSON.parse(review.context), ...body.context };
      store.db.prepare("UPDATE reviews SET context = ?, revision = revision + 1 WHERE id = ?")
        .run(JSON.stringify(context), review.id);
      return store.getReview(review.id, review.owner)!;
    });
    res.json(publicReview(updated));
  });

  app.post("/api/reviews/:id/runs", (req, res) => {
    requireReview(req);
    const body = runSchema.parse(req.body);
    const key = req.get("Idempotency-Key") ?? null;
    if (key !== null && !/^[A-Za-z0-9_-]{8,128}$/.test(key)) throw new ApiError(400, "INVALID_IDEMPOTENCY_KEY", "幂等键应为 8–128 个字母、数字、下划线或连字符。");
    const requestHash = hash(JSON.stringify(body));
    let replay = false;
    const run = store.transaction(() => {
      const review = requireReview(req);
      if (key) {
        const existing = store.db.prepare("SELECT * FROM runs WHERE reviewId = ? AND idempotencyKey = ?").get(review.id, key) as Run | undefined;
        if (existing) {
          if (existing.requestHash !== requestHash) throw new ApiError(409, "IDEMPOTENCY_CONFLICT", "同一个幂等键不能用于不同请求。");
          replay = true;
          return existing;
        }
      }
      assertRevision(review, body.expectedRevision);
      let parent: Run | undefined;
      if (body.resumeFromRunId) {
        const earlier = store.getRun(body.resumeFromRunId, review.id);
        if (!earlier) throw new ApiError(404, "PARENT_RUN_NOT_FOUND", "找不到要接续的上一次任务。");
        if (["queued", "running"].includes(earlier.status)) throw new ApiError(409, "PARENT_RUN_ACTIVE", "上一次任务仍在执行，请等它结束后再续查。");
        const previous = store.db.prepare("SELECT output FROM investigations WHERE runId = ?").get(earlier.id) as {output: string | null} | undefined;
        if (!previous?.output) throw new ApiError(409, "PARENT_RUN_NOT_RESUMABLE", "上一次任务没有可接续的结论，请新建调查任务。");
        // A successor run must carry new input; otherwise it would repeat the same investigation.
        if (review.revision <= earlier.inputRevision) throw new ApiError(409, "NO_NEW_INPUT", "材料或背景自上次任务后没有更新，无需续查。");
        parent = earlier;
      }
      if (documentsFor(store, review.id).some(doc => doc.role === "contract" && ["ready", "partial"].includes(doc.status))) {
        if (!investigator.configured()) throw new ApiError(503, "AGENT_NOT_CONNECTED", "请在本机 .env 配置 DEEPSEEK_API_KEY 后重启后端。");
      }
      const id = randomUUID(), now = new Date().toISOString();
      // The model runner is a later step. Do not manufacture a successful report.
      store.db.prepare(`INSERT INTO runs (id, reviewId, inputRevision, kind, goal, status, idempotencyKey, requestHash, parentRunId, createdAt, updatedAt)
        VALUES (?, ?, ?, ?, ?, 'awaiting_input', ?, ?, ?, ?, ?)`)
        .run(id, review.id, review.revision, body.kind, body.goal, key, requestHash, parent?.id ?? null, now, now);
      if (parent) store.addEvent(id, "run.resumed", { parentRunId: parent.id });
      const snapshot = snapshotFor(store, review.id);
      if (snapshot.documents.some(doc => ["queued", "parsing"].includes(doc.status))) throw new ApiError(409, "DOCUMENTS_PROCESSING", "请等待材料解析完成。");
      const usable = snapshot.documents.some(doc => doc.role === "contract" && ["ready", "partial"].includes(doc.status));
      if (usable) {
        if (store.db.prepare("SELECT id FROM runs WHERE reviewId = ? AND status IN ('queued','running') AND id != ?").get(review.id, id)) throw new ApiError(409, "RUN_ACTIVE", "已有调查正在执行。");
        store.db.prepare("UPDATE runs SET status = 'queued' WHERE id = ?").run(id);
        store.db.prepare("INSERT INTO investigations (runId, snapshot) VALUES (?, ?)").run(id, JSON.stringify(snapshot));
        store.addEvent(id, "run.status", {status:"queued"});
        return store.getRun(id, review.id)!;
      }
      const created = store.getRun(id, review.id)!;
      store.addEvent(id, "run.status", { status: "awaiting_input", reason: "missing_documents" });
      store.addEvent(id, "input.required", inputRequest(created));
      return created;
    });
    // The report is written in the language the page is currently showing, so the browser tells us
    // which one via Accept-Language; unknown values fall back to Traditional Chinese.
    if (!replay && run.status === "queued") investigator.start(run, parseLanguage(req.get("accept-language")));
    res.status(replay ? 200 : 202).json({ ...presentRun(run), eventsUrl: `/api/reviews/${run.reviewId}/runs/${run.id}/events` });
  });

  app.get("/api/reviews/:id/runs/:runId", (req, res) => {
    const run = requireRun(req);
    res.json(presentRun(run));
  });

  // A real stream: replay what already happened, then keep the connection open and push every new
  // event until the run reaches a terminal status. A heartbeat keeps proxies and browsers from
  // dropping an idle connection while the model is thinking.
  app.get("/api/reviews/:id/runs/:runId/events", (req, res) => {
    const run = requireRun(req);
    const raw = req.query.after ?? req.get("Last-Event-ID") ?? "0";
    if (typeof raw !== "string" || !/^\d+$/.test(raw) || !Number.isSafeInteger(Number(raw))) throw new ApiError(400, "INVALID_CURSOR", "事件游标必须是非负整数。");
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("X-Accel-Buffering", "no");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");

    let closed = false;
    let heartbeat: NodeJS.Timeout | undefined;
    let unsubscribe: () => void = () => {};
    const send = (event: {eventId: number; type: string; data: unknown}) => {
      res.write(`id: ${event.eventId}\nevent: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
    };
    const finish = () => {
      if (closed) return;
      closed = true;
      if (heartbeat) clearInterval(heartbeat);
      unsubscribe();
      res.end();
    };
    const current = store.getRun(run.id, run.reviewId);
    let replayEnded = !current || TERMINAL_STATUSES.has(current.status);

    // The whole replay is written first: a run that already finished may still have events after
    // its terminal status event, and a reconnecting client must not lose them.
    for (const event of store.events(run, Number(raw))) {
      send(event);
      if (event.type === "run.status" && TERMINAL_STATUSES.has((event.data as {status?: string})?.status ?? "")) {
        replayEnded = true;
      }
    }
    if (replayEnded) { res.end(); return; }

    unsubscribe = store.subscribe(event => {
      if (closed || event.runId !== run.id) return;
      send(event);
      if (event.type === "run.status" && TERMINAL_STATUSES.has((event.data as {status?: string})?.status ?? "")) finish();
    });
    heartbeat = setInterval(() => { if (!closed) res.write(": ping\n\n"); }, 15_000);
    // Headers are only flushed once the subscription is in place, so a client that has received
    // the response cannot miss an event that is emitted immediately afterwards.
    res.flushHeaders?.();
    req.on("close", () => {
      closed = true;
      if (heartbeat) clearInterval(heartbeat);
      unsubscribe();
    });
  });

  app.post("/api/reviews/:id/runs/:runId/cancel", (req, res) => {
    const run = store.transaction(() => {
      const current = requireRun(req);
      investigator.cancel(current.id);
      if (current.status !== "cancelled") {
        store.db.prepare("UPDATE runs SET status = 'cancelled', updatedAt = ? WHERE id = ?").run(new Date().toISOString(), current.id);
        store.addEvent(current.id, "run.status", { status: "cancelled", reason: "user_cancelled" });
      }
      return store.getRun(current.id, current.reviewId)!;
    });
    res.json(publicRun(run));
  });

  app.post("/api/reviews/:id/intake", async (req, res, next) => {
    try {
      const review = requireReview(req);
      const body = v3IntakeBodySchema.safeParse(req.body ?? {});
      if (!body.success) throw new ApiError(400, "INVALID_V3_INTAKE", "預檢請求參數無效。");
      const snapshot = snapshotFor(store, review.id);
      if (snapshot.documents.some(doc => ["queued", "parsing"].includes(doc.status))) {
        throw new ApiError(409, "DOCUMENTS_PROCESSING", "請等待文件解析完成。");
      }
      if (!snapshot.documents.some(doc => doc.role === "contract" && ["ready", "partial"].includes(doc.status))) {
        throw new ApiError(409, "MISSING_CONTRACT", "請先上傳保單文件。");
      }
      const lang = normalizeV3Lang(body.data.lang, req.get("accept-language"));
      const intake = await runV3Intake(store, review.id, lang, body.data.userNote);
      res.json(intake);
    } catch (error) {
      next(error);
    }
  });

  /**
   * Run one v3 job as a Server-Sent Events response.
   * `res` "close" (not `req` "close") marks a real disconnect: on Node 16+ `req` "close" fires as
   * soon as the JSON body has been read, which used to abort the agent at once and leave the
   * response open forever. The job is also capped at 150 s and the response is always ended.
   */
  async function runV3Sse(req: Request, res: Response, job: (signal: AbortSignal, emit: (event: Record<string, unknown>) => void) => Promise<void>, lang: string) {
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("X-Accel-Buffering", "no");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders?.();

    const controller = new AbortController();
    const timeout = AbortSignal.timeout(150_000);
    const signal = AbortSignal.any([controller.signal, timeout]);
    let closed = false;
    const heartbeat = setInterval(() => { if (!closed) res.write(": ping\n\n"); }, 12_000);
    res.on("close", () => {
      closed = true;
      clearInterval(heartbeat);
      controller.abort();
    });
    const emit = (event: Record<string, unknown>) => {
      if (closed) return;
      res.write(`data: ${JSON.stringify(event)}\n\n`);
    };
    try {
      await job(signal, emit);
    } catch (error) {
      if (!closed) {
        const message = timeout.aborted
          ? (lang === "en" ? "This took too long. Please retry." : "處理超時，請重試。")
          : error instanceof Error && error.message === "EVIDENCE_REVIEW_FAILED"
            ? (lang === "en" ? "The evidence check could not verify the report. Please retry." : "未能核實報告的原文依據，請重試。")
          : error instanceof Error && error.message === "REPORT_GENERATION_FAILED"
            ? (lang === "en" ? "The AI reply came back garbled. Please retry." : "AI 回覆格式出錯，請重試。")
            : error instanceof Error ? error.message : "處理過程發生錯誤，請重試。";
        emit({ type: "error", message });
      }
    } finally {
      clearInterval(heartbeat);
      if (!res.writableEnded) {
        closed = true;
        res.end();
      }
    }
  }

  function v3Ready(review: Review) {
    const snapshot = snapshotFor(store, review.id);
    if (snapshot.documents.some(doc => ["queued", "parsing"].includes(doc.status))) {
      throw new ApiError(409, "DOCUMENTS_PROCESSING", "請等待文件解析完成。");
    }
    if (!snapshot.documents.some(doc => doc.role === "contract" && ["ready", "partial"].includes(doc.status))) {
      throw new ApiError(409, "MISSING_CONTRACT", "請先上傳保單文件。");
    }
    if (!investigator.configured()) {
      throw new ApiError(503, "AGENT_NOT_CONNECTED", "請在本機 .env 配置 DEEPSEEK_API_KEY 後重啟服務。");
    }
  }

  app.post("/api/reviews/:id/v3-stream", async (req, res, next) => {
    let review: Review;
    let body: z.infer<typeof v3StreamSchema>;
    try {
      review = requireReview(req);
      const parsed = v3StreamSchema.safeParse(req.body ?? {});
      if (!parsed.success) throw new ApiError(400, "INVALID_V3_STREAM", "解讀請求參數無效。");
      body = parsed.data;
      v3Ready(review);
    } catch (error) { return next(error); }
    const lang = normalizeV3Lang(body.lang, req.get("accept-language"));
    await runV3Sse(req, res, (signal, emit) => streamV3Analysis({
      store, reviewId: review.id, contextSelections: body.context, contextTags: body.contextTags,
      userNote: body.userNote, lang, resume: body.resume, signal, emit,
    }), lang);
  });

  app.post("/api/reviews/:id/v3-claims", async (req, res, next) => {
    let review: Review;
    let lang: ReturnType<typeof normalizeV3Lang>;
    try {
      review = requireReview(req);
      const parsed = v3ClaimsSchema.safeParse(req.body ?? {});
      if (!parsed.success) throw new ApiError(400, "INVALID_V3_CLAIMS", "核對請求參數無效。");
      v3Ready(review);
      lang = normalizeV3Lang(parsed.data.lang, req.get("accept-language"));
    } catch (error) { return next(error); }
    await runV3Sse(req, res, (signal, emit) => streamV3Claims({ store, reviewId: review.id, lang, signal, emit }), lang);
  });

  app.post("/api/reviews/:id/v3-whatif", async (req, res, next) => {
    let review: Review;
    let body: z.infer<typeof v3WhatIfSchema>;
    try {
      review = requireReview(req);
      const parsed = v3WhatIfSchema.safeParse(req.body ?? {});
      if (!parsed.success) throw new ApiError(400, "INVALID_V3_WHATIF", "請用一兩句話描述你的情況。");
      body = parsed.data;
      v3Ready(review);
    } catch (error) { return next(error); }
    const lang = normalizeV3Lang(body.lang, req.get("accept-language"));
    await runV3Sse(req, res, (signal, emit) => streamV3WhatIf({ store, reviewId: review.id, question: body.question, lang, signal, emit }), lang);
  });

  app.delete("/api/reviews/:id", (req, res) => {
    const review = requireReview(req);
    store.db.prepare("DELETE FROM reviews WHERE id = ?").run(review.id);
    res.status(204).end();
  });

  app.use((_req, _res, next) => next(new ApiError(404, "NOT_FOUND", "接口不存在。")));
  const errorHandler: ErrorRequestHandler = (error, _req, res, _next) => {
    if (error instanceof multer.MulterError) {
      const tooLarge = error.code === "LIMIT_FILE_SIZE";
      res.status(tooLarge ? 413 : 400).json({ error: {
        code: tooLarge ? "FILE_TOO_LARGE" : "INVALID_UPLOAD",
        message: tooLarge ? "单份 PDF 不能超过 10 MB。" : "请仅上传一份 file 文件，并填写 role。",
        retryable: false, requestId: res.getHeader("X-Request-Id"),
      } });
      return;
    }
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: { code: "INVALID_INPUT", message: "请求参数不正确。", issues: error.issues.map(i => ({ path: i.path, message: i.message })), retryable: false, requestId: res.getHeader("X-Request-Id") } });
      return;
    }
    const status = error instanceof ApiError ? error.status : error?.type === "entity.too.large" ? 413 : error?.type === "entity.parse.failed" ? 400 : 500;
    const code = error instanceof ApiError ? error.code : status === 413 ? "BODY_TOO_LARGE" : status === 400 ? "INVALID_JSON" : "INTERNAL_ERROR";
    const message = error instanceof ApiError ? error.message : status === 413 ? "请求内容过大。" : status === 400 ? "请求不是有效的 JSON。" : "服务暂时无法处理请求。";
    res.status(status).json({ error: { code, message, retryable: status === 500, requestId: res.getHeader("X-Request-Id") } });
  };
  app.use(errorHandler);
  return app;
}
