import test from "node:test";
import assert from "node:assert/strict";
import { sampleSourcesForIntake, resolveOutputLanguage, alignCitation } from "../v3.js";

test("resolveOutputLanguage strictly follows UI language unless userNote switches language", () => {
  assert.equal(resolveOutputLanguage("en", ""), "en");
  assert.equal(resolveOutputLanguage("zh-HK", ""), "zh-HK");
  assert.equal(resolveOutputLanguage("zh-CN", ""), "zh-CN");

  // UI is English, user types Chinese -> overrides to Traditional Chinese
  assert.equal(resolveOutputLanguage("en", "去旅行超過一個月屋企無人住賠唔賠？"), "zh-HK");

  // UI is Chinese, user types English sentence -> overrides to English
  assert.equal(resolveOutputLanguage("zh-HK", "What happens if my flat is unoccupied for 45 days?"), "en");

  // Same language stays unchanged
  assert.equal(resolveOutputLanguage("zh-HK", "漏水墊底費幾多錢？"), "zh-HK");
  assert.equal(resolveOutputLanguage("en", "Does it cover laptop theft outside home?"), "en");
});

test("sampleSourcesForIntake spreads across opening, middle and closing pages", () => {
  const sources = Array.from({ length: 54 }, (_, i) => {
    const pageIndex = i + 1;
    let quote = `General paragraph on page ${pageIndex}.`;
    if (pageIndex === 1) quote = "PREMIER HOME COMPREHENSIVE INSURANCE POLICY";
    if (pageIndex === 12) quote = "This Policy does not cover wear and tear or gradual seepage.";
    if (pageIndex === 35) quote = "未得本公司書面同意下連續 14 日以上無人居住的住所內家居財物損失不保。";
    if (pageIndex === 50) quote = "賠償限額表：家居財物最高賠償額 HK$1,000,000，每件上限 HK$5,000。";
    return { sourceId: `doc:p${pageIndex}:b1`, pageIndex, quote };
  });

  const sampled = sampleSourcesForIntake(sources);
  assert.ok(sampled.length >= 6 && sampled.length <= 16);
  const pages = sampled.map(s => s.pageIndex);
  assert.ok(pages.includes(1), "includes opening page");
  assert.ok(pages.includes(35), "includes Chinese exclusion/unoccupied threshold in second half");
  assert.ok(pages.includes(50), "includes schedule table near end of document");
});

test("alignCitation resolves verbatim substring across whitespace differences", () => {
  const sources = [
    { sourceId: "doc:p25:b1", pageIndex: 25, quote: "1. HOME CONTENTS\n300,000/event   600,000/event   1,000,000/event" },
    { sourceId: "doc:p33:b2", pageIndex: 33, quote: "保障因住所被爆竊或行劫而引致現金損失，惟必須在發現該等損失後 24 小時內報警。" },
  ];
  const hit = alignCitation("wrong-id", "保障因住所被爆竊或行劫而引致現金損失，惟必須在發現該等損失後24小時內報警。", sources);
  assert.equal(hit.sourceId, "doc:p33:b2");
  assert.equal(hit.page, 33);
});

test("alignCitation flags a quote that is not in the policy instead of passing off another sentence", async () => {
  const { alignCitation } = await import("../v3.js");
  const sources = [
    { sourceId: "d1:p1:b1", pageIndex: 1, quote: "任何索償通知書必須在意外發生後十四天內送交本公司。" },
    { sourceId: "d1:p2:b1", pageIndex: 2, quote: "門診每次港幣 150 元。" },
  ];
  const hit = alignCitation("d1:p1:b1", "必須在意外發生後十四天內", sources);
  assert.equal(hit.found, true);
  assert.equal(hit.page, 1);
  const elsewhere = alignCitation("d1:p1:b1", "門診每次港幣 150 元", sources);
  assert.equal(elsewhere.found, true);
  assert.equal(elsewhere.sourceId, "d1:p2:b1");
  const missing = alignCitation("d1:p1:b1", "三個月內通知就可以", sources);
  assert.equal(missing.found, false);
  const strict = alignCitation("", "三個月內通知就可以", sources, { strict: true });
  assert.equal(strict.found, false);
  assert.equal(strict.quote, "三個月內通知就可以");
});

test("toWrittenChinese rewrites Cantonese colloquialisms only", async () => {
  const { toWrittenChinese } = await import("../v3.js");
  assert.equal(toWrittenChinese("屋企冇人，睇醫生唔賠嘅"), "家裏沒有人，看醫生不賠的");
  assert.equal(toWrittenChinese("門診每次只賠 150 元"), "門診每次只賠 150 元");
});

// Small gold retrieval set: relevant clauses deliberately occur after common distractors.
import { searchPolicySources } from "../retrieval.js";
import { reviewBeforePublish } from "../evidence-review.js";
import { assertReviewedCitations } from "../v3.js";
import { z } from "zod";

const retrievalGold = [
  { query: "家里没人", quote: "Loss while the home is unoccupied for over 30 days is excluded." },
  { query: "延误", quote: "Flight delay of at least six hours is required." },
  { query: "免赔", quote: "An excess of HKD 500 applies to each loss." },
  { query: "索赔 证明", quote: "Claims require proof of purchase and a carrier certificate." },
  { query: "批单", quote: "This endorsement changes the cover stated in the schedule." },
];
for (const fixture of retrievalGold) {
  test(`retrieval gold: ${fixture.query} finds the bilingual clause`, () => {
    const sources = Array.from({ length: 30 }, (_, i) => ({
      sourceId: `doc:p${i + 1}:b1`, documentId: "doc", quote: `General policy introduction ${i}.`,
    }));
    sources.push({ sourceId: "doc:p99:b1", documentId: "doc", quote: fixture.quote });
    assert.equal(searchPolicySources(sources, fixture.query)[0]?.sourceId, "doc:p99:b1");
  });
}

test("retrieval ranks multiple concepts above early single-term hits and deduplicates", () => {
  const sources = Array.from({ length: 12 }, (_, i) => ({ sourceId: `s${i}`, documentId: "d", quote: `Delay information ${i}` }));
  sources.push({ sourceId: "relevant", documentId: "d", quote: "Delay claims require a certificate as proof." });
  sources.push({ sourceId: "duplicate", documentId: "d", quote: "Delay claims require a certificate as proof." });
  const hits = searchPolicySources(sources, "delay proof");
  assert.equal(hits[0].sourceId, "relevant");
  assert.ok(!hits.some(hit => hit.sourceId === "duplicate"));
  assert.ok(hits.length <= 8);
});

test("retrieval keeps a late keyword visible as a verbatim substring", () => {
  const quote = "General text. ".repeat(150) + "Flight delay requires six hours.";
  const hits = searchPolicySources([{ sourceId: "late", documentId: "d", quote }], "delay");
  assert.ok(hits[0].quote.includes("Flight delay"));
  assert.ok(quote.includes(hits[0].quote));
});

test("retrieval does not match English keywords inside unrelated words or invent missing evidence", () => {
  const sources = [{ sourceId: "d:1", documentId: "d", quote: "Disclaimer: contact details." }];
  assert.deepEqual(searchPolicySources(sources, "claim"), []);
  assert.deepEqual(searchPolicySources(sources, "flood"), []);
  assert.deepEqual(searchPolicySources(sources, "  "), []);
});

const reviewedSchema = z.object({ answer: z.string() });
test("review gate publishes only the corrected result, after review completes", async () => {
  const published: unknown[] = [];
  const result = await reviewBeforePublish({ schema: reviewedSchema, draft: { answer: "Always paid" },
    signal: new AbortController().signal,
    review: async draft => {
      assert.equal(draft.answer, "Always paid");
      assert.deepEqual(published, []);
      return { answer: "Only if the stated conditions are met" };
    }, publish: item => published.push(item),
  });
  assert.deepEqual(published, [result]);
  assert.notEqual(result.answer, "Always paid");
});

test("review gate fails closed for unavailable, invalid, and unsupported evidence", async () => {
  for (const failure of ["unavailable", "invalid", "evidence"]) {
    const published: unknown[] = [];
    await assert.rejects(reviewBeforePublish({ schema: reviewedSchema, draft: { answer: "Draft" },
      signal: new AbortController().signal,
      review: async () => { if (failure === "unavailable") throw new Error("offline"); return failure === "invalid" ? {} : { answer: "Reviewed" }; },
      validate: () => { if (failure === "evidence") throw new Error("EVIDENCE_REVIEW_FAILED"); },
      publish: item => published.push(item),
    }));
    assert.deepEqual(published, []);
  }
});

test("review cancellation prevents publication before or during review", async () => {
  for (const alreadyAborted of [true, false]) {
    const controller = new AbortController();
    if (alreadyAborted) controller.abort();
    let calls = 0;
    await assert.rejects(reviewBeforePublish({ schema: reviewedSchema, draft: { answer: "Draft" }, signal: controller.signal,
      review: async () => { calls++; controller.abort(); return { answer: "Reviewed" }; },
      publish: () => assert.fail("cancelled review must not publish"),
    }));
    assert.equal(calls, alreadyAborted ? 0 : 1);
  }
});

test("reviewed citations reject invented wording and marketing used as contract evidence", () => {
  const contract = [{ sourceId: "c:p7:b1", pageIndex: 7, quote: "Delays of six hours require a carrier certificate." }];
  const promotion = [{ sourceId: "m:p1:b1", pageIndex: 1, quote: "Every delay is covered." }];
  assert.doesNotThrow(() => assertReviewedCitations({ scenarios: [{ sourceId: "c:p7:b1", quote: "Delays of six hours" }] }, contract));
  assert.throws(() => assertReviewedCitations({ sourceId: "c:p7:b1", quote: "Five hours always pays" }, contract), /EVIDENCE_REVIEW_FAILED/);
  assert.throws(() => assertReviewedCitations({ contractSourceId: "m:p1:b1", contractQuote: promotion[0].quote }, contract, promotion), /EVIDENCE_REVIEW_FAILED/);
  assert.doesNotThrow(() => assertReviewedCitations({ claimQuote: promotion[0].quote, contractSourceId: "c:p7:b1", contractQuote: "require a carrier certificate" }, contract, promotion));
});

test("review context includes adjacent exclusions and claims conditions beyond the short draft quote", async () => {
  const { reviewEvidenceContext } = await import("../v3.js");
  const sources = [
    { sourceId: "c:p7:b1", documentId: "c", pageIndex: 7, role: "contract" as const, quote: "Delay benefit starts at six hours." },
    { sourceId: "c:p8:b1", documentId: "c", pageIndex: 8, role: "contract" as const, quote: "No benefit for a cancelled flight." },
    { sourceId: "c:p90:b1", documentId: "c", pageIndex: 90, role: "contract" as const, quote: "Claims require a carrier certificate." },
  ];
  const context = reviewEvidenceContext(sources, { sourceId: "c:p7:b1", quote: "Delay benefit" });
  assert.deepEqual(new Set(context.map(src => src.sourceId)), new Set(sources.map(src => src.sourceId)));
  assert.equal(new Set(context.map(src => src.sourceId)).size, context.length);
});

test("authoritative scenario, gap and claim classifications require citations", () => {
  for (const result of [
    { verdict: "supported", contractQuote: "" },
    { hasGap: true, tableQuote: "", detailQuote: "" },
    { status: "excluded", visualKind: "none", quote: "" },
  ]) assert.throws(() => assertReviewedCitations(result, []), /EVIDENCE_REVIEW_FAILED/);
  assert.doesNotThrow(() => assertReviewedCitations({ status: "ask", visualKind: "none", quote: "" }, []));
});
