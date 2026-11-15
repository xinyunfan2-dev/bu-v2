/** Local bilingual retrieval. Returns original substrings, never translated evidence. */
const concepts = [
  ["延误", "延誤", "delay", "delayed"],
  ["空置", "无人居住", "無人居住", "家里没人", "家裏沒人", "unoccupied", "vacant", "unattended"],
  ["除外", "不保", "exclusion", "excluded", "not covered", "does not cover"],
  ["索赔", "索償", "索偿", "claim", "claims"],
  ["证明", "證明", "proof", "evidence", "certificate"],
  ["自负额", "自負額", "垫底", "墊底", "免赔", "deductible", "excess"],
  ["限额", "限額", "上限", "limit", "maximum", "cap"],
  ["附表", "schedule"], ["批单", "批單", "endorsement"],
  ["漏水", "渗水", "滲水", "leak", "leakage", "seepage"],
];
const normalize = (s: string) => s.normalize("NFKC").toLowerCase().replace(/\s+/g, " ").trim();
// English terms match words, not e.g. “claim” in “disclaimer”. Chinese matches substrings.
function position(text: string, term: string) {
  let at = text.indexOf(term);
  while (at >= 0) {
    if (!/^[a-z ]+$/.test(term) ||
      (!/[a-z]/.test(text[at - 1] ?? "") && !/[a-z]/.test(text[at + term.length] ?? ""))) return at;
    at = text.indexOf(term, at + 1);
  }
  return -1;
}
export function searchPolicySources<T extends { sourceId: string; documentId: string; quote: string }>(sources: T[], query: string): T[] {
  const terms = [...new Set(query.split(/[\s,，;；]+/).map(normalize).filter(Boolean))];
  if (!terms.length) return [];
  const groups = terms.map(term => [...new Set([term, ...concepts.filter(group =>
    group.some(alias => position(term, alias) >= 0)).flat()])]);
  const ranked = sources.map((source, order) => {
    const text = normalize(source.quote);
    let score = 0;
    groups.forEach((aliases, i) => {
      if (position(text, terms[i]) >= 0) score += 4;
      else if (aliases.some(alias => position(text, alias) >= 0)) score += 2;
    });
    return { source, score, order };
  }).filter(hit => hit.score > 0).sort((a, b) => b.score - a.score || a.order - b.order);
  const seen = new Set<string>();
  const result: T[] = [];
  for (const { source } of ranked) {
    const key = source.documentId + ":" + normalize(source.quote);
    if (seen.has(key)) continue;
    seen.add(key);
    // Keep the hit visible even when it occurs late in a long paragraph.
    const text = source.quote.toLowerCase();
    const matches = groups.flat().map(term => position(text, term)).filter(at => at >= 0);
    const start = Math.max(0, (matches.length ? Math.min(...matches) : 0) - 100);
    result.push({ ...source, quote: source.quote.slice(start, start + 640) });
    if (result.length === 8) break;
  }
  return result;
}
