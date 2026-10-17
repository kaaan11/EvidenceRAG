// Run from the repository root: node tests/fixtures/answer_contract/ui_warnings.cjs
// This small DOM stub checks rendering logic, not browser layout or model quality.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function element() {
    return {
        children: [], style: {}, textContent: "", className: "",
        addEventListener() {}, setAttribute() {},
        classList: { add() {}, remove() {}, toggle() {} },
        append(...items) { this.children.push(...items); },
        appendChild(item) { this.children.push(item); },
        replaceChildren(...items) { this.children = items; },
    };
}
const elements = new Map();
const document = {
    getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
    },
    createElement: element,
};
const context = vm.createContext({ document, window: { addEventListener() {} } });
const sourcePath = process.argv[2] || path.resolve(__dirname, "../../..", "static/app.js");
vm.runInContext(fs.readFileSync(sourcePath, "utf8"), context);
const warnings = (value) => Array.from(context.verificationWarnings(value));
let passed = 0;
function check(name, test) { test(); console.log(`PASS ${name}`); passed++; }

check("T01-A07 null and missing verification fields", () => {
    for (const value of [undefined, null, {}, { unsupported_numbers: [], invalid_citations: [] },
        { question_only_numbers: [], unsupported_numbers: [], invalid_citations: [] }]) {
        assert.deepEqual(warnings(value), []);
    }
});
check("T01-A05 question-only warning explains provenance", () => {
    const text = warnings({ question_only_numbers: ["2025"] }).join(" ");
    assert.match(text, /2025/);
    assert.match(text, /yalnız soruda geçiyor; kaynakta bulunamadı/);
});
check("T01-A07 old API warnings still render", () => {
    const result = warnings({ unsupported_numbers: ["2026"], invalid_citations: [4] });
    assert.equal(result.length, 2);
    assert.match(result[0], /2026/);
    assert.match(result[1], /Kanıt 4/);
});
check("T01-A07 raw answer is rendered unchanged with a warning", () => {
    const raw = "  2025 <script>sentetik</script> [Evidence 1]  ";
    context.renderAnswer(raw, [], { question_only_numbers: ["2025"] });
    const content = elements.get("answerCard").children[1];
    assert.equal(content.children[1].textContent, raw);
    assert.equal(content.children[2].className, "answer-warning");
});
check("T01-A06 cited badge explains membership without semantic claim", () => {
    context.renderSources([{ cited: true, score: 0.9, evidence_number: 2, text: "İlgisiz kaynak." }]);
    const badge = elements.get("sourcesList").children[0].children[0].children[1].children[0];
    assert.match(badge.title, /iddianın doğruluğu denetlenmedi/);
});
console.log(`${passed} UI contract checks passed (stubbed DOM; no browser).`);
