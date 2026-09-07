// FORGE v2 verification: the carrot-cake prompt that surfaced both failure
// modes, plus the 4-turn accumulation case that reproduced the loops.
//
// Checks three things the v1 build got wrong:
//   * structural repetition across turns (measured, not eyeballed)
//   * yeast in a carrot-cake recipe -- carrot cake is chemically leavened, so
//     yeast/dough/knead/rise are hallucination markers
//   * that factual recall survived
import Foundation
import HelixEngine

func line(_ s: String) { print(s) }

func repetitionRate(_ text: String, n: Int = 5) -> Double {
    let w = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
    guard w.count > n else { return 0 }
    var seen = Set<String>(); var rep = 0; var tot = 0
    for i in 0...(w.count - n) {
        tot += 1
        if !seen.insert(w[i..<(i + n)].joined(separator: " ").lowercased()).inserted { rep += 1 }
    }
    return tot > 0 ? Double(rep) / Double(tot) : 0
}

func longestSpan(_ text: String) -> Int {
    let w = text.split(whereSeparator: { $0.isWhitespace }).map { $0.lowercased() }
    guard w.count > 2 else { return 0 }
    for n in stride(from: min(80, w.count / 2), through: 3, by: -1) {
        var seen = Set<String>()
        for i in 0...(w.count - n) {
            if !seen.insert(w[i..<(i + n)].joined(separator: " ")).inserted { return n }
        }
    }
    return 0
}

/// Carrot cake uses baking soda/powder. Yeast, dough, kneading and proving all
/// belong to bread and are the v1 build's signature confusion.
let breadMarkers = ["yeast", "knead", "dough", "proof the", "proving", "rise for", "gluten"]
func hallucinationMarkers(_ t: String) -> [String] {
    let low = t.lowercased()
    return breadMarkers.filter { low.contains($0) }
}

// Same prompt, same settings, both builds -- the earlier v1 numbers used a
// different seed prompt and were not comparable.
// MODEL=v1 points at the uniformly-ternary build for A/B comparison; both
// resolve through $HELIX_MODEL_PATH / the executable directory by default.
let env = ProcessInfo.processInfo.environment
let v2 = EngineConfig.defaultModelPath
let v1 = env["HELIX_MODEL_PATH_V1"] ?? v2.replacingOccurrences(of: "forge-v2", with: "forge-mixed")
let which = env["MODEL"] == "v1" ? v1 : v2

let engine = LlamaEngine(config: EngineConfig(modelPath: which, maxOutputTokens: 400))
do { try await engine.load() } catch { line("LOAD FAILED: \(error)"); exit(1) }
line("model: \(which.split(separator: "/").last!)")
line("HELIX backend: \(HelixProbe.backend(for: .falconH1_7B).rawValue)")

func turn(_ prompt: String) async -> String {
    var out = ""
    for await ev in await engine.generate(prompt: prompt) {
        if case .token(let t) = ev { out += t }
        if case .failed(let w) = ev { line("FAILED: \(w)"); exit(1) }
    }
    return out
}

line("\n=== 1. carrot cake, single turn ===")
let cake = await turn("Make me a carrot cake.")
line(String(cake.prefix(700)))
let m1 = hallucinationMarkers(cake)
line(String(format: "\n  repetition %.1f%%  span %d  bread-markers: %@",
            repetitionRate(cake) * 100, longestSpan(cake),
            (m1.isEmpty ? "none" : m1.joined(separator: ", ")) as NSString))

line("\n=== 2. four accumulated turns (the case that looped in v1) ===")
var all = cake
var perTurn = [repetitionRate(cake)]
for _ in 0..<3 {
    let t = await turn("Continue with more detail.")
    perTurn.append(repetitionRate(t))
    all += " " + t
}
let within = perTurn.map { String(format: "%.0f%%", $0 * 100) }.joined(separator: " ")
line(String(format: "  within-turn[%@]  whole=%.1f%%  worst span=%d words",
            within as NSString, repetitionRate(all) * 100, longestSpan(all)))
let m2 = hallucinationMarkers(all)
line("  bread-markers across transcript: \(m2.isEmpty ? "none" : m2.joined(separator: ", "))")

line("\n=== 3. factual recall spot-check ===")
await engine.reset()
for q in ["What is the capital of Japan? Answer in one word.",
          "Who wrote the novel 1984? Answer with the name only.",
          "What is the chemical symbol for gold? Answer with the symbol only."] {
    let a = await turn(q).trimmingCharacters(in: .whitespacesAndNewlines)
    line("  Q: \(q)\n  A: \(a.prefix(90))")
    await engine.reset()
}

// ggml asserts at static-destructor time that every Metal residency set was
// released, and Swift never deinitialises globals -- so the engine has to be
// torn down explicitly or the process aborts after printing its results.
await engine.shutdown()
