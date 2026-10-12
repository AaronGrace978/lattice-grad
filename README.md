# Lattice-OS

**A multi-dimensional geometric AI runtime, packaged as a Tauri desktop app.**

Lattice-OS treats a codebase as a live, directed acyclic graph tensor instead of text files. Language
models from **Anthropic, OpenAI, Google Gemini and Ollama Cloud** do not write files. They emit tensor
projections `ΔT` that are compiled straight into an in-memory execution lattice. The lattice runs them,
isolates the nodes that fail, and guards the agent against oscillating between the same two answers.

```
[LLM token stream] -> [Continuous Graph Compiler] -> [Live Assembly Lattice]
                                                            |
[Loop-Breaking Matrix] <----- [Fluidity Routing Engine] <---+
```

## How the specification maps to code

| Concept | Implementation |
|---|---|
| `T_C = (V, E, W)` | `crates/lattice-core/src/graph.rs`. `V` holds `input` / `function` / `type` / `output` nodes. `E` holds port-bound data-flow edges. `W` holds per-node efficiency, velocity, reliability and latency plus per-edge flow, all written back after every run. |
| `T_C' = T_C + ΔT_agent` | `LatticeGraph::apply(&Delta)`: an immutable projection (`upsert_node`, `connect`, `remove_node`, probes…), validated for acyclicity, ports and boundaries before it is accepted. |
| Continuous Graph Compiler | `cgc.rs`: each node body compiles to an in-memory [Rhai](https://rhai.rs) AST, cached by source hash. A delta only recompiles the nodes it changed, and nothing touches the disk. |
| Fluidity Routing Engine | `fre.rs`: runs the lattice in a sandboxed engine with no I/O, no `eval`, and a per-node operation budget. A node that faults (compile error, exception, or exhausted budget) is wrapped in an isolation bubble. A fallback proxy (last-known-good output for the same inputs, then the node's declared `fallback`, then pass-through) keeps data flowing downstream. Each fault becomes a repair vector for the model. |
| Loop-Breaking Matrix | `lbm.rs`: every state gets a SHA-256 identity and every `ΔT` a direction in ℝ⁶⁴. It detects `A → B → A` revisits, re-entry into forbidden states, and anti-parallel swings (`cos(Δₜ, Δₜ₋₁) < −0.8`). On detection it rejects the delta, forbids the oscillating states, and picks the repair strategy most orthogonal to the oscillation axis (Gram–Schmidt rejection). That strategy is injected into the next prompt. |
| Embeddings | `embed.rs`: deterministic signed feature hashing of identifiers, ports and code tokens, plus PCA to 3D for the embedding-space view. It runs offline with no extra API calls. |
| Generation layer | `agent.rs` and `providers/`: streams the completion, extracts the `lattice` block, then runs the LBM check, CGC compile and FRE probe suite. The faults or the LBM directive feed the next iteration until every probe passes. |

## Providers and models

The Rust core handles all four providers natively and streams every one of them:

| Provider | API | Curated models (live list merged in at runtime) |
|---|---|---|
| Anthropic | Messages API (SSE) | Claude Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5, plus Fable 5 / Opus 5 / Sonnet 5 / Opus 4.8 |
| OpenAI | Responses API (SSE) | GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna, plus GPT-5.6 Sol / Luna and GPT-5.5 |
| Google Gemini | `streamGenerateContent` (SSE) | Gemini 3.8 Flash, 3.1 Pro (preview), 3.7 / 3.6 / 3.5 Flash, 3.5 / 3.1 Flash-Lite, 3 Flash (preview) |
| Ollama Cloud | `/api/chat` (NDJSON) | Kimi K3, GLM-5.3, DeepSeek V4 Pro, Kimi K2.7 Code, MiniMax M3, Mistral Large 4, Nemotron 3 Ultra / Super / Nano, DeepSeek V4.1 Flash, GLM-5.3 Flash, gpt-oss 120B / 20B, Gemma 4 31B, … |

Every provider's model endpoint is queried live. Models released after this build appear automatically
(marked `●`), and any model ID can be typed in by hand. You can override the Ollama base URL with
`http://localhost:11434` to use a local server without a key.

API keys are entered in **Settings** and stored in the OS app-config directory (`settings.json`, mode
600). They are read only by the Rust core and never sent back to the webview. If no key is saved, the
app falls back to `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` and `OLLAMA_API_KEY`.

## Getting started

Prerequisites: Rust ≥ 1.90, Node ≥ 20, and the
[Tauri system dependencies](https://v2.tauri.app/start/prerequisites/) for your OS. On Debian/Ubuntu:

```bash
sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file libxdo-dev \
  libssl-dev libayatana-appindicator3-dev librsvg2-dev
```

Then:

```bash
npm install
npm run tauri dev      # development app with hot reload
npm run tauri build    # release bundles (deb/rpm/AppImage, dmg, msi…)
```

To run the tests:

```bash
cargo test --workspace   # engine, providers, agent loop (scripted generator), settings
npm run typecheck
```

## Using the app

On first launch you get a small text-statistics lattice. One of its nodes, `avg_word_len`, has a latent
divide-by-zero. The `empty` probe trips it, the FRE isolates it in a bubble, and the downstream report
still gets a proxy value.

1. Choose a provider and model in the top bar, then add a key in **⚙ Settings**.
2. In the left console, describe a goal, for example *"Repair every isolated node so all probes pass"* or
   *"Add a `unique_words` metric with probes"*. Then press **Run agent**.
3. Each iteration shows the model's tokens streaming in, followed by the `ΔT` it proposed, the CGC
   compile stats, the FRE probe results, and any LBM intervention.
4. Inspect and edit nodes, ports, probes and live invocations on the right. Manual edits go through the
   same `ΔT` pipeline.
5. The bottom strip records the tensor history (click a revision to restore it) and the LBM's
   interventions, including the orthogonal vector it injected.

The center view switches between the **live assembly lattice** (a DAG with isolation bubbles and
flow-weighted edges), the **embedding space** (nodes projected onto their principal components), and
the raw **tensor W** (an adjacency heatmap plus execution weights).

### The ΔT protocol

Models respond with exactly one fenced block:

````
```lattice
{
  "summary": "guard empty input",
  "ops": [
    {"op": "upsert_node", "id": "avg_word_len", "body": "if count == 0 { 0 } else { chars / count }"},
    {"op": "upsert_probe", "name": "single", "inputs": {"text": "hi"}, "expect": {"report": {"word_count": 1, "avg_word_len": 2, "top_word": "hi"}}}
  ]
}
```
````

The full system prompt, including the Rhai cheat sheet, is in `crates/lattice-core/src/agent.rs`.

## Layout

```
crates/lattice-core/   engine: graph, embed, cgc, fre, lbm, runtime, agent, providers
src-tauri/             Tauri v2 shell: IPC commands, settings, event streaming
src/                   React + TypeScript UI
```
