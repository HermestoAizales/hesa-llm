#include "hesa/backend.hpp"
#include "hesa/engine.hpp"
#include "hesa/model.hpp"
#include "hesa/result.hpp"
#include "hesa/sampling.hpp"
#include "hesa/tensor.hpp"
#include "hesa/version.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>

static void print_help(const char* prog) {
    std::printf("hesa-llm v%d.%d.%d\n",
                hesa::VERSION_MAJOR, hesa::VERSION_MINOR, hesa::VERSION_PATCH);
    std::printf("Hierarchical External-State Augmented LLM Inference Engine\n\n");
    std::printf("Usage: %s [options]\n\n", prog);
    std::printf("Options:\n");
    std::printf("  -m, --model <file>       GGUF model file path (required)\n");
    std::printf("  -p, --prompt <text>      Input prompt (default: interactive)\n");
    std::printf("  -n, --max-tokens <N>     Max tokens to generate (default: 128)\n");
    std::printf("  -s, --seed <N>           Random seed (default: -1 = random)\n");
    std::printf("  -t, --temp <F>           Temperature (default: 0.8)\n");
    std::printf("      --top-p <F>          Top-p (default: 0.95)\n");
    std::printf("      --top-k <N>          Top-k (default: 40)\n");
    std::printf("  -T, --threads <N>        CPU threads (default: auto)\n");
    std::printf("      --backend <type>     Backend: cpu, cuda, metal, auto (default: auto)\n");
    std::printf("  -h, --help               Show this help message\n");
    std::printf("  -v, --version            Show version\n");
}

int main(int argc, char* argv[]) {
    // ── Argument parsing ──
    std::string model_path;
    std::string prompt;
    int   max_tokens  = 128;
    int   seed        = -1;
    float temperature = 0.8f;
    float top_p       = 0.95f;
    int   top_k       = 40;
    int   n_threads   = 0;
    hesa::BackendType backend_type = hesa::BackendType::AUTO;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_arg = [&]() -> std::string {
            if (i + 1 < argc) return argv[++i];
            std::fprintf(stderr, "Error: missing argument after %s\n", arg.c_str());
            return {};
        };

        if (arg == "-h" || arg == "--help") { print_help(argv[0]); return 0; }
        if (arg == "-v" || arg == "--version") {
            std::printf("hesa-llm v%d.%d.%d\n",
                        hesa::VERSION_MAJOR, hesa::VERSION_MINOR, hesa::VERSION_PATCH);
            return 0;
        }
        if (arg == "-m" || arg == "--model")        model_path    = next_arg();
        else if (arg == "-p" || arg == "--prompt")      prompt      = next_arg();
        else if (arg == "-n" || arg == "--max-tokens")  max_tokens  = std::stoi(next_arg());
        else if (arg == "-s" || arg == "--seed")        seed        = std::stoi(next_arg());
        else if (arg == "-t" || arg == "--temp")        temperature = std::stof(next_arg());
        else if (arg == "--top-p")                      top_p       = std::stof(next_arg());
        else if (arg == "--top-k")                      top_k       = std::stoi(next_arg());
        else if (arg == "-T" || arg == "--threads")     n_threads   = std::stoi(next_arg());
        else if (arg == "--backend") {
            std::string bt = next_arg();
            if      (bt == "cpu")   backend_type = hesa::BackendType::CPU_X86;
            else if (bt == "cuda")  backend_type = hesa::BackendType::CUDA;
            else if (bt == "metal") backend_type = hesa::BackendType::METAL;
            else if (bt == "arm")   backend_type = hesa::BackendType::CPU_ARM;
            else                    backend_type = hesa::BackendType::AUTO;
        }
    }

    // ── Banner ──
    std::printf("\n");
    std::printf("  __  __           _     __  __ _         \n");
    std::printf(" |  \\/  | ___   __| | __|  \\/  (_)___ ___ \n");
    std::printf(" | |\\/| |/ _ \\ / _` |/ _` | |\\/| / __/ __|\n");
    std::printf(" | |  | | (_) | (_| | (_| | |  | \\__ \\__ \\\n");
    std::printf(" |_|  |_|\\___/ \\__,_|\\__,_|_|  |_|___/___/\n");
    std::printf("  v%d.%d.%d  (Phase 1 — basic pipeline)\n\n",
                hesa::VERSION_MAJOR, hesa::VERSION_MINOR, hesa::VERSION_PATCH);

    if (model_path.empty()) {
        std::printf("No model specified. Use -m <path> to load a GGUF model.\n");
        std::printf("Run: %s --help\n", argv[0]);
        return 0;
    }

    // ── Create engine (backend → model → tokenizer) ──
    hesa::Engine::Config engine_cfg;
    engine_cfg.backend       = backend_type;
    engine_cfg.n_threads     = n_threads;
    engine_cfg.sampling.temperature     = temperature;
    engine_cfg.sampling.top_p           = top_p;
    engine_cfg.sampling.top_k           = top_k;
    engine_cfg.sampling.max_tokens      = max_tokens;
    engine_cfg.sampling.seed            = seed;

    auto engine_result = hesa::Engine::create(model_path, engine_cfg);
    if (!engine_result) {
        std::fprintf(stderr, "\nError: Failed to create engine.\n");
        return 1;
    }
    auto& engine = *engine_result;

    const auto& meta = engine->metadata();
    std::printf("\nModel: %s\n", model_path.c_str());
    std::printf("  Architecture: %s\n", meta.architecture.empty() ? "unknown" : meta.architecture.c_str());
    std::printf("  Vocab size:   %u\n", meta.vocab_size);
    std::printf("  Layers:       %u\n", meta.block_count);
    std::printf("  Embed dim:    %u\n", meta.embedding_length);
    std::printf("  FFN dim:      %u\n", meta.feed_forward_length);
    std::printf("  Context:      %u\n", meta.context_length);
    std::printf("  Tensors:      %zu\n", engine->model().tensor_names().size());

    // ── Tokenize prompt ──
    std::vector<int32_t> prompt_tokens;

    if (!prompt.empty()) {
        // Try encoding with tokenizer
        if (engine->metadata().vocab_size > 0) {
            std::fprintf(stderr, "\nTokenizing prompt (%zu chars)...\n", prompt.size());
            // For Phase 1, we use character-level tokenization as fallback
            // Each character maps to its byte value if within vocab
            // This is a crude fallback when no proper tokenizer is available
            prompt_tokens.reserve(prompt.size());
            for (char c : prompt) {
                int32_t tid = static_cast<int32_t>(static_cast<unsigned char>(c));
                if (static_cast<size_t>(tid) < engine->metadata().vocab_size) {
                    prompt_tokens.push_back(tid);
                }
            }
            if (prompt_tokens.empty()) {
                // Fallback: use token 0 repeated
                std::fprintf(stderr, "No valid tokens from prompt, using empty\n");
                prompt_tokens.push_back(0);
            }
        }
        std::printf("\nPrompt: \"%s\"\n", prompt.c_str());
        std::printf("Prompt tokens: %zu\n", prompt_tokens.size());
    } else {
        std::printf("\nNo prompt provided. Generation will start from EOS.\n");
        prompt_tokens.push_back(0);
    }

    // ── Generate ──
    std::printf("\nGenerating %d tokens (temp=%.2f, top_p=%.2f)...\n\n",
                max_tokens, temperature, top_p);

    auto gen_result = engine->generate(
        std::span<const int32_t>(prompt_tokens.data(), prompt_tokens.size()),
        static_cast<size_t>(max_tokens),
        temperature,
        top_p
    );

    if (!gen_result) {
        std::fprintf(stderr, "\nError: Generation failed.\n");
        return 1;
    }

    // ── Output ──
    std::printf("\nGenerated %zu tokens:\n", gen_result->size());
    for (size_t i = 0; i < gen_result->size(); ++i) {
        int32_t tid = (*gen_result)[i];
        // Try to decode: if byte-level, convert back to char
        if (tid >= 0 && tid < 128) {
            std::printf("%c", static_cast<char>(tid));
        } else {
            std::printf("<%d>", tid);
        }
    }
    std::printf("\n\n");

    return 0;
}
