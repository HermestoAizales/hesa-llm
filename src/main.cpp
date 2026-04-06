#include <hesa/hesa.h>
#include <hesa/version.hpp>
#include <hesa/backend.hpp>
#include <hesa/tensor.hpp>
#include <hesa/model.hpp>
#include <hesa/tokenizer.hpp>
#include <hesa/sampling.hpp>

#include <iostream>
#include <string>
#include <cstring>

static void print_help(const char* prog) {
    std::cout << "hesa-llm v" << hesa::VERSION_MAJOR << "."
              << hesa::VERSION_MINOR << "." << hesa::VERSION_PATCH << "\n"
              << "Hierarchical External-State Augmented LLM Inference Engine\n\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -m, --model <file>       GGUF model file path\n"
              << "  -p, --prompt <text>      Input prompt (default: interactive)\n"
              << "  -n, --n-predict <N>      Max tokens to generate (default: 128)\n"
              << "  -s, --seed <N>           Random seed (default: -1 = random)\n"
              << "  -t, --temp <F>           Temperature (default: 0.8)\n"
              << "  --top-p <F>              Top-p (default: 0.95)\n"
              << "  --top-k <N>              Top-k (default: 40)\n"
              << "  -T, --threads <N>        CPU threads (default: auto)\n"
              << "  --backend <type>         Backend: cpu, cuda, metal, auto (default: auto)\n"
              << "  -h, --help               Show this help message\n"
              << "  -v, --version            Show version\n";
}

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string model_path;
    std::string prompt;
    int n_predict = 128;
    int seed = -1;
    float temperature = 0.8f;
    int n_threads = 0;
    hesa::BackendType backend_type = hesa::BackendType::AUTO;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto get_next = [&]() -> std::string {
            if (i + 1 < argc) return argv[++i];
            std::cerr << "Missing argument after " << arg << "\n";
            return {};
        };

        if ((arg == "-h" || arg == "--help")) { print_help(argv[0]); return 0; }
        if (arg == "-v" || arg == "--version") {
            std::cout << "hesa-llm v" << hesa::VERSION_MAJOR << "."
                      << hesa::VERSION_MINOR << "." << hesa::VERSION_PATCH << "\n";
            return 0;
        }
        if (arg == "-m" || arg == "--model")    model_path = get_next();
        else if (arg == "-p" || arg == "--prompt") prompt = get_next();
        else if (arg == "-n" || arg == "--n-predict") n_predict = std::stoi(get_next());
        else if (arg == "-s" || arg == "--seed")      seed = std::stoi(get_next());
        else if (arg == "-t" || arg == "--temp")       temperature = std::stof(get_next());
        else if (arg == "--top-p")                     {} // top_p = stof(get_next());
        else if (arg == "--top-k")                     {} // top_k = stoi(get_next());
        else if (arg == "-T" || arg == "--threads")    n_threads = std::stoi(get_next());
        else if (arg == "--backend") {
            std::string bt = get_next();
            if (bt == "cpu")  backend_type = hesa::BackendType::CPU_X86;
            else if (bt == "cuda") backend_type = hesa::BackendType::CUDA;
            else if (bt == "metal") backend_type = hesa::BackendType::METAL;
            else if (bt == "arm") backend_type = hesa::BackendType::CPU_ARM;
        }
    }

    // Banner
    std::cout << "\n"
              << "  __  __           _     __  __ _         \n"
              << " |  \\/  | ___   __| | __|  \\/  (_)___ ___ \n"
              << " | |\\/| |/ _ \\ / _` |/ _` | |\\/| / __/ __|\n"
              << " | |  | | (_) | (_| | (_| | |  | \\__ \\__ \\\n"
              << " |_|  |_|\\___/ \\__,_|\\__,_|_|  |_|___/___/\n"
              << "  v" << hesa::VERSION_MAJOR << "." << hesa::VERSION_MINOR << "."
              << hesa::VERSION_PATCH << "\n\n";

    // Create backend
    hesa::DeviceConfig cfg;
    cfg.n_threads = n_threads;

    auto backend_result = hesa::create_backend(backend_type, cfg);
    if (!backend_result) {
        std::cerr << "Error: Failed to create backend\n";
        return 1;
    }
    auto& backend = *backend_result;
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "Threads: " << (n_threads > 0 ? n_threads : backend->name()[0] == 'C' ? 1 : 1) << "\n";

    if (model_path.empty()) {
        std::cout << "No model specified. Use -m <path> to load a GGUF model.\n";
        std::cout << "\nRun: " << argv[0] << " --help\n";
        return 0;
    }

    // Load model
    std::cout << "\nLoading model: " << model_path << " ...\n";
    auto model_result = hesa::Model::load(model_path, backend.get());
    if (!model_result) {
        std::cerr << "Error: Failed to load model: " << model_path << "\n";
        return 1;
    }
    auto& model = *model_result;

    std::cout << "  Architecture: " << model->metadata().architecture << "\n";
    std::cout << "  Vocab size:   " << model->metadata().vocab_size << "\n";
    std::cout << "  Layers:       " << model->metadata().block_count << "\n";
    std::cout << "  Embed size:   " << model->metadata().embedding_length << "\n";
    std::cout << "  Context:      " << model->metadata().context_length << "\n";
    std::cout << "  Tensors:      " << model->tensor_names().size() << "\n";

    // Initialize tokenizer
    auto tok_result = hesa::create_tokenizer_from_model(*model);
    if (!tok_result) {
        std::cerr << "Error: Failed to initialize tokenizer\n";
        return 1;
    }
    auto& tokenizer = *tok_result;

    // Get prompt
    if (prompt.empty()) {
        std::cout << "\nInteractive mode (Ctrl+D to exit):\n> ";
        std::string line;
        while (std::getline(std::cin, line)) {
            prompt += line + "\n";
        }
    }
    std::cout << "\nPrompt tokens: " << tokenizer->encode(prompt).size() << "\n";

    // Generation config
    hesa::GenerationConfig gen_cfg;
    gen_cfg.temperature = temperature;
    gen_cfg.max_tokens = n_predict;
    gen_cfg.seed = seed;

    std::cout << "\n[TODO: Full generation loop coming in Phase 1]\n";
    std::cout << "Model loaded successfully with "
              << model->tensor_names().size() << " tensors.\n";

    return 0;
}
