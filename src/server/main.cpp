#include "server/server.hpp"
#include <iostream>
#include <csignal>

static std::unique_ptr<hesa::server::Server> g_server;
static bool g_running = true;

static void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cerr << "\n[main] Shutting down...\n";
        g_running = false;
        if (g_server) {
            g_server->stop();
        }
    }
}

int main(int argc, char** argv) {
    // Default configuration
    hesa::server::ServerConfig cfg;
    cfg.host = "0.0.0.0";
    cfg.port = 8000;
    cfg.model_path = "models/smollm2-135m-instruct-q8_0.gguf";
    cfg.model_name = "hesa-llm";
    cfg.n_threads = 0;
    cfg.kv_cache_ctx = 4096;

    // Simple arg parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cerr << "hesa-server [options]\n"
                      << "  -h, --help           Show this help\n"
                      << "  -m, --model PATH     Model file path (default: " << cfg.model_path << ")\n"
                      << "  -p, --port PORT      Port to listen on (default: " << cfg.port << ")\n"
                      << "  -H, --host HOST      Host to bind (default: " << cfg.host << ")\n"
                      << "  -t, --threads N      Number of threads (default: auto)\n"
                      << "  -n, --name NAME      Model name (default: " << cfg.model_name << ")\n";
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) cfg.model_path = argv[++i];
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) cfg.port = std::stoi(argv[++i]);
        } else if (arg == "-H" || arg == "--host") {
            if (i + 1 < argc) cfg.host = argv[++i];
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) cfg.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-n" || arg == "--name") {
            if (i + 1 < argc) cfg.model_name = argv[++i];
        }
    }

    // Setup signal handling
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Create and start server
    g_server = std::make_unique<hesa::server::Server>(cfg);
    if (!g_server->start()) {
        std::cerr << "[main] Failed to start server\n";
        return 1;
    }

    // Wait for server thread (blocks until stopped)
    g_server->wait();
    return 0;
}
