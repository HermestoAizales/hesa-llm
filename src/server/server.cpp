#include "server/server.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>

using json = nlohmann::json;
using namespace hesa;

namespace hesa {
namespace server {

Server::Server(const ServerConfig& cfg) : cfg_(cfg) {}

Server::~Server() {
    stop();
}

bool Server::start() {
    if (running_.load()) return false;

    // Load engine
    Engine::Config eng_cfg;
    eng_cfg.backend = BackendType::AUTO;
    eng_cfg.n_threads = cfg_.n_threads;
    eng_cfg.kv_cache_size = cfg_.kv_cache_ctx;

    auto result = Engine::create(cfg_.model_path, eng_cfg);
    if (!result) {
        fprintf(stderr, "[Server] ERROR: failed to load engine: %s\n",
                result.error().message().c_str());
        return false;
    }
    engine_ = std::move(result.value());

    running_.store(true);
    server_thread_ = std::thread(&Server::run_server_loop, this);
    fprintf(stderr, "[Server] Listening on %s:%u\n", cfg_.host.c_str(), cfg_.port);
    fprintf(stderr, "[Server] Model: %s (%s, %u layers, vocab=%u)\n",
            cfg_.model_path.c_str(),
            engine_->metadata().architecture.empty() ? "?" : engine_->metadata().architecture.c_str(),
            engine_->metadata().block_count,
            engine_->metadata().vocab_size);
    fprintf(stderr, "[Server] API endpoints:\n");
    fprintf(stderr, "  GET  /v1/models\n");
    fprintf(stderr, "  POST /v1/chat/completions\n");
    fprintf(stderr, "  POST /v1/completions\n");
    fprintf(stderr, "  POST /v1/tokenize\n");
    fprintf(stderr, "  POST /v1/detokenize\n");
    fprintf(stderr, "  GET  /health\n");
    fprintf(stderr, "  GET  /metrics\n");
    return true;
}

void Server::stop() {
    if (!running_.load()) return;
    running_.store(false);
    engine_->stop();
    if (server_thread_.joinable()) server_thread_.join();
}

void Server::wait() {
    if (server_thread_.joinable()) server_thread_.join();
}


static httplib::Server g_server;
static Engine* g_engine = nullptr;
static SessionManager g_sessions;
static std::atomic<int64_t> g_total_requests{0};
static std::atomic<int64_t> g_total_tokens{0};
static std::atomic<int64_t> g_total_errors{0};

// --- Helper: parse JSON safely ---
static json parse_json(const std::string& body, bool* ok) {
    try {
        *ok = true;
        return json::parse(body);
    } catch (...) {
        *ok = false;
        return json::object();
    }
}

// --- Helper: decode token IDs to string ---
static std::string decode_tokens(Engine& engine, const std::vector<int32_t>& tokens) {
    try {
        return engine.tokenizer()->decode(tokens);
    } catch (...) {
        std::string result;
        for (auto t : tokens) result += "<" + std::to_string(t) + ">";
        return result;
    }
}

void Server::run_server_loop() {
    auto& s = g_server;
    g_engine = engine_.get();

    // --- GET /v1/models ---
    s.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        const auto& meta = engine_->metadata();
        json models = json::object();
        models["object"] = "list";
        json data = json::array();
        json m = json::object();
        m["id"] = cfg_.model_name;
        m["object"] = "model";
        m["created"] = 1712000000;
        m["owned_by"] = "hesa-llm";
        m["architecture"] = meta.architecture;
        m["parameters"] = json::object();
        m["parameters"]["vocab_size"] = meta.vocab_size;
        m["parameters"]["num_hidden_layers"] = meta.block_count;
        m["parameters"]["hidden_size"] = meta.embedding_length;
        data.push_back(m);
        models["data"] = data;
        res.set_content(models.dump(), "application/json");
    });

    // --- POST /v1/chat/completions ---
    s.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        ++g_total_requests;
        bool json_ok = false;
        json body = parse_json(req.body, &json_ok);
        if (!json_ok) {
            json err = json::object();
            err["error"] = json::object();
            err["error"]["message"] = "Invalid JSON";
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        float temperature = body.value("temperature", 0.7f);
        float top_p = body.value("top_p", 0.95f);
        int32_t top_k = body.value("top_k", 40);
        float rep_penalty = body.value("repetition_penalty", 1.1f);
        int32_t max_tokens = body.value("max_tokens", 2048);
        bool stream = body.value("stream", false);

        // Build prompt from messages
        std::string full_prompt;
        if (body.contains("messages") && body["messages"].is_array()) {
            for (const auto& msg : body["messages"]) {
                std::string role = msg.value("role", "user");
                std::string content = msg.value("content", "");
                full_prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
            }
            full_prompt += "<|im_start|>assistant\n";
        } else if (body.contains("prompt")) {
            full_prompt = body["prompt"].get<std::string>();
        }

        // Tokenize
        auto prompt_tokens = engine_->tokenizer()->encode(full_prompt);
        std::string chat_id = "chatcmpl-hesa-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

        if (stream) {
            // Run full generation first, then stream results
            auto result = g_engine->generate(prompt_tokens, max_tokens > 0 ? (size_t)max_tokens : 2048,
                                              temperature, top_p);

            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_header("X-Accel-Buffering", "no");
            res.set_header("Transfer-Encoding", "chunked");
            res.status = 200;

            // Collect all SSE data and send at once (cpp-httplib v0.16 limitation)
            std::string sse_data;
            
            int32_t eos = g_engine->tokenizer()->eos_token_id();
            int gen_count = 0;
            
            if (result) {
                const auto& gen_tokens = result.value();
                for (size_t i = 0; i < gen_tokens.size(); ++i) {
                    int32_t tok = gen_tokens[i];
                    if (tok == eos) break;

                    std::string text = decode_tokens(*g_engine, {tok});
                    json chunk = json::object();
                    chunk["id"] = chat_id;
                    chunk["object"] = "chat.completion.chunk";
                    chunk["created"] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                    chunk["model"] = "hesa-llm";
                    chunk["choices"] = json::array({json::object()});
                    chunk["choices"][0]["index"] = 0;
                    chunk["choices"][0]["delta"]["content"] = text;
                    chunk["usage"] = json::object();
                    chunk["usage"]["prompt_tokens"] = (int)prompt_tokens.size();
                    chunk["usage"]["completion_tokens"] = (int)(i + 1);
                    chunk["usage"]["total_tokens"] = (int)(prompt_tokens.size() + i + 1);
                    sse_data += "data: " + chunk.dump() + "\n\n";
                    ++gen_count;
                }

                // Final chunk
                json finish = json::object();
                finish["id"] = chat_id;
                finish["object"] = "chat.completion.chunk";
                finish["created"] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                finish["model"] = "hesa-llm";
                finish["choices"] = json::array({json::object()});
                finish["choices"][0]["index"] = 0;
                finish["choices"][0]["delta"] = json::object();
                finish["choices"][0]["finish_reason"] = "stop";
                finish["usage"] = json::object();
                finish["usage"]["prompt_tokens"] = (int)prompt_tokens.size();
                finish["usage"]["completion_tokens"] = gen_count;
                finish["usage"]["total_tokens"] = (int)(prompt_tokens.size() + gen_count);
                sse_data += "data: " + finish.dump() + "\n\n";

                g_total_tokens += gen_count;
            } else {
                json err_chunk = json::object();
                err_chunk["id"] = chat_id;
                err_chunk["choices"] = json::array({json::object()});
                err_chunk["choices"][0]["finish_reason"] = "error";
                sse_data += "data: " + err_chunk.dump() + "\n\n";
                ++g_total_errors;
            }

            sse_data += "data: [DONE]\n\n";
            res.set_content(sse_data, "text/event-stream");
        } else {
            auto result = g_engine->generate(prompt_tokens, max_tokens > 0 ? (size_t)max_tokens : 2048,
                                              temperature, top_p);

            json response = json::object();
            response["id"] = chat_id;
            response["object"] = "chat.completion";
            response["created"] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            response["model"] = "hesa-llm";

            if (result) {
                const auto& gen_tokens = result.value();
                int eos = g_engine->tokenizer()->eos_token_id();
                std::vector<int32_t> content_tokens;
                for (auto t : gen_tokens) { if (t != eos) content_tokens.push_back(t); else break; }

                std::string text = decode_tokens(*g_engine, content_tokens);
                response["choices"] = json::array({json::object()});
                response["choices"][0]["index"] = 0;
                response["choices"][0]["message"] = json::object();
                response["choices"][0]["message"]["role"] = "assistant";
                response["choices"][0]["message"]["content"] = text;
                response["choices"][0]["finish_reason"] = "stop";

                response["usage"] = json::object();
                response["usage"]["prompt_tokens"] = (int)prompt_tokens.size();
                response["usage"]["completion_tokens"] = (int)content_tokens.size();
                response["usage"]["total_tokens"] = (int)(prompt_tokens.size() + content_tokens.size());

                g_total_tokens += content_tokens.size();
            } else {
                response["error"] = json::object();
                response["error"]["message"] = result.error().message();
                response["error"]["type"] = "inference_error";
                ++g_total_errors;
            }

            res.set_content(response.dump(), "application/json");
        }
    });

    // --- POST /v1/completions ---
    s.Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
        ++g_total_requests;
        bool json_ok = false;
        json body = parse_json(req.body, &json_ok);
        if (!json_ok) {
            json err = json::object();
            err["error"] = json::object();
            err["error"]["message"] = "Invalid JSON";
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        float temperature = body.value("temperature", 0.7f);
        float top_p = body.value("top_p", 0.95f);
        int32_t max_tokens = body.value("max_tokens", 2048);
        bool stream = body.value("stream", false);

        std::string prompt = body.value("prompt", "");
        auto prompt_tokens = engine_->tokenizer()->encode(prompt);
        std::string chat_id = "cmpl-hesa-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

        auto result = g_engine->generate(prompt_tokens, max_tokens > 0 ? (size_t)max_tokens : 2048,
                                          temperature, top_p);

        json response = json::object();
        response["id"] = chat_id;
        response["object"] = "text_completion";
        response["created"] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        response["model"] = "hesa-llm";

        if (result) {
            const auto& gen_tokens = result.value();
            int eos = g_engine->tokenizer()->eos_token_id();
            std::vector<int32_t> content_tokens;
            for (auto t : gen_tokens) { if (t != eos) content_tokens.push_back(t); else break; }

            std::string text = decode_tokens(*g_engine, content_tokens);
            response["choices"] = json::array({json::object()});
            response["choices"][0]["index"] = 0;
            response["choices"][0]["text"] = text;
            response["choices"][0]["finish_reason"] = "stop";
            response["usage"] = json::object();
            response["usage"]["prompt_tokens"] = (int)prompt_tokens.size();
            response["usage"]["completion_tokens"] = (int)content_tokens.size();
            response["usage"]["total_tokens"] = (int)(prompt_tokens.size() + content_tokens.size());
            g_total_tokens += content_tokens.size();
        } else {
            response["error"] = json::object();
            response["error"]["message"] = result.error().message();
            ++g_total_errors;
        }

        res.set_content(response.dump(), "application/json");
    });

    // --- POST /v1/tokenize ---
    s.Post("/v1/tokenize", [this](const httplib::Request& req, httplib::Response& res) {
        ++g_total_requests;
        bool json_ok = false;
        json body = parse_json(req.body, &json_ok);
        if (!json_ok) {
            json err = json::object();
            err["error"] = "Invalid JSON";
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::string text = body.value("text", "");
        auto tokens = engine_->tokenizer()->encode(text);

        json result = json::object();
        result["tokens"] = tokens;
        result["count"] = (int)tokens.size();
        res.set_content(result.dump(), "application/json");
    });

    // --- POST /v1/detokenize ---
    s.Post("/v1/detokenize", [this](const httplib::Request& req, httplib::Response& res) {
        ++g_total_requests;
        bool json_ok = false;
        json body = parse_json(req.body, &json_ok);
        if (!json_ok) {
            json err = json::object();
            err["error"] = "Invalid JSON";
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::vector<int32_t> tokens;
        if (body.contains("tokens")) {
            tokens = body["tokens"].get<std::vector<int32_t>>();
        }
        std::string text = decode_tokens(*engine_, tokens);

        json result = json::object();
        result["text"] = text;
        result["count"] = (int)tokens.size();
        res.set_content(result.dump(), "application/json");
    });

    // --- GET /health ---
    s.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        json health = json::object();
        health["status"] = running_.load() ? "ok" : "stopping";
        health["model"] = cfg_.model_path;
        health["model_name"] = cfg_.model_name;
        const auto& meta = engine_->metadata();
        health["architecture"] = meta.architecture;
        health["block_count"] = meta.block_count;
        health["vocab_size"] = meta.vocab_size;
        health["context_length"] = meta.context_length;
        res.set_content(health.dump(), "application/json");
    });

    // --- GET /metrics ---
    s.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        json metrics = json::object();
        metrics["total_requests"] = g_total_requests.load();
        metrics["total_tokens_generated"] = g_total_tokens.load();
        metrics["total_errors"] = g_total_errors.load();
        metrics["active_sessions"] = g_sessions.list_sessions().size();
        metrics["model_loaded"] = (g_engine != nullptr);
        res.set_content(metrics.dump(), "application/json");
    });

    // --- Session management ---
    s.Post("/sessions", [this](const httplib::Request&, httplib::Response& res) {
        std::string sid = g_sessions.create_session(*engine_);
        json r = json::object();
        r["session_id"] = sid;
        res.set_content(r.dump(), "application/json");
    });

    s.Post("/sessions/:id/reset", [this](const httplib::Request& req, httplib::Response& res) {
        std::string sid = req.path_params.at("id");
        bool ok = g_sessions.destroy_session(sid);
        json r = json::object();
        r["success"] = ok;
        res.set_content(r.dump(), "application/json");
    });

    s.Get("/sessions", [](const httplib::Request&, httplib::Response& res) {
        auto ids = g_sessions.list_sessions();
        json r = json::object();
        r["sessions"] = ids;
        r["count"] = (int)ids.size();
        res.set_content(r.dump(), "application/json");
    });

    // --- Start server ---
    fprintf(stderr, "[Server] Starting httplib server on %s:%u (thread pool: 4)\n",
            cfg_.host.c_str(), (unsigned)cfg_.port);
    s.set_post_routing_handler([](const auto& req, auto& res) {
        res.set_header("X-Server", "hesa-llm/0.1.0");
        if (req.method == "POST") {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
        }
    });
    // Handle CORS preflight
    s.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
        res.status = 200;
    });

    s.listen(cfg_.host.c_str(), cfg_.port);
    running_.store(false);
    fprintf(stderr, "[Server] Stopped\n");
}


// --- Session Manager Implementation ---
std::string SessionManager::create_session(Engine& /*engine*/) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::string id = "sess-" + std::to_string(now);
    auto session = std::make_unique<Session>();
    session->created_at = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    sessions_[id] = std::move(session);
    return id;
}

bool SessionManager::destroy_session(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx_);
    return sessions_.erase(id) > 0;
}

Session* SessionManager::get_session(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = sessions_.find(id);
    if (it != sessions_.end()) return it->second.get();
    return nullptr;
}

std::vector<std::string> SessionManager::list_sessions() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::string> result;
    for (const auto& pair : sessions_) result.push_back(pair.first);
    return result;
}

void SessionManager::cleanup_expired() {
    std::lock_guard<std::mutex> lock(mtx_);
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    for (auto it = sessions_.begin(); it != sessions_.end(); ) {
        if ((now - it->second->created_at) > EXPIRY_SECONDS) {
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace server
} // namespace hesa
