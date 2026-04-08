#ifndef HESA_SERVER_HPP
#define HESA_SERVER_HPP

#include "hesa/engine.hpp"
#include "hesa/model.hpp"
#include "hesa/tokenizer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <unordered_map>

namespace hesa {
namespace server {

struct ServerConfig {
    std::string host         = "127.0.0.1";
    uint16_t    port         = 8080;
    std::string model_path;
    int32_t     n_threads    = 0;
    size_t      kv_cache_ctx = 4096;
    const char* model_name   = "hesa-llm";
};

class Server {
public:
    Server(const ServerConfig& cfg);
    ~Server();

    bool start();
    void stop();
    void wait();
    bool running() const { return running_.load(); }

private:
    ServerConfig cfg_;
    std::unique_ptr<Engine> engine_;
    std::atomic<bool> running_{false};
    std::thread server_thread_;
    void run_server_loop();
};

// Session management for concurrent requests
struct Session {
    std::vector<int32_t> token_history;
    size_t   ctx_len   = 4096;
    bool     persistent = false;
    int64_t  created_at = 0;
};

class SessionManager {
public:
    std::string create_session(Engine& engine);
    bool destroy_session(const std::string& id);
    Session* get_session(const std::string& id);
    std::vector<std::string> list_sessions() const;
    void cleanup_expired();

private:
    mutable std::mutex mtx_;
    std::unordered_map<std::string, std::unique_ptr<Session>> sessions_;
    static constexpr int EXPIRY_SECONDS = 3600;
};

} // namespace server
} // namespace hesa
#endif // HESA_SERVER_HPP
