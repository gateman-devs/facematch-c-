#ifndef REDIS_CACHE_HPP
#define REDIS_CACHE_HPP

#include <string>
#include <memory>
#include <optional>
#include <hiredis/hiredis.h>
#include <nlohmann/json.hpp>
#include "challenge_generator.hpp"

using json = nlohmann::json;

class RedisCache {
public:
    RedisCache();
    ~RedisCache();
    
    // Initialize connection to Redis
    bool initialize(const std::string& host = "127.0.0.1", int port = 6379, const std::string& password = "");
    
    // Store a challenge in Redis with TTL
    bool storeChallenge(const Challenge& challenge);
    
    // Retrieve a challenge from Redis
    std::optional<Challenge> getChallenge(const std::string& challenge_id);
    
    // Delete a challenge from Redis
    bool deleteChallenge(const std::string& challenge_id);
    
    // Check if Redis connection is healthy
    bool isConnected() const;
    
    // Get connection status
    std::string getConnectionStatus() const;
    
private:
    redisContext* context;
    std::string host_;
    int port_;
    bool connected_;
    
    // Helper methods
    bool reconnect();
    void cleanup();
    bool executeCommand(const std::string& command);
    std::optional<std::string> executeCommandWithResult(const std::string& command);
    
    // Redis key prefix for challenges
    static const std::string CHALLENGE_PREFIX;
    std::string getChallengeKey(const std::string& challenge_id) const;
};

#endif // REDIS_CACHE_HPP
