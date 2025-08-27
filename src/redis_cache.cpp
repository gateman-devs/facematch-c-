#include "redis_cache.hpp"
#include <iostream>
#include <sstream>

const std::string RedisCache::CHALLENGE_PREFIX = "gateman:challenge:";

RedisCache::RedisCache() : context(nullptr), port_(6379), connected_(false) {
}

RedisCache::~RedisCache() {
    cleanup();
}

bool RedisCache::initialize(const std::string& host, int port, const std::string& password) {
    host_ = host;
    port_ = port;
    
    // Cleanup any existing connection
    cleanup();
    
    // Create Redis connection
    context = redisConnect(host_.c_str(), port_);
    
    if (context == nullptr || context->err) {
        if (context) {
            std::cerr << "Redis connection error: " << context->errstr << std::endl;
            redisFree(context);
            context = nullptr;
        } else {
            std::cerr << "Redis connection error: Can't allocate redis context" << std::endl;
        }
        connected_ = false;
        return false;
    }
    
    // Authenticate if password is provided
    if (!password.empty()) {
        redisReply* reply = (redisReply*)redisCommand(context, "AUTH %s", password.c_str());
        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::cerr << "Redis authentication failed";
            if (reply && reply->str) {
                std::cerr << ": " << reply->str;
            }
            std::cerr << std::endl;
            
            if (reply) freeReplyObject(reply);
            cleanup();
            return false;
        }
        freeReplyObject(reply);
    }
    
    // Test connection with PING
    redisReply* ping_reply = (redisReply*)redisCommand(context, "PING");
    if (ping_reply == nullptr || ping_reply->type != REDIS_REPLY_STATUS || 
        std::string(ping_reply->str) != "PONG") {
        std::cerr << "Redis PING test failed" << std::endl;
        if (ping_reply) freeReplyObject(ping_reply);
        cleanup();
        return false;
    }
    freeReplyObject(ping_reply);
    
    connected_ = true;
    std::cout << "Redis connection established successfully to " << host_ << ":" << port_ << std::endl;
    return true;
}

bool RedisCache::storeChallenge(const Challenge& challenge) {
    if (!connected_ || !context) {
        std::cerr << "Redis not connected" << std::endl;
        return false;
    }
    
    try {
        // Convert challenge to JSON string
        json challenge_json = challenge.toJson();
        std::string challenge_data = challenge_json.dump();
        
        // Store in Redis with TTL
        std::string key = getChallengeKey(challenge.id);
        redisReply* reply = (redisReply*)redisCommand(context, 
            "SETEX %s %d %s", 
            key.c_str(), 
            challenge.ttl_seconds, 
            challenge_data.c_str());
        
        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::cerr << "Failed to store challenge in Redis";
            if (reply && reply->str) {
                std::cerr << ": " << reply->str;
            }
            std::cerr << std::endl;
            
            if (reply) freeReplyObject(reply);
            return false;
        }
        
        bool success = (reply->type == REDIS_REPLY_STATUS && 
                       std::string(reply->str) == "OK");
        
        freeReplyObject(reply);
        
        if (success) {
            std::cout << "Challenge " << challenge.id << " stored in Redis with TTL " 
                      << challenge.ttl_seconds << " seconds" << std::endl;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "Error storing challenge in Redis: " << e.what() << std::endl;
        return false;
    }
}

std::optional<Challenge> RedisCache::getChallenge(const std::string& challenge_id) {
    if (!connected_ || !context) {
        std::cerr << "Redis not connected" << std::endl;
        return std::nullopt;
    }
    
    try {
        std::string key = getChallengeKey(challenge_id);
        redisReply* reply = (redisReply*)redisCommand(context, "GET %s", key.c_str());
        
        if (reply == nullptr) {
            std::cerr << "Redis GET command failed" << std::endl;
            return std::nullopt;
        }
        
        if (reply->type == REDIS_REPLY_NIL) {
            // Challenge not found or expired
            freeReplyObject(reply);
            return std::nullopt;
        }
        
        if (reply->type != REDIS_REPLY_STRING) {
            std::cerr << "Unexpected Redis reply type for challenge: " << reply->type << std::endl;
            freeReplyObject(reply);
            return std::nullopt;
        }
        
        // Parse JSON data
        std::string challenge_data(reply->str, reply->len);
        freeReplyObject(reply);
        
        json challenge_json = json::parse(challenge_data);
        Challenge challenge = Challenge::fromJson(challenge_json);
        
        // Check if challenge has expired (additional safety check)
        if (challenge.isExpired()) {
            std::cout << "Challenge " << challenge_id << " has expired, removing from cache" << std::endl;
            deleteChallenge(challenge_id);
            return std::nullopt;
        }
        
        std::cout << "Retrieved challenge " << challenge_id << " from Redis" << std::endl;
        return challenge;
        
    } catch (const std::exception& e) {
        std::cerr << "Error retrieving challenge from Redis: " << e.what() << std::endl;
        return std::nullopt;
    }
}

bool RedisCache::deleteChallenge(const std::string& challenge_id) {
    if (!connected_ || !context) {
        std::cerr << "Redis not connected" << std::endl;
        return false;
    }
    
    try {
        std::string key = getChallengeKey(challenge_id);
        redisReply* reply = (redisReply*)redisCommand(context, "DEL %s", key.c_str());
        
        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::cerr << "Failed to delete challenge from Redis";
            if (reply && reply->str) {
                std::cerr << ": " << reply->str;
            }
            std::cerr << std::endl;
            
            if (reply) freeReplyObject(reply);
            return false;
        }
        
        bool success = (reply->type == REDIS_REPLY_INTEGER && reply->integer > 0);
        freeReplyObject(reply);
        
        if (success) {
            std::cout << "Challenge " << challenge_id << " deleted from Redis" << std::endl;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "Error deleting challenge from Redis: " << e.what() << std::endl;
        return false;
    }
}

bool RedisCache::isConnected() const {
    return connected_ && context != nullptr && context->err == 0;
}

std::string RedisCache::getConnectionStatus() const {
    if (!connected_ || !context) {
        return "Disconnected";
    }
    
    if (context->err != 0) {
        return std::string("Error: ") + context->errstr;
    }
    
    return "Connected to " + host_ + ":" + std::to_string(port_);
}

bool RedisCache::reconnect() {
    cleanup();
    return initialize(host_, port_);
}

void RedisCache::cleanup() {
    if (context) {
        redisFree(context);
        context = nullptr;
    }
    connected_ = false;
}

bool RedisCache::executeCommand(const std::string& command) {
    auto result = executeCommandWithResult(command);
    return result.has_value();
}

std::optional<std::string> RedisCache::executeCommandWithResult(const std::string& command) {
    if (!connected_ || !context) {
        return std::nullopt;
    }
    
    redisReply* reply = (redisReply*)redisCommand(context, command.c_str());
    if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
        if (reply) freeReplyObject(reply);
        return std::nullopt;
    }
    
    std::string result;
    if (reply->type == REDIS_REPLY_STRING || reply->type == REDIS_REPLY_STATUS) {
        result = std::string(reply->str, reply->len);
    }
    
    freeReplyObject(reply);
    return result;
}

std::string RedisCache::getChallengeKey(const std::string& challenge_id) const {
    return CHALLENGE_PREFIX + challenge_id;
}
