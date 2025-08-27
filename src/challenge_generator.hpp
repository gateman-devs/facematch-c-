#ifndef CHALLENGE_GENERATOR_HPP
#define CHALLENGE_GENERATOR_HPP

#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

enum class ChallengeDirection {
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3
};

struct Challenge {
    std::string id;
    std::vector<ChallengeDirection> directions;
    std::chrono::time_point<std::chrono::system_clock> created_at;
    int ttl_seconds;
    
    Challenge() : ttl_seconds(300) { // 5 minutes default TTL
        created_at = std::chrono::system_clock::now();
    }
    
    // Check if challenge has expired
    bool isExpired() const {
        auto now = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - created_at).count();
        return elapsed > ttl_seconds;
    }
    
    // Convert to JSON for storage/transmission
    json toJson() const {
        json j;
        j["id"] = id;
        j["directions"] = json::array();
        for (const auto& dir : directions) {
            j["directions"].push_back(directionToString(dir));
        }
        j["created_at"] = std::chrono::duration_cast<std::chrono::seconds>(
            created_at.time_since_epoch()).count();
        j["ttl_seconds"] = ttl_seconds;
        return j;
    }
    
    // Create from JSON
    static Challenge fromJson(const json& j) {
        Challenge challenge;
        challenge.id = j["id"];
        challenge.ttl_seconds = j["ttl_seconds"];
        
        // Parse created_at
        auto timestamp = j["created_at"].get<int64_t>();
        challenge.created_at = std::chrono::system_clock::from_time_t(timestamp);
        
        // Parse directions
        for (const auto& dir_str : j["directions"]) {
            challenge.directions.push_back(stringToDirection(dir_str));
        }
        
        return challenge;
    }
    
    static std::string directionToString(ChallengeDirection dir) {
        switch (dir) {
            case ChallengeDirection::UP: return "up";
            case ChallengeDirection::DOWN: return "down";
            case ChallengeDirection::LEFT: return "left";
            case ChallengeDirection::RIGHT: return "right";
            default: return "unknown";
        }
    }
    
    static ChallengeDirection stringToDirection(const std::string& str) {
        if (str == "up") return ChallengeDirection::UP;
        if (str == "down") return ChallengeDirection::DOWN;
        if (str == "left") return ChallengeDirection::LEFT;
        if (str == "right") return ChallengeDirection::RIGHT;
        return ChallengeDirection::UP; // Default fallback
    }
};

class ChallengeGenerator {
public:
    ChallengeGenerator();
    ~ChallengeGenerator() = default;
    
    // Generate a new challenge with 4 random directions
    Challenge generateChallenge(int ttl_seconds = 300);
    
    // Generate a unique challenge ID
    std::string generateChallengeId();
    
private:
    std::mt19937 rng;
    std::uniform_int_distribution<int> direction_dist;
    
    // Available directions
    static constexpr int NUM_DIRECTIONS = 4;
    static constexpr ChallengeDirection DIRECTIONS[] = {
        ChallengeDirection::UP,
        ChallengeDirection::DOWN,
        ChallengeDirection::LEFT,
        ChallengeDirection::RIGHT
    };
};

#endif // CHALLENGE_GENERATOR_HPP
