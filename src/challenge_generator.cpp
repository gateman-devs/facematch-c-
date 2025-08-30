#include "challenge_generator.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

// Define the static constexpr array
constexpr ChallengeDirection ChallengeGenerator::DIRECTIONS[];

ChallengeGenerator::ChallengeGenerator() 
    : rng(std::chrono::steady_clock::now().time_since_epoch().count()),
      direction_dist(0, NUM_DIRECTIONS - 1) {
}

Challenge ChallengeGenerator::generateChallenge(int ttl_seconds) {
    Challenge challenge;
    challenge.id = generateChallengeId();
    challenge.ttl_seconds = ttl_seconds;
    challenge.created_at = std::chrono::system_clock::now();

    // Create a vector with all available directions
    std::vector<ChallengeDirection> all_directions(DIRECTIONS, DIRECTIONS + NUM_DIRECTIONS);

    // Shuffle the directions to get a random permutation
    std::shuffle(all_directions.begin(), all_directions.end(), rng);

    // Use all shuffled directions (guaranteed to be unique)
    challenge.directions = std::move(all_directions);

    std::cout << "Generated challenge " << challenge.id << " with directions: ";
    for (size_t i = 0; i < challenge.directions.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << Challenge::directionToString(challenge.directions[i]);
    }
    std::cout << std::endl;

    return challenge;
}

std::string ChallengeGenerator::generateChallengeId() {
    // Generate a unique ID using timestamp and random number
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    // Add some randomness to ensure uniqueness
    std::uniform_int_distribution<int> random_dist(1000, 9999);
    int random_part = random_dist(rng);
    
    std::stringstream ss;
    ss << "challenge_" << timestamp << "_" << random_part;
    return ss.str();
}
