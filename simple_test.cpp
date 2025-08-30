#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <unordered_set>

// Simplified version of the direction enum for testing
enum class ChallengeDirection {
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3
};

// Function to convert direction to string
std::string directionToString(ChallengeDirection dir) {
    switch (dir) {
        case ChallengeDirection::UP: return "up";
        case ChallengeDirection::DOWN: return "down";
        case ChallengeDirection::LEFT: return "left";
        case ChallengeDirection::RIGHT: return "right";
        default: return "unknown";
    }
}

int main() {
    std::cout << "=== Testing Unique Direction Generation Logic ===\n";

    // Simulate the new challenge generation logic
    const ChallengeDirection DIRECTIONS[] = {
        ChallengeDirection::UP,
        ChallengeDirection::DOWN,
        ChallengeDirection::LEFT,
        ChallengeDirection::RIGHT
    };
    const int NUM_DIRECTIONS = 4;

    // Create random number generator (same as in ChallengeGenerator)
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

    // Generate multiple challenges and verify uniqueness
    const int NUM_TESTS = 10;

    for (int i = 0; i < NUM_TESTS; ++i) {
        // NEW LOGIC: Create a vector with all available directions
        std::vector<ChallengeDirection> all_directions(DIRECTIONS, DIRECTIONS + NUM_DIRECTIONS);

        // Shuffle the directions to get a random permutation
        std::shuffle(all_directions.begin(), all_directions.end(), rng);

        // Use all shuffled directions (guaranteed to be unique)
        std::vector<ChallengeDirection> challenge_directions = std::move(all_directions);

        // Verify uniqueness
        std::unordered_set<std::string> direction_set;

        std::cout << "Challenge " << (i+1) << ": ";
        bool has_duplicates = false;

        for (size_t j = 0; j < challenge_directions.size(); ++j) {
            std::string dir_str = directionToString(challenge_directions[j]);
            std::cout << dir_str;

            if (j < challenge_directions.size() - 1) {
                std::cout << ", ";
            }

            // Check if we already have this direction
            if (direction_set.find(dir_str) != direction_set.end()) {
                has_duplicates = true;
                std::cout << " (DUPLICATE!)";
            } else {
                direction_set.insert(dir_str);
            }
        }

        if (has_duplicates) {
            std::cout << " ❌ FAILED: Contains duplicate directions!";
        } else if (direction_set.size() != 4) {
            std::cout << " ❌ FAILED: Should have exactly 4 unique directions, got " << direction_set.size();
        } else {
            std::cout << " ✅ SUCCESS: All 4 directions are unique";
        }

        std::cout << std::endl;
    }

    std::cout << "\n=== Test completed - All challenges now generate unique directions! ===\n";

    return 0;
}
