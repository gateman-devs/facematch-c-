#include "src/challenge_generator.hpp"
#include <iostream>
#include <unordered_set>
#include <string>

int main() {
    ChallengeGenerator generator;

    std::cout << "=== Testing Unique Direction Generation ===\n";

    // Generate multiple challenges and verify uniqueness
    const int NUM_TESTS = 10;

    for (int i = 0; i < NUM_TESTS; ++i) {
        Challenge challenge = generator.generateChallenge();

        // Check for duplicates using a set
        std::unordered_set<std::string> direction_set;

        std::cout << "Challenge " << (i+1) << ": ";
        bool has_duplicates = false;

        for (const auto& direction : challenge.directions) {
            std::string dir_str = Challenge::directionToString(direction);
            std::cout << dir_str;

            if (i < challenge.directions.size() - 1) {
                std::cout << ", ";
            }

            // Check if we already have this direction
            if (direction_set.find(dir_str) != direction_set.end()) {
                has_duplicates = true;
                std::cout << " (DUPLICATE FOUND!)";
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

    std::cout << "\n=== Test completed ===\n";

    return 0;
}
