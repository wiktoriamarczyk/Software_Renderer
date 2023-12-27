#pragma once

#include <SFML/Graphics.hpp>

#include "../imgui/imgui.h"
#include "../imgui/imgui-SFML.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <chrono>
#include <string>
#include <filesystem>
#include <functional>

using namespace std;

const int SCREEN_WIDTH = 1000;
const int SCREEN_HEIGHT = 800;
const int MAX_MODEL_VERTICES = 100'000; // 100 K
const int MAX_TEXTURE_SIZE = 4096; // 4K
const int FULL_ANGLE = 360;
const float PI = std::numbers::pi;
const int TRIANGLE_VERT_COUNT = 3;
const string INIT_TEXTURE_PATH = filesystem::current_path().string() + "/../Data/Checkerboard.png";
const string DEFAULT_TEXTURE_PATH = filesystem::current_path().string() + "/../Data/Default.png";

struct MyModelPaths
{
    string modelPath;
    string texturePath;
};