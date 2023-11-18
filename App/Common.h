#pragma once

#include <SFML/Graphics.hpp>

#include "../imgui/imgui.h"
#include "../imgui/imgui-SFML.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <chrono>

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const int fullCircle = 360;
const float pi = std::numbers::pi;
const int triangleVerticesCount = 3;

using namespace std;