/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2023
*/

#pragma once
#include <SFML/Graphics.hpp>
#include "../imgui/imgui.h"
#include "../imgui/imgui-SFML.h"
#include <tracy/Tracy.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <chrono>
#include <string>
#include <filesystem>
#include <functional>
#include <future>
#include <optional>
#include <atomic>
#include <semaphore>

using std::vector;
using std::string;
using std::shared_ptr;
using std::make_shared;
using std::unique_ptr;
using std::make_unique;
using std::function;
using std::filesystem::current_path;
using namespace std::filesystem;
using std::filesystem::exists;
using std::promise;
using std::future;
using std::optional;
using std::atomic_bool;
using std::atomic_int;
using std::counting_semaphore;
using std::max;
using std::thread;

const int SCREEN_WIDTH = 1024;
const int SCREEN_HEIGHT = 768;
const int MAX_MODEL_TRIANGLES = 200'000;
const int MAX_TEXTURE_SIZE = 4096;
const int FULL_ANGLE = 360;
const float PI = std::numbers::pi;
const int TRIANGLE_VERT_COUNT = 3;
const string INIT_TEXTURE_PATH = "../Data/Checkerboard.png";
const string DEFAULT_TEXTURE_PATH = "../Data/Default.png";
inline const char* MODEL_FORMATS = ".fbx,.glb,.gltf,.blend,.obj";
inline const char* TEXTURE_FORMATS = ".png,.jpg,.jpeg,.bmp";

struct DrawStats
{
    int m_FrameTriangles         = 0;
    int m_FrameTrianglesDrawn    = 0;
    int m_FramePixels            = 0;
    int m_FramePixelsDrawn       = 0;
    int m_DrawTimeUS            = 0;
    int m_DrawTimePerThreadUS   = 0;
    int m_FillrateKP            = 0;
    int m_DT                    = 0;
};