/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
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
inline const char* MODEL_FORMATS = ".fbx,.glb,.gltf,.blend,.obj";
inline const char* TEXTURE_FORMATS = ".png,.jpg,.jpeg,.bmp";

/**
* Przechowuje statystyki rysowania.
*/
struct DrawStats
{
    int m_FrameTriangles         = 0; ///< liczba przetworzonych trójk¹tów
    int m_FrameTrianglesDrawn    = 0; ///< liczba narysowanych trójk¹tów
    int m_FramePixels            = 0; ///< liczba przetworzonych pikseli
    int m_FramePixelsDrawn       = 0; ///< liczba narysowanych pikseli

    int m_RasterTimeUS           = 0; ///< czas rasteryzacji
    int m_RasterTimePerThreadUS  = 0; ///< czas rasteryzacji na w¹tek
    int m_TransformTimeUS        = 0; ///< czas transformacji

    int m_DrawTimeUS             = 0; ///< czas rysowania
    int m_DrawTimePerThreadUS    = 0; ///< czas rysowania na w¹tek
    int m_FillrateKP             = 0; ///< iloœæ pikseli wype³nionych na sekundê
    int m_DT                     = 0; ///< czas rysowania jednego piksela
};