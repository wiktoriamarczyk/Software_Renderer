#*****************************************************
# specify minimum version of CMake to use
# (find_package requires 3.23)
#*****************************************************
cmake_minimum_required(VERSION 3.26)

#*****************************************************
project(SoftwareRenderer LANGUAGES CXX)
#*****************************************************

#*****************************************************
# add source files
#*****************************************************

SET(APP_SRC_FILES_LIST
    App/Application.h
    App/Application.cpp
    App/DrawStatsSystem.h
    App/DrawStatsSystem.cpp
    App/main.cpp
    App/Common.h
    App/IRenderer.h
    App/IRenderer.cpp
    App/SoftwareRenderer.h
    App/SoftwareRenderer.cpp
    App/Vector3f.h
    App/Vector3f.cpp
    App/Matrix4.h
    App/Matrix4.cpp
    App/Vector2f.h
    App/Vector2f.cpp
    App/Vector4f.h
    App/Vector4f.cpp
    App/Math.h
    App/Math.cpp
    App/VertexInterpolator.h
    App/VertexInterpolator.cpp
    App/TransformedVertex.h
    App/TransformedVertex.cpp
    App/Texture.h
    App/Texture.cpp
    App/SimpleThreadPool.h
    App/SimpleThreadPool.cpp
    App/GlRenderer.h
    App/GlRenderer.cpp

    imgui/imgui.h
    imgui/imgui.cpp
    imgui/imgui-SFML.h
    imgui/imgui-SFML.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp

    stb/stb_image.h

    ImGuiFileDialog/ImGuiFileDialog.h
    ImGuiFileDialog/ImGuiFileDialog.cpp

    Dependencies/Tracy/TracyClient.cpp

)

#*****************************************************
# Find SFML library (set proper path and run find_package)
#*****************************************************

SET(SFML_DIR "./Dependencies/SFML-2.6.0/lib/cmake/SFML" )
set(SFML_STATIC_LIBRARIES TRUE)

add_definitions(-DTRACY_ENABLE)
add_definitions(-DTRACY_ON_DEMAND)

find_package(SFML 2.6.0 COMPONENTS graphics audio REQUIRED)

include_directories(SoftwareRenderer "${CMAKE_SOURCE_DIR}/imgui")
include_directories(SoftwareRenderer "${CMAKE_SOURCE_DIR}/Dependencies/assimp/include")
include_directories(SoftwareRenderer "${CMAKE_SOURCE_DIR}/Dependencies/Tracy")

#*****************************************************
# generate app exe
#*****************************************************

ADD_EXECUTABLE(SoftwareRenderer ${APP_SRC_FILES_LIST} )

#*****************************************************
# Set language version to C++20
#*****************************************************

set_property(TARGET SoftwareRenderer PROPERTY CXX_STANDARD 20)

#*****************************************************
# Specify which SFML libraries to link with exe
#*****************************************************

target_link_libraries(SoftwareRenderer sfml-graphics sfml-audio)
target_link_libraries(SoftwareRenderer ${CMAKE_SOURCE_DIR}/Dependencies/assimp/lib/assimp-vc143-mt.lib )
set_target_properties(SoftwareRenderer PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/Data"
)