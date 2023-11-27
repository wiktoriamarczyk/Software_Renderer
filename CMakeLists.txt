#*****************************************************
# specify minimum version of CMake to use
# (find_package requires 3.23)
#*****************************************************
cmake_minimum_required(VERSION 3.23)

#*****************************************************
project(SoftwareRenderer LANGUAGES CXX)
#*****************************************************

#*****************************************************
# add source files
#*****************************************************

SET(APP_SRC_FILES_LIST
    App/main.cpp
    App/Common.h
    App/Renderer.h
    App/Renderer.cpp
    App/Vector3f.h
    App/Vector3f.cpp
    App/Matrix4.h
    App/Matrix4.cpp
	App/Vector2f.h
	App/Vector2f.cpp
	App/Vector4f.h
	App/Vector4f.cpp
	App/Line2D.h
	App/Line2D.cpp

    imgui/imgui.h
    imgui/imgui.cpp
    imgui/imgui-SFML.h
    imgui/imgui-SFML.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
	
	stb/stb_image.h
	stb/stb_image.cpp

)

#*****************************************************
# Find SFML library (set proper path and run find_package)
#*****************************************************

SET(SFML_DIR "./Dependencies/SFML-2.6.0/lib/cmake/SFML" )
set(SFML_STATIC_LIBRARIES TRUE)

find_package(SFML 2.6.0 COMPONENTS graphics audio REQUIRED)


include_directories(SoftwareRenderer "./imgui")
include_directories(SoftwareRenderer "./Dependencies/assimp/include")

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
target_link_libraries(SoftwareRenderer F:/repos/Git-C++/SoftwareRenderer/Dependencies/assimp/lib/assimp-vc143-mt.lib )