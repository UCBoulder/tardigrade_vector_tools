#===================================================================================================== PROJECT SETUP ===
cmake_minimum_required(VERSION 3.30)
project(tardigrade_vector_tools LANGUAGES CXX)

# Set common project paths relative to project root directory
set(CPP_SRC_PATH "src/cpp")
set(CMAKE_SRC_PATH "src/cmake")

# Add a flag for if a full build of all tardigrade repositories should be performed
set(TARDIGRADE_FULL_BUILD OFF CACHE BOOL "Flag for whether a full build of Tardigrade should be performed (i.e., all repos pulled from git and built)")

# Add the cmake folder to locate project CMake module(s)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/${CMAKE_SRC_PATH}" ${CMAKE_MODULE_PATH})

# Set build type checks
string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_lower)
set(not_conda_test "true")
if(cmake_build_type_lower STREQUAL "conda-test")
    set(not_conda_test "false")
endif()

set(ADDITIONAL_HEADER_ONLY_LIBRARIES "tardigrade_interp_ndCartesian")
if(${not_conda_test} STREQUAL "true")
    foreach(package ${ADDITIONAL_HEADER_ONLY_LIBRARIES})
        add_library(${package} INTERFACE "${CPP_SRC_PATH}/${package}.cpp" "${CPP_SRC_PATH}/${package}.h")
    endforeach(package)
endif()

# Get version number from Git
set(VERSION_UPDATE_FROM_GIT True)
if(${not_conda_test} STREQUAL "true")
    if (DEFINED ENV{BUILD_PREFIX})
        set(Python_ROOT_DIR "$ENV{BUILD_PREFIX}/bin")
    endif()
    set(Python_FIND_STRATEGY LOCATION)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    execute_process(COMMAND ${Python_EXECUTABLE} -m setuptools_scm
                            OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_STRING_FULL
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+\.[0-9]+\.[0-9]+" ${PROJECT_NAME}_VERSION ${${PROJECT_NAME}_VERSION_STRING_FULL})
    if(${${PROJECT_NAME}_VERSION} STREQUAL "")
        set(${PROJECT_NAME}_VERSION 0.0.0)
    endif()
    project(${PROJECT_NAME} VERSION ${${PROJECT_NAME}_VERSION} LANGUAGES CXX)
endif()

# Add installation directory variables
include(GNUInstallDirs)

# Set the c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt -O3 -DNDEBUG")
endif()

# Enable CTest
enable_testing()

#================================================================================================= FIND DEPENDENCIES ===
# Find eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
if(EIGEN3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
endif()

# Set build type checks
string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_lower)
set(upstream_required "")
set(not_conda_test "true")
set(project_link_string ${PROJECT_NAME})
if(cmake_build_type_lower STREQUAL "release")
    set(upstream_required "REQUIRED")
elseif(cmake_build_type_lower STREQUAL "conda-test")
    set(upstream_required "REQUIRED")
    set(not_conda_test "false")
endif()

# Find related, but external, projects in installed environment
include(FetchContent)
set(upstream_packages "tardigrade_error_tools")
foreach(package ${upstream_packages})
    string(TOUPPER "${package}" package_upper)
    if(TARDIGRADE_FULL_BUILD)
        message("Not attempting to find ${package}")
    else()
        find_package(${package} ${upstream_required} CONFIG)
    endif()
    if(${package}_FOUND)
        message(STATUS "Found ${package}: ${${package}_DIR}")
    else()
        # Find related, but external, projects using FetchContent and building locally
        message(WARNING "Did not find an installed ${package} package. Attempting local build with FetchContent.")
        if(NOT DEFINED ${package_upper}_FETCHCONTENT_VERSION)
            set(${package_upper}_FETCHCONTENT_VERSION "origin/dev")
        endif()
        message("${package_upper} is being built with version ${${package_upper}_FETCHCONTENT_VERSION}")
        FetchContent_Declare(
            ${package}
            GIT_REPOSITORY https://github.com/UCBoulder/${package}.git
            GIT_TAG        ${${package_upper}_FETCHCONTENT_VERSION}
            SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/${package}-src
            BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/${package}-src/build
        )
        FetchContent_MakeAvailable(${package})
    endif()
endforeach(package)

#=============================================================================================== ADD PROJECT TARGETS ===
# MUST COME AFTER DEPENDENCY LOCATING
# Add project source directories
if(${not_conda_test} STREQUAL "true")
    include_directories("${CPP_SRC_PATH}")
    add_subdirectory("${CPP_SRC_PATH}")
endif()

# Only add tests and documentation for current project builds. Protects downstream project builds.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Find Boost. Required for tests
    find_package(Boost 1.82.0 REQUIRED)
    # Add c++ tests and docs
    add_subdirectory("${CPP_SRC_PATH}/tests")
    if(${not_conda_test} STREQUAL "true")
        add_subdirectory("docs")
    endif()
endif()

#==================================================================================== SETUP INSTALLATION CMAKE FILES ===
if(${not_conda_test} STREQUAL "true")
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                     VERSION ${PROJECT_VERSION}
                                     COMPATIBILITY SameMajorVersion)
    configure_package_config_file(
      "${PROJECT_SOURCE_DIR}/${CMAKE_SRC_PATH}/${PROJECT_NAME}Config.cmake.in"
      "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
      INSTALL_DESTINATION
      ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

    # CMake won't build the targets for local builds of upstream projects
    if(cmake_build_type_lower STREQUAL release)
        install(EXPORT ${PROJECT_NAME}_Targets
                FILE ${PROJECT_NAME}Targets.cmake
                DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
    endif()

    install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
endif()
