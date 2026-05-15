# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 4.3.2)

# Reject any attempt to use a toolchain file. We must not use one because
# we could be downloading it here. If the CMAKE_TOOLCHAIN_FILE environment
# variable is set, the cache variable will have been initialized from it.
unset(CMAKE_TOOLCHAIN_FILE CACHE)
unset(ENV{CMAKE_TOOLCHAIN_FILE})

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(gudhi-populate NONE)



include(ExternalProject)
ExternalProject_Add(gudhi-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "TRUE" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "URL" "https://github.com/GUDHI/gudhi-devel/releases/download/tags%2Fgudhi-release-3.10.0/gudhi.3.10.0.tar.gz" "URL_HASH" "MD5=00fcb5b873bb806f959178c4bf5d2f9b"
                    SOURCE_DIR          "/home/nickhernd/Desktop/mammoviz-tda/build_test/_deps/gudhi-src"
                    BINARY_DIR          "/home/nickhernd/Desktop/mammoviz-tda/build_test/_deps/gudhi-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


