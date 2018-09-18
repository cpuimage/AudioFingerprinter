cmake_minimum_required(VERSION 2.8)
project(AudioFingerprinter LANGUAGES C)
add_executable(AudioFingerprinter AudioFingerprinter.c)
target_link_libraries(AudioFingerprinter -lm)