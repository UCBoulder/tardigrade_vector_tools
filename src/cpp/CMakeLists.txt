add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(
    ${PROJECT_NAME}
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/${CPP_SRC_PATH}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}_Targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES ${PROJECT_NAME}.h ${PROJECT_NAME}.cpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

get_target_property(target_sources ${PROJECT_NAME} SOURCES)
foreach(source ${target_sources})
    set_property(GLOBAL APPEND PROPERTY CLANG_FORMAT_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
endforeach(source)

foreach(package ${ADDITIONAL_HEADER_ONLY_LIBRARIES})
    target_include_directories(
        ${package}
        INTERFACE
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/${CPP_SRC_PATH}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    install(
        TARGETS ${package}
        EXPORT ${package}_Targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(FILES ${package}.h ${package}.cpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    get_target_property(target_sources ${package} SOURCES)
    foreach(source ${target_sources})
        set_property(GLOBAL APPEND PROPERTY CLANG_FORMAT_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    endforeach(source)
endforeach(package)
