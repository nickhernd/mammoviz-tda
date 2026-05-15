# Compile GLSL → SPIR-V using glslangValidator
find_program(GLSLANG glslangValidator)

set(SHADER_SOURCES
    glsl/volume_raycast.vert
    glsl/volume_raycast.frag
    glsl/volume_raycast.comp
    glsl/diagram.vert
    glsl/diagram.frag
    glsl/manifold.vert
    glsl/manifold.frag
)

foreach(shader ${SHADER_SOURCES})
    get_filename_component(name ${shader} NAME)
    set(out "${CMAKE_CURRENT_SOURCE_DIR}/spirv/${name}.spv")
    if(GLSLANG)
        add_custom_command(
            OUTPUT  ${out}
            COMMAND ${GLSLANG} -V ${CMAKE_CURRENT_SOURCE_DIR}/${shader} -o ${out}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${shader}
            COMMENT "Compiling shader: ${shader}"
        )
        list(APPEND SPIRV_OUTPUTS ${out})
    endif()
endforeach()

if(GLSLANG AND SPIRV_OUTPUTS)
    add_custom_target(shaders ALL DEPENDS ${SPIRV_OUTPUTS})
endif()
