idf_component_register(SRCS "hello_world_main.c" "llm.c"
                    INCLUDE_DIRS "")

# Create a SPIFFS image from the contents of the 'spiffs_image' directory
# that fits the partition named 'data'. FLASH_IN_PROJECT indicates that
# the generated image should be flashed when the entire project is flashed to
# the target with 'idf.py -p PORT flash'.
spiffs_create_partition_image(data ../data FLASH_IN_PROJECT)
