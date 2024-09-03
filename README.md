# Running an LLM on the ESP32
![LLM on ESP32](/ESP32_LLM.jpg)

## Summary
I wanted to see if it was possible to run a Large Language Model (LLM) on the ESP32. Surprisingly it is possible, though probably not very useful.

The "Large" Language Model used is actually quite small. It is a 260K parameter [tinyllamas checkpoint](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K) trained on the [tiny stories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

The LLM implementation is done using [llama.2c](https://github.com/karpathy/llama2.c) with minor optimizations to make it run faster on the ESP32.

## Hardware
LLMs require a great deal of memory. Even this small one still requires 1MB of RAM. I used the [ESP32-S3FH4R2](https://www.mouser.com/ProductDetail/Espressif-Systems/ESP32-S3FH4R2?qs=tlsG%2FOw5FFjPrwkmZSBQNA%3D%3D) because it has 2MB of embedded PSRAM.

## Optimizing Llama2.c for the ESP32

With the following changes to `llama2.c`, I am able to achieve **19.13 tok/s**:

1. Utilizing both cores of the ESP32 during math heavy operations.
2. Utilizing some special [dot product functions](https://github.com/espressif/esp-dsp/tree/master/modules/dotprod/float) from the [ESP-DSP library](https://github.com/espressif/esp-dsp) that are designed for the ESP32-S3. These functions utilize some of the [few SIMD instructions](https://bitbanksoftware.blogspot.com/2024/01/surprise-esp32-s3-has-few-simd.html) the ESP32-S3 has.
3. Maxing out CPU speed to 240 MHz and PSRAM speed to 80MHZ and increasing the instruction cache size.


## Setup
This requires the [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html#installation) toolchain to be installed

```
idf.py build
idf.py -p /dev/{DEVICE_PORT} flash
```


