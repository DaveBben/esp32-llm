#include <stdio.h>
#include <inttypes.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>

#include "esp_spiffs.h"
#include "sdkconfig.h"
#include "esp_err.h"
#include "esp_log.h"

#include "driver/uart.h"
#include "esp_vfs_dev.h"

#include "llm.h"
#include "llama.h"

#include <u8g2.h>
#include "u8g2_esp32_hal.h"
#include <driver/i2c.h>


static const char *TAG = "MAIN";

u8g2_t u8g2;

#define PIN_SDA 8
#define PIN_SCL 9
#define OLED_I2C_ADDRESS 0x78

#define PROMPT_BUFFER_SIZE 256
#define CONSOLE_UART_NUM UART_NUM_0


/**
 * @brief Configure SSD1306 display.
 *
 * Currently not called because the OLED is not connected.
 */
void init_display(void)
{
    u8g2_esp32_hal_t u8g2_esp32_hal = U8G2_ESP32_HAL_DEFAULT;

    u8g2_esp32_hal.bus.i2c.sda = PIN_SDA;
    u8g2_esp32_hal.bus.i2c.scl = PIN_SCL;

    u8g2_esp32_hal_init(u8g2_esp32_hal);

    u8g2_Setup_ssd1306_i2c_128x64_noname_f(
        &u8g2,
        U8G2_R0,
        u8g2_esp32_i2c_byte_cb,
        u8g2_esp32_gpio_and_delay_cb
    );

    u8x8_SetI2CAddress(
        &u8g2.u8x8,
        OLED_I2C_ADDRESS
    );

    u8g2_InitDisplay(&u8g2);
    u8g2_SetPowerSave(&u8g2, 0);

    u8g2_ClearBuffer(&u8g2);

    u8g2_SetFont(
        &u8g2,
        u8g2_font_ncenB08_tr
    );

    u8g2_SendBuffer(&u8g2);

    ESP_LOGI(TAG, "Display initialized");
}


/**
 * @brief Mount SPIFFS containing the model and tokenizer.
 */
void init_storage(void)
{
    ESP_LOGI(TAG, "Initializing SPIFFS");

    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/data",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = false
    };

    esp_err_t ret = esp_vfs_spiffs_register(&conf);

    if (ret != ESP_OK)
    {
        if (ret == ESP_FAIL)
        {
            ESP_LOGE(
                TAG,
                "Failed to mount or format filesystem"
            );
        }
        else if (ret == ESP_ERR_NOT_FOUND)
        {
            ESP_LOGE(
                TAG,
                "Failed to find SPIFFS partition"
            );
        }
        else
        {
            ESP_LOGE(
                TAG,
                "Failed to initialize SPIFFS (%s)",
                esp_err_to_name(ret)
            );
        }

        return;
    }

    size_t total = 0;
    size_t used = 0;

    ret = esp_spiffs_info(
        NULL,
        &total,
        &used
    );

    if (ret != ESP_OK)
    {
        ESP_LOGE(
            TAG,
            "Failed to get SPIFFS partition information (%s)",
            esp_err_to_name(ret)
        );
    }
    else
    {
        ESP_LOGI(
            TAG,
            "Partition size: total: %d, used: %d",
            total,
            used
        );
    }
}


/**
 * @brief Write text to OLED.
 *
 * Currently unused.
 */
void write_display(char *text)
{
    u8g2_ClearBuffer(&u8g2);

    u8g2_DrawStr(
        &u8g2,
        0,
        u8g2_GetDisplayHeight(&u8g2) / 2,
        text
    );

    u8g2_SendBuffer(&u8g2);
}


/**
 * @brief Callback after LLM generation completes.
 */
void generate_complete_cb(float tk_s)
{
    ESP_LOGI(
        TAG,
        "Generation speed: %.2f tok/s",
        tk_s
    );
}


/**
 * @brief Draw llama image on OLED.
 *
 * Currently unused.
 */
void draw_llama(void)
{
    u8g2_DrawXBM(
        &u8g2,
        0,
        0,
        u8g2_GetDisplayWidth(&u8g2),
        u8g2_GetDisplayHeight(&u8g2),
        llama_bmp
    );

    u8g2_SendBuffer(&u8g2);
}


/**
 * @brief Configure UART0 for blocking stdin.
 */
void init_serial_input(void)
{
    ESP_LOGI(
        TAG,
        "Initializing UART0 console input"
    );

    /*
     * Install UART driver.
     *
     * UART0 is already the ESP-IDF console,
     * but the driver lets VFS use blocking reads.
     */
    esp_err_t ret = uart_driver_install(
        CONSOLE_UART_NUM,
        1024,
        1024,
        0,
        NULL,
        0
    );

    /*
     * UART0 may already have a driver installed.
     * Treat that case as non-fatal.
     */
    if (
        ret != ESP_OK &&
        ret != ESP_ERR_INVALID_STATE
    )
    {
        ESP_LOGE(
            TAG,
            "UART driver install failed: %s",
            esp_err_to_name(ret)
        );

        return;
    }

    /*
     * Route stdin/stdout through UART driver mode.
     */
    esp_vfs_dev_uart_use_driver(
        CONSOLE_UART_NUM
    );

    /*
     * Disable stdio buffering.
     */
    setvbuf(
        stdin,
        NULL,
        _IONBF,
        0
    );

    setvbuf(
        stdout,
        NULL,
        _IONBF,
        0
    );

    ESP_LOGI(
        TAG,
        "UART0 console ready"
    );
}


/**
 * @brief Read one prompt from UART0.
 */
bool read_prompt(
    char *buffer,
    size_t buffer_size
)
{
    printf("\n");
    printf("========================================\n");
    printf("ESP32 Tiny LLM\n");
    printf("========================================\n");
    printf("Enter prompt: ");

    fflush(stdout);

    /*
     * fgets() should now block until Enter is pressed.
     */
    if (
        fgets(
            buffer,
            buffer_size,
            stdin
        ) == NULL
    )
    {
        ESP_LOGE(
            TAG,
            "Failed to read prompt"
        );

        return false;
    }

    /*
     * Remove CR/LF characters.
     */
    size_t len =
        strlen(buffer);

    while (
        len > 0 &&
        (
            buffer[len - 1] == '\n' ||
            buffer[len - 1] == '\r'
        )
    )
    {
        buffer[len - 1] =
            '\0';

        len--;
    }

    if (len == 0)
    {
        return false;
    }

    return true;
}


void app_main(void)
{
    ESP_LOGI(
        TAG,
        "Starting ESP32 LLM application"
    );


    /*
     * ---------------------------------------------------------
     * Interactive UART console
     * ---------------------------------------------------------
     */

    init_serial_input();


    /*
     * ---------------------------------------------------------
     * OLED currently disabled
     * ---------------------------------------------------------
     */

    // init_display();
    // write_display("Loading Model");


    /*
     * ---------------------------------------------------------
     * Mount model storage
     * ---------------------------------------------------------
     */

    init_storage();


    /*
     * ---------------------------------------------------------
     * Model files
     * ---------------------------------------------------------
     */

    char *checkpoint_path =
        "/data/stories260K.bin";

    char *tokenizer_path =
        "/data/tok512.bin";


    /*
     * ---------------------------------------------------------
     * Generation parameters
     * ---------------------------------------------------------
     *
     * temperature = 0:
     * greedy deterministic generation.
     *
     * topp = 1:
     * nucleus sampling disabled.
     */

    float temperature = 0.0f;
    float topp = 1.0f;

    int steps = 256;


    /*
     * RNG seed
     */
    unsigned long long rng_seed = 0;

    if (rng_seed <= 0)
    {
        rng_seed =
            (unsigned int)time(NULL);
    }


    /*
     * ---------------------------------------------------------
     * Load Transformer
     * ---------------------------------------------------------
     */

    Transformer transformer;

    ESP_LOGI(
        TAG,
        "LLM checkpoint path: %s",
        checkpoint_path
    );

    ESP_LOGI(
        TAG,
        "Loading Transformer model..."
    );

    build_transformer(
        &transformer,
        checkpoint_path
    );

    ESP_LOGI(
        TAG,
        "Transformer model loaded"
    );


    /*
     * Prevent generation beyond model context length.
     */
    if (
        steps == 0 ||
        steps > transformer.config.seq_len
    )
    {
        steps =
            transformer.config.seq_len;
    }


    /*
     * ---------------------------------------------------------
     * Load tokenizer
     * ---------------------------------------------------------
     */

    Tokenizer tokenizer;

    ESP_LOGI(
        TAG,
        "Loading tokenizer: %s",
        tokenizer_path
    );

    build_tokenizer(
        &tokenizer,
        tokenizer_path,
        transformer.config.vocab_size
    );

    ESP_LOGI(
        TAG,
        "Tokenizer loaded"
    );


    /*
     * ---------------------------------------------------------
     * Build sampler
     * ---------------------------------------------------------
     */

    Sampler sampler;

    build_sampler(
        &sampler,
        transformer.config.vocab_size,
        temperature,
        topp,
        rng_seed
    );

    ESP_LOGI(
        TAG,
        "Sampler initialized"
    );


    /*
     * ---------------------------------------------------------
     * Interactive prompt loop
     * ---------------------------------------------------------
     */

    char prompt_buffer[
        PROMPT_BUFFER_SIZE
    ];

    ESP_LOGI(
        TAG,
        "LLM ready for interactive prompts"
    );

    while (1)
    {
        memset(
            prompt_buffer,
            0,
            sizeof(prompt_buffer)
        );


        /*
         * Wait until the user types a line.
         */
        if (
            !read_prompt(
                prompt_buffer,
                sizeof(prompt_buffer)
            )
        )
        {
            continue;
        }


        ESP_LOGI(
            TAG,
            "Prompt received: %s",
            prompt_buffer
        );


        /*
         * OLED support can later show:
         *
         * write_display("Thinking");
         */


        printf("\nAI: ");

        fflush(stdout);


        /*
         * Run LLM inference.
         */
        generate(
            &transformer,
            &tokenizer,
            &sampler,
            prompt_buffer,
            steps,
            &generate_complete_cb
        );


        ESP_LOGI(
            TAG,
            "LLM generation complete"
        );
    }
}