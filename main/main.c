#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <time.h>
#include <stdlib.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2c.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_spiffs.h"

#include "llm.h"


static const char *TAG = "AI_ASSISTANT";


/* =========================================================
 * CONFIGURATION
 * =========================================================
 */

#define I2C_PORT           I2C_NUM_0
#define OLED_SDA_GPIO      5
#define OLED_SCL_GPIO      6
#define I2C_FREQ_HZ        50000

#define OLED_ADDR          0x3C

#define OLED_WIDTH         128
#define OLED_HEIGHT        64
#define OLED_PAGES         8

#define CONSOLE_UART_NUM   UART_NUM_0
#define PROMPT_BUFFER_SIZE 256


static uint8_t framebuffer[
    OLED_WIDTH * OLED_PAGES
];


/* =========================================================
 * EMOTION TYPES
 * =========================================================
 */

typedef enum
{
    EMOTION_NEUTRAL,
    EMOTION_HAPPY,
    EMOTION_SAD,
    EMOTION_ANGRY,
    EMOTION_SURPRISED

} emotion_t;


/* =========================================================
 * FRAMEBUFFER
 * =========================================================
 */

static void fb_clear(void)
{
    memset(
        framebuffer,
        0,
        sizeof(framebuffer)
    );
}


static void set_pixel(
    int x,
    int y
)
{
    if (
        x < 0 ||
        x >= OLED_WIDTH ||
        y < 0 ||
        y >= OLED_HEIGHT
    )
    {
        return;
    }

    int page =
        y / 8;

    int bit =
        y % 8;

    framebuffer[
        page * OLED_WIDTH + x
    ] |=
        (1 << bit);
}


/* =========================================================
 * GRAPHICS
 * =========================================================
 */

static void draw_line(
    int x0,
    int y0,
    int x1,
    int y1
)
{
    int dx =
        abs(x1 - x0);

    int sx =
        x0 < x1 ? 1 : -1;

    int dy =
        -abs(y1 - y0);

    int sy =
        y0 < y1 ? 1 : -1;

    int err =
        dx + dy;


    while (1)
    {
        set_pixel(
            x0,
            y0
        );

        if (
            x0 == x1 &&
            y0 == y1
        )
        {
            break;
        }

        int e2 =
            2 * err;

        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }

        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
    }
}


static void draw_circle(
    int cx,
    int cy,
    int radius
)
{
    int x =
        radius;

    int y =
        0;

    int err =
        0;


    while (x >= y)
    {
        set_pixel(cx + x, cy + y);
        set_pixel(cx + y, cy + x);

        set_pixel(cx - y, cy + x);
        set_pixel(cx - x, cy + y);

        set_pixel(cx - x, cy - y);
        set_pixel(cx - y, cy - x);

        set_pixel(cx + y, cy - x);
        set_pixel(cx + x, cy - y);

        y++;

        if (err <= 0)
        {
            err +=
                2 * y + 1;
        }

        if (err > 0)
        {
            x--;

            err -=
                2 * x + 1;
        }
    }
}


/* =========================================================
 * SIMPLE FONT
 * =========================================================
 */

static const uint8_t FONT_SPACE[5] =
{
    0x00, 0x00, 0x00, 0x00, 0x00
};

static const uint8_t FONT_A[5] =
{
    0x7E, 0x11, 0x11, 0x11, 0x7E
};

static const uint8_t FONT_D[5] =
{
    0x7F, 0x41, 0x41, 0x22, 0x1C
};

static const uint8_t FONT_E[5] =
{
    0x7F, 0x49, 0x49, 0x49, 0x41
};

static const uint8_t FONT_G[5] =
{
    0x3E, 0x41, 0x49, 0x49, 0x3A
};

static const uint8_t FONT_H[5] =
{
    0x7F, 0x08, 0x08, 0x08, 0x7F
};

static const uint8_t FONT_I[5] =
{
    0x00, 0x41, 0x7F, 0x41, 0x00
};

static const uint8_t FONT_K[5] =
{
    0x7F, 0x08, 0x14, 0x22, 0x41
};

static const uint8_t FONT_L[5] =
{
    0x7F, 0x40, 0x40, 0x40, 0x40
};

static const uint8_t FONT_N[5] =
{
    0x7F, 0x04, 0x08, 0x10, 0x7F
};

static const uint8_t FONT_O[5] =
{
    0x3E, 0x41, 0x41, 0x41, 0x3E
};

static const uint8_t FONT_P[5] =
{
    0x7F, 0x09, 0x09, 0x09, 0x06
};

static const uint8_t FONT_R[5] =
{
    0x7F, 0x09, 0x19, 0x29, 0x46
};

static const uint8_t FONT_S[5] =
{
    0x46, 0x49, 0x49, 0x49, 0x31
};

static const uint8_t FONT_T[5] =
{
    0x01, 0x01, 0x7F, 0x01, 0x01
};

static const uint8_t FONT_U[5] =
{
    0x3F, 0x40, 0x40, 0x40, 0x3F
};

static const uint8_t FONT_V[5] =
{
    0x1F, 0x20, 0x40, 0x20, 0x1F
};

static const uint8_t FONT_Y[5] =
{
    0x07, 0x08, 0x70, 0x08, 0x07
};


static const uint8_t *
get_font_char(
    char c
)
{
    switch (c)
    {
        case 'A': return FONT_A;
        case 'D': return FONT_D;
        case 'E': return FONT_E;
        case 'G': return FONT_G;
        case 'H': return FONT_H;
        case 'I': return FONT_I;
        case 'K': return FONT_K;
        case 'L': return FONT_L;
        case 'N': return FONT_N;
        case 'O': return FONT_O;
        case 'P': return FONT_P;
        case 'R': return FONT_R;
        case 'S': return FONT_S;
        case 'T': return FONT_T;
        case 'U': return FONT_U;
        case 'V': return FONT_V;
        case 'Y': return FONT_Y;

        case ' ':
        default:
            return FONT_SPACE;
    }
}


static void draw_char(
    int x,
    int y,
    char c
)
{
    const uint8_t *bitmap =
        get_font_char(c);

    for (
        int col = 0;
        col < 5;
        col++
    )
    {
        uint8_t bits =
            bitmap[col];

        for (
            int row = 0;
            row < 7;
            row++
        )
        {
            if (
                bits &
                (1 << row)
            )
            {
                set_pixel(
                    x + col,
                    y + row
                );
            }
        }
    }
}


static void draw_text(
    int x,
    int y,
    const char *text
)
{
    int cursor =
        x;

    while (*text)
    {
        draw_char(
            cursor,
            y,
            toupper(
                (unsigned char)*text
            )
        );

        cursor += 6;
        text++;
    }
}


/* =========================================================
 * I2C
 * =========================================================
 */

static esp_err_t init_i2c(void)
{
    ESP_LOGI(
        TAG,
        "Initializing I2C"
    );

    i2c_config_t config =
    {
        .mode =
            I2C_MODE_MASTER,

        .sda_io_num =
            OLED_SDA_GPIO,

        .scl_io_num =
            OLED_SCL_GPIO,

        .sda_pullup_en =
            GPIO_PULLUP_ENABLE,

        .scl_pullup_en =
            GPIO_PULLUP_ENABLE,

        .master.clk_speed =
            I2C_FREQ_HZ,

        .clk_flags =
            0
    };

    esp_err_t ret =
        i2c_param_config(
            I2C_PORT,
            &config
        );

    if (ret != ESP_OK)
    {
        ESP_LOGE(
            TAG,
            "I2C config failed: %s",
            esp_err_to_name(ret)
        );

        return ret;
    }

    ret =
        i2c_driver_install(
            I2C_PORT,
            I2C_MODE_MASTER,
            0,
            0,
            0
        );

    if (
        ret != ESP_OK &&
        ret != ESP_ERR_INVALID_STATE
    )
    {
        ESP_LOGE(
            TAG,
            "I2C driver failed: %s",
            esp_err_to_name(ret)
        );

        return ret;
    }

    ESP_LOGI(
        TAG,
        "I2C ready SDA=%d SCK=%d",
        OLED_SDA_GPIO,
        OLED_SCL_GPIO
    );

    return ESP_OK;
}


/* =========================================================
 * OLED
 * =========================================================
 */

static esp_err_t oled_probe(void)
{
    i2c_cmd_handle_t cmd =
        i2c_cmd_link_create();

    if (cmd == NULL)
    {
        return ESP_ERR_NO_MEM;
    }

    i2c_master_start(cmd);

    i2c_master_write_byte(
        cmd,
        (OLED_ADDR << 1)
            | I2C_MASTER_WRITE,
        true
    );

    i2c_master_stop(cmd);

    esp_err_t ret =
        i2c_master_cmd_begin(
            I2C_PORT,
            cmd,
            pdMS_TO_TICKS(100)
        );

    i2c_cmd_link_delete(cmd);

    return ret;
}


static bool wait_for_oled(void)
{
    ESP_LOGI(
        TAG,
        "Waiting for OLED at 0x3C"
    );

    for (
        int attempt = 1;
        attempt <= 30;
        attempt++
    )
    {
        if (
            oled_probe() ==
            ESP_OK
        )
        {
            ESP_LOGI(
                TAG,
                "OLED detected"
            );

            return true;
        }

        ESP_LOGW(
            TAG,
            "OLED retry %d",
            attempt
        );

        vTaskDelay(
            pdMS_TO_TICKS(200)
        );
    }

    return false;
}


static esp_err_t oled_command(
    uint8_t command
)
{
    uint8_t data[2] =
    {
        0x00,
        command
    };

    return
        i2c_master_write_to_device(
            I2C_PORT,
            OLED_ADDR,
            data,
            sizeof(data),
            pdMS_TO_TICKS(200)
        );
}


static esp_err_t oled_data(
    const uint8_t *data,
    size_t length
)
{
    uint8_t packet[17];

    size_t offset =
        0;

    while (
        offset < length
    )
    {
        size_t chunk =
            length - offset;

        if (
            chunk > 16
        )
        {
            chunk =
                16;
        }

        packet[0] =
            0x40;

        memcpy(
            &packet[1],
            &data[offset],
            chunk
        );

        esp_err_t ret =
            i2c_master_write_to_device(
                I2C_PORT,
                OLED_ADDR,
                packet,
                chunk + 1,
                pdMS_TO_TICKS(200)
            );

        if (
            ret != ESP_OK
        )
        {
            return ret;
        }

        offset +=
            chunk;
    }

    return ESP_OK;
}


static bool init_oled(void)
{
    const uint8_t sequence[] =
    {
        0xAE,

        0xD5,
        0x80,

        0xA8,
        0x3F,

        0xD3,
        0x00,

        0x40,

        0x8D,
        0x14,

        0x20,
        0x00,

        0xA1,

        0xC8,

        0xDA,
        0x12,

        0x81,
        0x7F,

        0xD9,
        0xF1,

        0xDB,
        0x40,

        0xA4,

        0xA6,

        0xAF
    };

    ESP_LOGI(
        TAG,
        "Initializing OLED"
    );

    for (
        size_t i = 0;
        i < sizeof(sequence);
        i++
    )
    {
        if (
            oled_command(
                sequence[i]
            ) != ESP_OK
        )
        {
            ESP_LOGE(
                TAG,
                "OLED initialization failed"
            );

            return false;
        }

        vTaskDelay(
            pdMS_TO_TICKS(2)
        );
    }

    ESP_LOGI(
        TAG,
        "OLED initialized"
    );

    return true;
}


static bool oled_refresh(void)
{
    if (oled_command(0x21) != ESP_OK)
        return false;

    if (oled_command(0x00) != ESP_OK)
        return false;

    if (oled_command(0x7F) != ESP_OK)
        return false;

    if (oled_command(0x22) != ESP_OK)
        return false;

    if (oled_command(0x00) != ESP_OK)
        return false;

    if (oled_command(0x07) != ESP_OK)
        return false;

    return
        oled_data(
            framebuffer,
            sizeof(framebuffer)
        ) == ESP_OK;
}


/* =========================================================
 * FACES
 * =========================================================
 */

static void draw_face_outline(void)
{
    draw_circle(
        64,
        27,
        23
    );
}


static void draw_happy_face(void)
{
    fb_clear();

    draw_face_outline();

    draw_line(
        48, 22,
        53, 18
    );

    draw_line(
        53, 18,
        58, 22
    );

    draw_line(
        70, 22,
        75, 18
    );

    draw_line(
        75, 18,
        80, 22
    );

    draw_line(
        50, 34,
        55, 39
    );

    draw_line(
        55, 39,
        64, 42
    );

    draw_line(
        64, 42,
        73, 39
    );

    draw_line(
        73, 39,
        78, 34
    );

    draw_text(
        49,
        55,
        "HAPPY"
    );

    oled_refresh();
}


static void draw_sad_face(void)
{
    fb_clear();

    draw_face_outline();

    draw_circle(
        53,
        22,
        2
    );

    draw_circle(
        75,
        22,
        2
    );

    draw_line(
        52, 42,
        58, 37
    );

    draw_line(
        58, 37,
        64, 35
    );

    draw_line(
        64, 35,
        70, 37
    );

    draw_line(
        70, 37,
        76, 42
    );

    draw_text(
        55,
        55,
        "SAD"
    );

    oled_refresh();
}


static void draw_angry_face(void)
{
    fb_clear();

    draw_face_outline();

    draw_line(
        47, 17,
        58, 22
    );

    draw_line(
        70, 22,
        81, 17
    );

    draw_circle(
        54,
        25,
        2
    );

    draw_circle(
        74,
        25,
        2
    );

    draw_line(
        53, 40,
        64, 35
    );

    draw_line(
        64, 35,
        75, 40
    );

    draw_text(
        49,
        55,
        "ANGRY"
    );

    oled_refresh();
}


static void draw_surprised_face(void)
{
    fb_clear();

    draw_face_outline();

    draw_circle(
        53,
        21,
        3
    );

    draw_circle(
        75,
        21,
        3
    );

    draw_circle(
        64,
        37,
        5
    );

    draw_text(
        37,
        55,
        "SURPRISED"
    );

    oled_refresh();
}


static void draw_neutral_face(void)
{
    fb_clear();

    draw_face_outline();

    draw_circle(
        53,
        22,
        2
    );

    draw_circle(
        75,
        22,
        2
    );

    draw_line(
        53, 39,
        75, 39
    );

    draw_text(
        43,
        55,
        "NEUTRAL"
    );

    oled_refresh();
}


static void draw_thinking_face(void)
{
    fb_clear();

    draw_face_outline();

    draw_circle(
        53,
        22,
        2
    );

    draw_circle(
        74,
        23,
        2
    );

    draw_line(
        69, 18,
        80, 15
    );

    draw_line(
        55, 39,
        72, 37
    );

    draw_text(
        40,
        55,
        "THINKING"
    );

    oled_refresh();
}


/* =========================================================
 * EMOTION DETECTION
 * =========================================================
 */

static bool contains_word(
    const char *text,
    const char *word
)
{
    char lower[
        PROMPT_BUFFER_SIZE
    ];

    size_t len =
        strlen(text);

    if (
        len >= sizeof(lower)
    )
    {
        len =
            sizeof(lower) - 1;
    }

    for (
        size_t i = 0;
        i < len;
        i++
    )
    {
        lower[i] =
            tolower(
                (unsigned char)
                text[i]
            );
    }

    lower[len] =
        '\0';

    return
        strstr(
            lower,
            word
        ) != NULL;
}


static emotion_t detect_emotion(
    const char *prompt
)
{
    if (
        contains_word(prompt, "happy") ||
        contains_word(prompt, "great") ||
        contains_word(prompt, "love") ||
        contains_word(prompt, "good") ||
        contains_word(prompt, "wonderful") ||
        contains_word(prompt, "excited")
    )
    {
        return
            EMOTION_HAPPY;
    }

    if (
        contains_word(prompt, "sad") ||
        contains_word(prompt, "cry") ||
        contains_word(prompt, "unhappy") ||
        contains_word(prompt, "lonely") ||
        contains_word(prompt, "hurt")
    )
    {
        return
            EMOTION_SAD;
    }

    if (
        contains_word(prompt, "angry") ||
        contains_word(prompt, "mad") ||
        contains_word(prompt, "hate") ||
        contains_word(prompt, "furious")
    )
    {
        return
            EMOTION_ANGRY;
    }

    if (
        contains_word(prompt, "wow") ||
        contains_word(prompt, "surprise") ||
        contains_word(prompt, "amazing") ||
        contains_word(prompt, "shocked")
    )
    {
        return
            EMOTION_SURPRISED;
    }

    return
        EMOTION_NEUTRAL;
}


static void show_emotion(
    emotion_t emotion
)
{
    switch (emotion)
    {
        case EMOTION_HAPPY:
            draw_happy_face();
            break;

        case EMOTION_SAD:
            draw_sad_face();
            break;

        case EMOTION_ANGRY:
            draw_angry_face();
            break;

        case EMOTION_SURPRISED:
            draw_surprised_face();
            break;

        case EMOTION_NEUTRAL:

        default:
            draw_neutral_face();
            break;
    }
}


/* =========================================================
 * STORAGE
 * =========================================================
 */

static bool init_storage(void)
{
    ESP_LOGI(
        TAG,
        "Initializing SPIFFS"
    );

    esp_vfs_spiffs_conf_t conf =
    {
        .base_path =
            "/data",

        .partition_label =
            NULL,

        .max_files =
            5,

        .format_if_mount_failed =
            false
    };

    esp_err_t ret =
        esp_vfs_spiffs_register(
            &conf
        );

    if (
        ret != ESP_OK
    )
    {
        ESP_LOGE(
            TAG,
            "SPIFFS failed: %s",
            esp_err_to_name(ret)
        );

        return false;
    }

    return true;
}


/* =========================================================
 * UART
 * =========================================================
 */

static bool init_serial(void)
{
    ESP_LOGI(
        TAG,
        "Initializing UART0"
    );

    esp_err_t ret =
        uart_driver_install(
            CONSOLE_UART_NUM,
            1024,
            1024,
            0,
            NULL,
            0
        );

    if (
        ret != ESP_OK &&
        ret != ESP_ERR_INVALID_STATE
    )
    {
        ESP_LOGE(
            TAG,
            "UART init failed: %s",
            esp_err_to_name(ret)
        );

        return false;
    }

    /*
     * Correct ESP-IDF 5.3 API.
     * Declared in driver/uart_vfs.h
     */
    uart_vfs_dev_use_driver(
        CONSOLE_UART_NUM
    );

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
        "UART0 ready"
    );

    return true;
}


/* =========================================================
 * PROMPT INPUT
 * =========================================================
 */

static bool read_prompt(
    char *buffer,
    size_t buffer_size
)
{
    printf("\n");
    printf("==============================\n");
    printf("ESP32 AI FACE\n");
    printf("==============================\n");
    printf("Enter prompt: ");

    fflush(stdout);

    if (
        fgets(
            buffer,
            buffer_size,
            stdin
        ) == NULL
    )
    {
        return false;
    }

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
        buffer[
            len - 1
        ] =
            '\0';

        len--;
    }

    return
        len > 0;
}


/* =========================================================
 * LLM CALLBACK
 * =========================================================
 */

static void generation_complete(
    float tokens_per_second
)
{
    ESP_LOGI(
        TAG,
        "Generation speed: %.2f tokens/sec",
        tokens_per_second
    );
}


/* =========================================================
 * MAIN
 * =========================================================
 */

void app_main(void)
{
    ESP_LOGI(
        TAG,
        "Starting ESP32 AI Face Assistant"
    );


    /*
     * OLED startup
     */
    vTaskDelay(
        pdMS_TO_TICKS(1500)
    );


    if (
        init_i2c() !=
        ESP_OK
    )
    {
        return;
    }


    if (
        !wait_for_oled()
    )
    {
        ESP_LOGE(
            TAG,
            "OLED not detected"
        );

        return;
    }


    if (
        !init_oled()
    )
    {
        ESP_LOGE(
            TAG,
            "OLED init failed"
        );

        return;
    }


    draw_neutral_face();


    /*
     * UART
     */
    if (
        !init_serial()
    )
    {
        return;
    }


    /*
     * Storage
     */
    if (
        !init_storage()
    )
    {
        return;
    }


    /*
     * Model files
     */
    char *checkpoint_path =
        "/data/stories260K.bin";


    char *tokenizer_path =
        "/data/tok512.bin";


    /*
     * Generation settings
     */
    float temperature =
        0.0f;


    float topp =
        1.0f;


    int steps =
        256;


    unsigned long long rng_seed =
        (unsigned int)
        time(NULL);


    /*
     * Transformer
     */
    Transformer transformer;


    draw_thinking_face();


    ESP_LOGI(
        TAG,
        "Loading Transformer..."
    );


    build_transformer(
        &transformer,
        checkpoint_path
    );


    ESP_LOGI(
        TAG,
        "Transformer loaded"
    );


    if (
        steps >
        transformer.config.seq_len
    )
    {
        steps =
            transformer.config.seq_len;
    }


    /*
     * Tokenizer
     */
    Tokenizer tokenizer;


    ESP_LOGI(
        TAG,
        "Loading tokenizer..."
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
     * Sampler
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
        "Sampler ready"
    );


    /*
     * Ready
     */
    draw_happy_face();


    ESP_LOGI(
        TAG,
        "AI Assistant ready"
    );


    char prompt[
        PROMPT_BUFFER_SIZE
    ];


    /*
     * Interactive loop
     */
    while (1)
    {
        memset(
            prompt,
            0,
            sizeof(prompt)
        );


        if (
            !read_prompt(
                prompt,
                sizeof(prompt)
            )
        )
        {
            continue;
        }


        ESP_LOGI(
            TAG,
            "Prompt received: %s",
            prompt
        );


        emotion_t emotion =
            detect_emotion(
                prompt
            );


        /*
         * Show emotion immediately.
         */
        show_emotion(
            emotion
        );


        vTaskDelay(
            pdMS_TO_TICKS(1000)
        );


        /*
         * LLM generating.
         */
        draw_thinking_face();


        printf("\nAI: ");

        fflush(stdout);


        /*
         * Your local generate() requires seven arguments.
         *
         * Seventh parameter is the token callback.
         * We aren't using it yet.
         */
        generate(
            &transformer,
            &tokenizer,
            &sampler,
            prompt,
            steps,
            &generation_complete,
            NULL
        );


        /*
         * Return to detected expression.
         */
        show_emotion(
            emotion
        );


        ESP_LOGI(
            TAG,
            "Generation complete"
        );
    }
}