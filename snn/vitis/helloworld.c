#include <stdio.h>
#include <stdint.h>

#include "xil_printf.h"
#include "xil_io.h"
#include "xparameters.h"
#include "bspconfig.h"

// Vitis 2023.2+ (incl. 2026.1) uses the System Device Tree (SDT) flow:
//  - xtime_l.h was replaced by the xiltimer library (xiltimer.h)
//  - drivers are looked up by base address instead of DEVICE_ID
//  - platform.h / init_platform() are not part of the empty template
#ifdef SDT
#include "xiltimer.h"
#include "xtimer_config.h"
#else
#include "xtime_l.h"
#endif
#include "xuartps.h"
#include "xgpio.h"
#include "xsnn_mlp_hw.h"

extern char inbyte(void);
extern void outbyte(char c);

// ============================================================
// UART CONFIG
// ============================================================
// UART used by inbyte()/outbyte() (BSP stdin/stdout).
// SDT: base address; legacy flow: device ID.
#ifndef UART_ID
#ifdef SDT
#define UART_ID          STDOUT_BASEADDRESS
#else
#define UART_ID          XPAR_XUARTPS_0_DEVICE_ID
#endif
#endif

#ifndef UART_BAUD
#define UART_BAUD        4000000u
#endif

// ============================================================
// SNN feature config
// Python sends already-preprocessed int16 features
// 12x16 grid, ON/OFF = 384 features for T=1
// ============================================================
#define T_STEPS     1
#define GRID_H      12
#define GRID_W      16
#define N_FEATURES  (GRID_H * GRID_W * 2)
#define RX_BYTES    (N_FEATURES * sizeof(int16_t))

// Python packet:
// 0xAA 0x55 <384 int16 little-endian features>
#define FRAME_START_0  0xAA
#define FRAME_START_1  0x55

// ============================================================
// SNN AXI-Lite base + registers
// ============================================================
#ifdef SDT
#define IP_BASE_ADDR  XPAR_SNN_MLP_0_BASEADDR
#else
#define IP_BASE_ADDR  XPAR_SNN_MLP_0_S_AXI_CTRL_BASEADDR
#endif

#define REG_AP_CTRL        XSNN_MLP_CTRL_ADDR_AP_CTRL
#define REG_FEATURE_MEM_L  XSNN_MLP_CTRL_ADDR_FEATURE_MEM_DATA
#define REG_FEATURE_MEM_H  (XSNN_MLP_CTRL_ADDR_FEATURE_MEM_DATA + 0x04)

#define REG_PRED_OUT       XSNN_MLP_CTRL_ADDR_PREDICTION_DATA
#define REG_LOGITS_BASE    XSNN_MLP_CTRL_ADDR_LOGITS_OUT_BASE

#define AP_DONE_BIT        1

// ============================================================
// AXI BRAM Controller base
// SNN m_axi reads from this BRAM address
// ============================================================
#ifdef SDT
#define FEATURE_BRAM_BASE  XPAR_AXI_BRAM_CTRL_0_BASEADDR
#else
#define FEATURE_BRAM_BASE  XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR
#endif

static inline void ip_write(uint32_t off, uint32_t val)
{
    Xil_Out32(IP_BASE_ADDR + off, val);
}

static inline uint32_t ip_read(uint32_t off)
{
    return Xil_In32(IP_BASE_ADDR + off);
}

// ============================================================
// AXI GPIO timing pin
// ============================================================
// SDT: base address; legacy flow: device ID.
#ifndef GPIO_ID
#ifdef SDT
#define GPIO_ID       XPAR_AXI_GPIO_0_BASEADDR
#else
#define GPIO_ID       XPAR_AXI_GPIO_0_DEVICE_ID
#endif
#endif

#ifndef GPIO_CH
#define GPIO_CH       1
#endif

#ifndef GPIO_BIT
#define GPIO_BIT      0
#endif

static XGpio Gpio;

static int gpio_init(void)
{
    int st = XGpio_Initialize(&Gpio, GPIO_ID);

    if (st != XST_SUCCESS) {
        xil_printf("XGpio_Initialize failed: %d\r\n", st);
        return st;
    }

    XGpio_SetDataDirection(&Gpio, GPIO_CH, 0x00000000);
    XGpio_DiscreteWrite(&Gpio, GPIO_CH, 0x00000000);

    xil_printf("AXI GPIO init OK id=0x%08lx ch=%d bit=%d\r\n",
               (unsigned long)GPIO_ID, GPIO_CH, GPIO_BIT);

    return XST_SUCCESS;
}

static inline void timing_pin_high(void)
{
    XGpio_DiscreteWrite(&Gpio, GPIO_CH, (1u << GPIO_BIT));
}

static inline void timing_pin_low(void)
{
    XGpio_DiscreteWrite(&Gpio, GPIO_CH, 0x00000000);
}

// ============================================================
// UART helpers
// ============================================================
static void uart_set_baudrate(u32 baud)
{
    XUartPs Uart;
    XUartPs_Config *Cfg = XUartPs_LookupConfig(UART_ID);

    if (!Cfg) {
        return;
    }

    if (XUartPs_CfgInitialize(&Uart, Cfg, Cfg->BaseAddress) != XST_SUCCESS) {
        return;
    }

    XUartPs_SetOperMode(&Uart, XUARTPS_OPER_MODE_NORMAL);
    (void)XUartPs_SetBaudRate(&Uart, baud);
}

static uint8_t read_byte_blocking(void)
{
    return (uint8_t)inbyte();
}

static void wait_for_frame_header(void)
{
    while (1) {
        uint8_t b0 = read_byte_blocking();

        if (b0 == FRAME_START_0) {
            uint8_t b1 = read_byte_blocking();

            if (b1 == FRAME_START_1) {
                return;
            }
        }
    }
}

// ============================================================
// Feature buffer
// ============================================================
static int16_t feature_mem[N_FEATURES];

// ============================================================
// Receive 384 int16 features from Python
// Python sends little-endian int16 values
// ============================================================
static void receive_features_from_uart(void)
{
    for (int i = 0; i < N_FEATURES; i++) {
        uint8_t lo = read_byte_blocking();
        uint8_t hi = read_byte_blocking();

        uint16_t u = ((uint16_t)lo) | ((uint16_t)hi << 8);
        feature_mem[i] = (int16_t)u;
    }
}

// ============================================================
// Write feature_mem[] into AXI BRAM Controller memory
// Accelerator m_axi reads from this BRAM address
// ============================================================
static void write_features_to_bram(void)
{
    for (int i = 0; i < N_FEATURES; i++) {
        Xil_Out16(
            FEATURE_BRAM_BASE + (i * sizeof(int16_t)),
            (uint16_t)feature_mem[i]
        );
    }
}

// ============================================================
// Reply:
// 0xA5 <pred> <time_us u32 LE> <NUM_CLASSES logits i32 LE> 0x5A
// Total = 7 + 4 * NUM_CLASSES bytes (31 for 6 classes)
// ============================================================
#define NUM_CLASSES 6

static void uart_send_pred_time_logits_frame(
    uint8_t pred,
    uint32_t time_us,
    const int32_t logits[NUM_CLASSES]
)
{
    outbyte((char)0xA5);
    outbyte((char)pred);

    outbyte((char)((time_us >> 0) & 0xFF));
    outbyte((char)((time_us >> 8) & 0xFF));
    outbyte((char)((time_us >> 16) & 0xFF));
    outbyte((char)((time_us >> 24) & 0xFF));

    for (int i = 0; i < NUM_CLASSES; i++) {
        uint32_t v = (uint32_t)logits[i];

        outbyte((char)((v >> 0) & 0xFF));
        outbyte((char)((v >> 8) & 0xFF));
        outbyte((char)((v >> 16) & 0xFF));
        outbyte((char)((v >> 24) & 0xFF));
    }

    outbyte((char)0x5A);
}

// ============================================================
// MAIN
// ============================================================
int main(void)
{
    uart_set_baudrate(UART_BAUD);

    if (gpio_init() != XST_SUCCESS) {
        xil_printf("GPIO init failed. Check AXI GPIO in block design.\r\n");
        return 1;
    }

    timing_pin_low();

    xil_printf("\r\nAXI GPIO init OK ...\r\n");
    xil_printf("COUNTS_PER_SECOND=%lu\r\n", (unsigned long)(COUNTS_PER_SECOND));
    xil_printf("Framed UART(int16 features) -> SNN MLP\r\n");
    xil_printf("UART_ID=0x%08lx UART_BAUD=%lu\r\n",
               (unsigned long)UART_ID, (unsigned long)UART_BAUD);
    xil_printf("N_FEATURES=%d\r\n", N_FEATURES);
    xil_printf("RX_BYTES=%d\r\n", RX_BYTES);
    xil_printf("FEATURE_BRAM_BASE=0x%08lx\r\n",
               (unsigned long)FEATURE_BRAM_BASE);
    xil_printf("Reply: 0xA5 <pred> <time_us u32 LE> <%d logits i32 LE> 0x5A\r\n", NUM_CLASSES);

    while (1) {
        wait_for_frame_header();

        receive_features_from_uart();

        write_features_to_bram();

        // Tell SNN accelerator where feature memory starts.
        // Since SNN m_axi is connected to AXI BRAM Controller,
        // this is the AXI BRAM base address.
        ip_write(REG_FEATURE_MEM_L, FEATURE_BRAM_BASE);
        ip_write(REG_FEATURE_MEM_H, 0u);

        XTime t0;
        XTime t1;

        XTime_GetTime(&t0);

        timing_pin_high();

        ip_write(REG_AP_CTRL, 0x01);

        while (((ip_read(REG_AP_CTRL) >> AP_DONE_BIT) & 0x1u) == 0u) {
            // wait for ap_done
        }

        timing_pin_low();

        XTime_GetTime(&t1);

        uint64_t dt_ticks = (uint64_t)(t1 - t0);

        uint32_t time_us =
            (uint32_t)((dt_ticks * 1000000ULL) /
                       (uint64_t)(COUNTS_PER_SECOND));

        uint32_t pred = ip_read(REG_PRED_OUT);

        int32_t logits[NUM_CLASSES];

        for (int i = 0; i < NUM_CLASSES; i++) {
            logits[i] = (int32_t)ip_read(REG_LOGITS_BASE + 4 * i);
        }

        uart_send_pred_time_logits_frame(
            (uint8_t)(pred & 0xFFu),
            time_us,
            logits
        );
    }

    return 0;
}
