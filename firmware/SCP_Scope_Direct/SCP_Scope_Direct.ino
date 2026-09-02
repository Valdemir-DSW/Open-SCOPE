#include <Arduino.h>
#include <string.h>
#include <stdlib.h>

// ============================================================================
// SCP SCOPE DIRECT STREAM — PROFESSIONAL v7
// STM32F103 + ATmega32U4 / Arduino Leonardo
//
// Acquisition policy:
//   ADC -> tiny ping-pong transport block -> USB/Serial -> Windows
//
// There is NO oscilloscope history, trigger, pre-trigger or triggered capture
// stored on the MCU. Triggering and record history belong to the PC client.
// Small blocks below exist only to decouple ADC timing from USB packet writes.
// ============================================================================

// ============================================================================
// USER CONFIGURATION
// ============================================================================

#define SCOPE_TARGET_AUTO              0
#define SCOPE_TARGET_STM32F103         1
#define SCOPE_TARGET_LEONARDO          2

#define SCOPE_TARGET                   SCOPE_TARGET_AUTO

// 0 = target default. STM32: 2, 3 or 4. Leonardo: 2.
#define SCOPE_CHANNEL_COUNT            0

// 0 = profile default at boot.
#define SCOPE_SAMPLE_RATE_HZ           0UL

// Transport block only; NOT capture/history depth. 0 = target default.
#define SCOPE_STREAM_CHUNK_FRAMES      0U

// Profile rates. 0 = target defaults.
#define SCOPE_HIGH_RATE_HZ             0UL
#define SCOPE_STANDARD_RATE_HZ           0UL
#define SCOPE_LONG_RATE_HZ             0UL

#define SCOPE_SERIAL_BAUD              2000000UL

// Leonardo ADC clock: 16 MHz / 16 = 1 MHz.
#define SCOPE_LEONARDO_ADC_PRESCALER   16UL

// STM32 inputs.
#define SCOPE_STM32_CH1_PIN            PA0
#define SCOPE_STM32_CH1_ADC_CHANNEL    0U
#define SCOPE_STM32_CH2_PIN            PA1
#define SCOPE_STM32_CH2_ADC_CHANNEL    1U
#define SCOPE_STM32_CH3_PIN            PA2
#define SCOPE_STM32_CH3_ADC_CHANNEL    2U
#define SCOPE_STM32_CH4_PIN            PA3
#define SCOPE_STM32_CH4_ADC_CHANNEL    3U

// Leonardo A0=ADC7, A1=ADC6.
#define SCOPE_LEONARDO_CH1_PIN         A0
#define SCOPE_LEONARDO_CH1_ADC_CHANNEL 7U
#define SCOPE_LEONARDO_CH2_PIN         A1
#define SCOPE_LEONARDO_CH2_ADC_CHANNEL 6U

// ============================================================================
// TARGET DETECTION / PROFILE
// ============================================================================

#if SCOPE_TARGET == SCOPE_TARGET_AUTO
    #if defined(ARDUINO_ARCH_STM32) && defined(STM32F1xx)
        #define SCOPE_ACTIVE_TARGET SCOPE_TARGET_STM32F103
    #elif defined(__AVR_ATmega32U4__)
        #define SCOPE_ACTIVE_TARGET SCOPE_TARGET_LEONARDO
    #else
        #error "Select STM32F103 (STM32duino) or ATmega32U4/Arduino Leonardo."
    #endif
#else
    #define SCOPE_ACTIVE_TARGET SCOPE_TARGET
#endif

#if SCOPE_ACTIVE_TARGET == SCOPE_TARGET_STM32F103
    #if !defined(ARDUINO_ARCH_STM32) || !defined(STM32F1xx)
        #error "STM32F103 backend requires STM32duino STM32F1."
    #endif

    #if SCOPE_CHANNEL_COUNT == 0
        #define SCOPE_ACTIVE_CHANNELS 3U
    #else
        #define SCOPE_ACTIVE_CHANNELS SCOPE_CHANNEL_COUNT
    #endif

    #if (SCOPE_ACTIVE_CHANNELS != 2U) && (SCOPE_ACTIVE_CHANNELS != 3U) && (SCOPE_ACTIVE_CHANNELS != 4U)
        #error "STM32F103 supports 2, 3 or 4 channels."
    #endif

    #define SCOPE_ADC_BITS 12U
    #define SCOPE_ADC_MAX 4095U

    #if SCOPE_ACTIVE_CHANNELS == 4U
        // Four channels use two simultaneous ADC ranks. Rates are kept below
        // the 3-channel profile so the USB/serial payload stays near the same
        // transport bandwidth as the proven 3-channel configuration.
        #define SCOPE_ADC_RANKS                2U
        #define SCOPE_PROFILE_MAX_RATE         120000UL
        #define SCOPE_PROFILE_HIGH_RATE        110000UL
        #define SCOPE_PROFILE_STANDARD_RATE      60000UL
        #define SCOPE_PROFILE_LONG_RATE        15000UL
        #define SCOPE_PROFILE_CHUNK            256U
    #elif SCOPE_ACTIVE_CHANNELS == 3U
        #define SCOPE_ADC_RANKS                2U
        #define SCOPE_PROFILE_MAX_RATE         160000UL
        #define SCOPE_PROFILE_HIGH_RATE        150000UL
        #define SCOPE_PROFILE_STANDARD_RATE      80000UL
        #define SCOPE_PROFILE_LONG_RATE        20000UL
        #define SCOPE_PROFILE_CHUNK            256U
    #else
        #define SCOPE_ADC_RANKS                1U
        #define SCOPE_PROFILE_MAX_RATE         240000UL
        #define SCOPE_PROFILE_HIGH_RATE        220000UL
        #define SCOPE_PROFILE_STANDARD_RATE      120000UL
        #define SCOPE_PROFILE_LONG_RATE        30000UL
        #define SCOPE_PROFILE_CHUNK            256U
    #endif

#elif SCOPE_ACTIVE_TARGET == SCOPE_TARGET_LEONARDO
    #if !defined(__AVR_ATmega32U4__)
        #error "Leonardo backend requires ATmega32U4."
    #endif

    #if SCOPE_CHANNEL_COUNT == 0
        #define SCOPE_ACTIVE_CHANNELS 2U
    #else
        #define SCOPE_ACTIVE_CHANNELS SCOPE_CHANNEL_COUNT
    #endif

    #if SCOPE_ACTIVE_CHANNELS != 2U
        #error "Leonardo direct backend supports exactly 2 channels."
    #endif

    #if (SCOPE_LEONARDO_ADC_PRESCALER != 16UL) && \
        (SCOPE_LEONARDO_ADC_PRESCALER != 32UL) && \
        (SCOPE_LEONARDO_ADC_PRESCALER != 64UL) && \
        (SCOPE_LEONARDO_ADC_PRESCALER != 128UL)
        #error "SCOPE_LEONARDO_ADC_PRESCALER must be 16, 32, 64 or 128."
    #endif

    #define SCOPE_ADC_BITS 10U
    #define SCOPE_ADC_MAX 1023U
    #define SCOPE_PROFILE_MAX_RATE         32000UL
    #define SCOPE_PROFILE_HIGH_RATE        30000UL
    #define SCOPE_PROFILE_STANDARD_RATE      25000UL
    #define SCOPE_PROFILE_LONG_RATE        8000UL
    #define SCOPE_PROFILE_CHUNK             64U
#else
    #error "Invalid SCOPE_TARGET."
#endif

#if SCOPE_STREAM_CHUNK_FRAMES == 0U
    #define SCOPE_ACTIVE_CHUNK SCOPE_PROFILE_CHUNK
#else
    #define SCOPE_ACTIVE_CHUNK SCOPE_STREAM_CHUNK_FRAMES
#endif

#if SCOPE_ACTIVE_CHUNK < 8U || SCOPE_ACTIVE_CHUNK > 256U
    #error "SCOPE_STREAM_CHUNK_FRAMES must be 8..256."
#endif

#if SCOPE_HIGH_RATE_HZ == 0UL
    #define SCOPE_ACTIVE_HIGH_RATE SCOPE_PROFILE_HIGH_RATE
#else
    #define SCOPE_ACTIVE_HIGH_RATE SCOPE_HIGH_RATE_HZ
#endif

#if SCOPE_STANDARD_RATE_HZ == 0UL
    #define SCOPE_ACTIVE_STANDARD_RATE SCOPE_PROFILE_STANDARD_RATE
#else
    #define SCOPE_ACTIVE_STANDARD_RATE SCOPE_STANDARD_RATE_HZ
#endif

#if SCOPE_LONG_RATE_HZ == 0UL
    #define SCOPE_ACTIVE_LONG_RATE SCOPE_PROFILE_LONG_RATE
#else
    #define SCOPE_ACTIVE_LONG_RATE SCOPE_LONG_RATE_HZ
#endif

#if SCOPE_SAMPLE_RATE_HZ == 0UL
    #define SCOPE_INITIAL_RATE SCOPE_ACTIVE_STANDARD_RATE
#else
    #define SCOPE_INITIAL_RATE SCOPE_SAMPLE_RATE_HZ
#endif

// ============================================================================
// SCP1 v5 DIRECT STREAM PROTOCOL
// ============================================================================

static const uint32_t SCOPE_MAGIC = 0x31504353UL; // SCP1
static const uint16_t SCOPE_PROTOCOL_VERSION = 5U;
static const uint8_t SCOPE_PACKET_STREAM = 1U;
static const uint8_t SCOPE_FLAG_DISCONTINUITY = 0x01U;
static const uint8_t SCOPE_FLAG_DIRECT = 0x02U;

struct __attribute__((packed)) ScopeCaptureHeader
{
    uint32_t magic;
    uint16_t version;
    uint16_t headerSize;
    uint32_t sampleRate;
    uint16_t frameCount;
    uint16_t preTriggerFrames;
    uint16_t triggerLevel;
    uint8_t channelCount;
    uint8_t triggerChannel;
    uint8_t triggerEdge;
    uint8_t adcBits;
    uint32_t payloadBytes;
    uint32_t crc32;
};

struct __attribute__((packed)) ScopePacketExtension
{
    uint8_t packetType;
    uint8_t flags;
    uint16_t reserved;
    uint32_t sequence;
};

static_assert(sizeof(ScopeCaptureHeader) == 30U, "SCP1 header must be 30 bytes");
static_assert(sizeof(ScopePacketExtension) == 8U, "SCP1 extension must be 8 bytes");

#define SCOPE_PAYLOAD_BYTES ((uint16_t)(SCOPE_ACTIVE_CHUNK * SCOPE_ACTIVE_CHANNELS * 2U))

static uint8_t scopeTxPayload[SCOPE_PAYLOAD_BYTES];
static uint32_t scopePacketSequence = 0UL;
static uint32_t scopeRequestedRate = SCOPE_INITIAL_RATE;
static uint32_t scopeActualRate = 0UL;
static bool scopeRunning = true;
static volatile uint8_t scopePendingFlags = SCOPE_FLAG_DIRECT;

// Small 16-entry CRC32 table: substantially cheaper than bit-at-a-time CRC on AVR.
static const uint32_t scopeCrcNibble[16] = {
    0x00000000UL, 0x1DB71064UL, 0x3B6E20C8UL, 0x26D930ACUL,
    0x76DC4190UL, 0x6B6B51F4UL, 0x4DB26158UL, 0x5005713CUL,
    0xEDB88320UL, 0xF00F9344UL, 0xD6D6A3E8UL, 0xCB61B38CUL,
    0x9B64C2B0UL, 0x86D3D2D4UL, 0xA00AE278UL, 0xBDBDF21CUL
};

static uint32_t scopeCrc32(const uint8_t *data, uint16_t length)
{
    uint32_t crc = 0xFFFFFFFFUL;
    while (length--)
    {
        crc ^= *data++;
        crc = (crc >> 4U) ^ scopeCrcNibble[crc & 0x0FUL];
        crc = (crc >> 4U) ^ scopeCrcNibble[crc & 0x0FUL];
    }
    return ~crc;
}

static void scopeSendPacket(uint8_t flags)
{
    ScopeCaptureHeader h;
    h.magic = SCOPE_MAGIC;
    h.version = SCOPE_PROTOCOL_VERSION;
    h.headerSize = sizeof(ScopeCaptureHeader) + sizeof(ScopePacketExtension);
    h.sampleRate = scopeActualRate;
    h.frameCount = SCOPE_ACTIVE_CHUNK;
    h.preTriggerFrames = 0U;          // Windows owns trigger/pre-trigger.
    h.triggerLevel = (SCOPE_ADC_MAX + 1U) / 2U; // informational only.
    h.channelCount = SCOPE_ACTIVE_CHANNELS;
    h.triggerChannel = 1U;            // informational only.
    h.triggerEdge = 1U;               // informational only.
    h.adcBits = SCOPE_ADC_BITS;
    h.payloadBytes = SCOPE_PAYLOAD_BYTES;
    h.crc32 = scopeCrc32(scopeTxPayload, SCOPE_PAYLOAD_BYTES);

    ScopePacketExtension ext;
    ext.packetType = SCOPE_PACKET_STREAM;
    ext.flags = (uint8_t)(flags | SCOPE_FLAG_DIRECT);
    ext.reserved = 0U;
    ext.sequence = scopePacketSequence++;

    Serial.write((const uint8_t *)&h, sizeof(h));
    Serial.write((const uint8_t *)&ext, sizeof(ext));
    Serial.write(scopeTxPayload, SCOPE_PAYLOAD_BYTES);
}

// ============================================================================
// HARDWARE API
// ============================================================================

static void scopeHardwareInit();
static void scopeHardwareStart();
static void scopeHardwareStop();
static uint32_t scopeHardwareSetRate(uint32_t requested);
static bool scopeBuildNextPayload(uint8_t *flags);

// ============================================================================
// STM32F103: ADC1+ADC2 -> tiny two-half circular DMA transport buffer
// ============================================================================

#if SCOPE_ACTIVE_TARGET == SCOPE_TARGET_STM32F103

#define SCOPE_STM_WORDS_PER_BLOCK (SCOPE_ACTIVE_CHUNK * SCOPE_ADC_RANKS)
#define SCOPE_STM_DMA_WORDS       (SCOPE_STM_WORDS_PER_BLOCK * 2U)
static_assert(SCOPE_STM_DMA_WORDS <= 65535U, "DMA CNDTR overflow");

alignas(4) static volatile uint32_t scopeStmDma[SCOPE_STM_DMA_WORDS];
static volatile uint32_t scopeStmGeneration[2] = {0UL, 0UL};
static uint32_t scopeStmConsumed[2] = {0UL, 0UL};
static uint8_t scopeStmNextBlock = 0U;

static inline uint16_t stmLow(uint32_t v)  { return (uint16_t)(v & 0x0FFFU); }
static inline uint16_t stmHigh(uint32_t v) { return (uint16_t)((v >> 16U) & 0x0FFFU); }

static void stmSetSampleTime(ADC_TypeDef *adc, uint8_t channel)
{
    if (channel <= 9U)
        adc->SMPR2 &= ~(7UL << ((uint32_t)channel * 3U));
    else
        adc->SMPR1 &= ~(7UL << ((uint32_t)(channel - 10U) * 3U));
}

static void stmSetRank(ADC_TypeDef *adc, uint8_t rank, uint8_t channel)
{
    const uint32_t shift = (uint32_t)(rank - 1U) * 5U;
    adc->SQR3 = (adc->SQR3 & ~(0x1FUL << shift)) | ((uint32_t)channel << shift);
}

static void stmSetLength(ADC_TypeDef *adc, uint8_t length)
{
    adc->SQR1 = (adc->SQR1 & ~(0xFUL << 20U)) | ((uint32_t)(length - 1U) << 20U);
}

static void stmCalibrate(ADC_TypeDef *adc)
{
    adc->CR2 |= ADC_CR2_ADON;
    delayMicroseconds(2);
    adc->CR2 |= ADC_CR2_RSTCAL;
    while (adc->CR2 & ADC_CR2_RSTCAL) {}
    adc->CR2 |= ADC_CR2_CAL;
    while (adc->CR2 & ADC_CR2_CAL) {}
}

static uint32_t stmTimerClock()
{
    uint32_t clock = HAL_RCC_GetPCLK1Freq();
    if ((RCC->CFGR & RCC_CFGR_PPRE1) != 0U)
        clock *= 2U;
    return clock;
}

static uint32_t scopeHardwareSetRate(uint32_t requested)
{
    if (requested < 10UL) requested = 10UL;
    if (requested > SCOPE_PROFILE_MAX_RATE) requested = SCOPE_PROFILE_MAX_RATE;

    const uint32_t timerClock = stmTimerClock();
    uint32_t prescaler = (uint32_t)(((uint64_t)timerClock + ((uint64_t)requested * 65536ULL) - 1ULL) /
                                    ((uint64_t)requested * 65536ULL));
    if (prescaler < 1U) prescaler = 1U;
    if (prescaler > 65536U) prescaler = 65536U;

    const uint64_t denom = (uint64_t)requested * prescaler;
    uint32_t ticks = (uint32_t)(((uint64_t)timerClock + denom / 2ULL) / denom);
    if (ticks < 1U) ticks = 1U;
    if (ticks > 65536U) ticks = 65536U;

    TIM3->PSC = prescaler - 1U;
    TIM3->ARR = ticks - 1U;
    TIM3->CNT = 0U;
    TIM3->EGR = TIM_EGR_UG;
    scopeActualRate = timerClock / prescaler / ticks;
    return scopeActualRate;
}

static void scopeHardwareInit()
{
    RCC->APB2ENR |= RCC_APB2ENR_ADC1EN | RCC_APB2ENR_ADC2EN;
    RCC->AHBENR  |= RCC_AHBENR_DMA1EN;
    RCC->APB1ENR |= RCC_APB1ENR_TIM3EN;

    pinMode(SCOPE_STM32_CH1_PIN, INPUT_ANALOG);
    pinMode(SCOPE_STM32_CH2_PIN, INPUT_ANALOG);
#if SCOPE_ACTIVE_CHANNELS >= 3U
    pinMode(SCOPE_STM32_CH3_PIN, INPUT_ANALOG);
#endif
#if SCOPE_ACTIVE_CHANNELS >= 4U
    pinMode(SCOPE_STM32_CH4_PIN, INPUT_ANALOG);
#endif

    // 72 MHz / 6 = 12 MHz ADC clock on the normal F103 clock tree.
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_ADCPRE) | RCC_CFGR_ADCPRE_1;

    ADC1->CR1 = ADC1->CR2 = 0U;
    ADC2->CR1 = ADC2->CR2 = 0U;
    ADC1->SQR1 = ADC1->SQR2 = ADC1->SQR3 = 0U;
    ADC2->SQR1 = ADC2->SQR2 = ADC2->SQR3 = 0U;

#if SCOPE_ACTIVE_CHANNELS == 2U
    stmSetLength(ADC1, 1U);
    stmSetLength(ADC2, 1U);
    stmSetRank(ADC1, 1U, SCOPE_STM32_CH1_ADC_CHANNEL);
    stmSetRank(ADC2, 1U, SCOPE_STM32_CH2_ADC_CHANNEL);
    stmSetSampleTime(ADC1, SCOPE_STM32_CH1_ADC_CHANNEL);
    stmSetSampleTime(ADC2, SCOPE_STM32_CH2_ADC_CHANNEL);
#else
    // One timer event = one logical 3/4-channel frame across two ADC ranks.
    // Rank1: ADC1 CH1 + ADC2 CH3; Rank2: ADC1 CH2 + ADC2 CH4 (or CH3 spare).
    stmSetLength(ADC1, 2U);
    stmSetLength(ADC2, 2U);
    ADC1->CR1 |= ADC_CR1_SCAN;
    ADC2->CR1 |= ADC_CR1_SCAN;
    stmSetRank(ADC1, 1U, SCOPE_STM32_CH1_ADC_CHANNEL);
    stmSetRank(ADC1, 2U, SCOPE_STM32_CH2_ADC_CHANNEL);
    stmSetRank(ADC2, 1U, SCOPE_STM32_CH3_ADC_CHANNEL);
#if SCOPE_ACTIVE_CHANNELS == 4U
    stmSetRank(ADC2, 2U, SCOPE_STM32_CH4_ADC_CHANNEL);
#else
    stmSetRank(ADC2, 2U, SCOPE_STM32_CH3_ADC_CHANNEL);
#endif
    stmSetSampleTime(ADC1, SCOPE_STM32_CH1_ADC_CHANNEL);
    stmSetSampleTime(ADC1, SCOPE_STM32_CH2_ADC_CHANNEL);
    stmSetSampleTime(ADC2, SCOPE_STM32_CH3_ADC_CHANNEL);
#if SCOPE_ACTIVE_CHANNELS == 4U
    stmSetSampleTime(ADC2, SCOPE_STM32_CH4_ADC_CHANNEL);
#endif
#endif

    // Dual regular simultaneous = 0110.
    ADC1->CR1 &= ~ADC_CR1_DUALMOD;
    ADC1->CR1 |= ADC_CR1_DUALMOD_2 | ADC_CR1_DUALMOD_1;
    ADC1->CR2 = ADC_CR2_DMA | ADC_CR2_EXTTRIG | ADC_CR2_EXTSEL_2; // TIM3 TRGO
    ADC2->CR2 = ADC_CR2_EXTTRIG | ADC_CR2_EXTSEL_2 | ADC_CR2_EXTSEL_1 | ADC_CR2_EXTSEL_0;

    stmCalibrate(ADC1);
    stmCalibrate(ADC2);
    ADC1->CR2 |= ADC_CR2_ADON;
    ADC2->CR2 |= ADC_CR2_ADON;

    DMA1_Channel1->CCR = 0U;
    DMA1_Channel1->CPAR = (uint32_t)&ADC1->DR;
    DMA1_Channel1->CMAR = (uint32_t)scopeStmDma;
    DMA1_Channel1->CNDTR = SCOPE_STM_DMA_WORDS;
    DMA1_Channel1->CCR = DMA_CCR_MINC | DMA_CCR_CIRC |
                         DMA_CCR_PSIZE_1 | DMA_CCR_MSIZE_1 |
                         DMA_CCR_PL_1 | DMA_CCR_PL_0 |
                         DMA_CCR_HTIE | DMA_CCR_TCIE;
    DMA1->IFCR = DMA_IFCR_CGIF1;

    NVIC_SetPriority(DMA1_Channel1_IRQn, 1U);
    NVIC_EnableIRQ(DMA1_Channel1_IRQn);

    TIM3->CR1 = TIM_CR1_ARPE;
    TIM3->CR2 = TIM_CR2_MMS_1; // update = TRGO
    TIM3->SMCR = 0U;
    TIM3->DIER = 0U;

    scopeHardwareSetRate(scopeRequestedRate);
}

extern "C" void DMA1_Channel1_IRQHandler(void)
{
    const uint32_t isr = DMA1->ISR;
    if (isr & DMA_ISR_HTIF1)
    {
        DMA1->IFCR = DMA_IFCR_CHTIF1;
        ++scopeStmGeneration[0];
    }
    if (isr & DMA_ISR_TCIF1)
    {
        DMA1->IFCR = DMA_IFCR_CTCIF1;
        ++scopeStmGeneration[1];
    }
    if (isr & DMA_ISR_TEIF1)
    {
        DMA1->IFCR = DMA_IFCR_CTEIF1;
        scopePendingFlags |= SCOPE_FLAG_DISCONTINUITY;
    }
}

static void scopeHardwareStop()
{
    TIM3->CR1 &= ~TIM_CR1_CEN;
    DMA1_Channel1->CCR &= ~DMA_CCR_EN;
}

static void scopeHardwareStart()
{
    scopeHardwareStop();
    scopeStmGeneration[0] = scopeStmGeneration[1] = 0UL;
    scopeStmConsumed[0] = scopeStmConsumed[1] = 0UL;
    scopeStmNextBlock = 0U;
    DMA1->IFCR = DMA_IFCR_CGIF1;
    DMA1_Channel1->CNDTR = SCOPE_STM_DMA_WORDS;
    DMA1_Channel1->CCR |= DMA_CCR_EN;
    TIM3->CNT = 0U;
    TIM3->SR = 0U;
    TIM3->CR1 |= TIM_CR1_CEN;
}

static bool scopeBuildNextPayload(uint8_t *flags)
{
    uint8_t block = scopeStmNextBlock;
    uint32_t generation = scopeStmGeneration[block];

    if (generation == scopeStmConsumed[block])
    {
        // If the other half is already newer, the expected half was lost.
        const uint8_t other = block ^ 1U;
        if (scopeStmGeneration[other] == scopeStmConsumed[other])
            return false;
        *flags |= SCOPE_FLAG_DISCONTINUITY;
        block = other;
        generation = scopeStmGeneration[block];
    }

    if ((generation - scopeStmConsumed[block]) > 1UL)
        *flags |= SCOPE_FLAG_DISCONTINUITY;

    const uint32_t base = (uint32_t)block * SCOPE_STM_WORDS_PER_BLOCK;
    uint16_t out = 0U;

    for (uint16_t i = 0U; i < SCOPE_ACTIVE_CHUNK; ++i)
    {
#if SCOPE_ACTIVE_CHANNELS == 2U
        const uint32_t p = scopeStmDma[base + i];
        const uint16_t samples[2] = { stmLow(p), stmHigh(p) };
        for (uint8_t ch = 0U; ch < 2U; ++ch)
        {
            scopeTxPayload[out++] = (uint8_t)(samples[ch] & 0xFFU);
            scopeTxPayload[out++] = (uint8_t)(samples[ch] >> 8U);
        }
#elif SCOPE_ACTIVE_CHANNELS == 3U
        const uint32_t p0 = scopeStmDma[base + (uint32_t)i * 2U];
        const uint32_t p1 = scopeStmDma[base + (uint32_t)i * 2U + 1U];
        const uint16_t samples[3] = { stmLow(p0), stmLow(p1), stmHigh(p0) };
        for (uint8_t ch = 0U; ch < 3U; ++ch)
        {
            scopeTxPayload[out++] = (uint8_t)(samples[ch] & 0xFFU);
            scopeTxPayload[out++] = (uint8_t)(samples[ch] >> 8U);
        }
#else
        const uint32_t p0 = scopeStmDma[base + (uint32_t)i * 2U];
        const uint32_t p1 = scopeStmDma[base + (uint32_t)i * 2U + 1U];
        const uint16_t samples[4] = { stmLow(p0), stmLow(p1), stmHigh(p0), stmHigh(p1) };
        for (uint8_t ch = 0U; ch < 4U; ++ch)
        {
            scopeTxPayload[out++] = (uint8_t)(samples[ch] & 0xFFU);
            scopeTxPayload[out++] = (uint8_t)(samples[ch] >> 8U);
        }
#endif
    }

    // DMA may have lapped this half while it was being copied.
    const uint32_t after = scopeStmGeneration[block];
    if (after != generation)
    {
        *flags |= SCOPE_FLAG_DISCONTINUITY;
        scopeStmConsumed[block] = after - 1UL;
        scopeStmNextBlock = block;
        return false;
    }

    scopeStmConsumed[block] = generation;
    scopeStmNextBlock = block ^ 1U;
    return true;
}

// ============================================================================
// ATmega32U4 / Leonardo: timer-paced ADC -> two tiny transport blocks
// ============================================================================

#else

#include <avr/interrupt.h>

struct ScopeAvrFrame { uint16_t ch1; uint16_t ch2; };
static volatile ScopeAvrFrame scopeAvrBlocks[2][SCOPE_ACTIVE_CHUNK];
static volatile uint8_t scopeAvrGeneration[2] = {0U, 0U};
static uint8_t scopeAvrConsumed[2] = {0U, 0U};
static volatile uint8_t scopeAvrWriteBlock = 0U;
static volatile uint16_t scopeAvrWriteIndex = 0U;
static volatile uint8_t scopeAvrCurrentChannel = 0U;
static volatile uint16_t scopeAvrPendingCh1 = 0U;
static uint8_t scopeAvrNextBlock = 0U;
static uint8_t scopeAvrTimerCsBits = _BV(CS10);

static uint8_t avrAdcPrescalerBits()
{
#if SCOPE_LEONARDO_ADC_PRESCALER == 16UL
    return _BV(ADPS2);
#elif SCOPE_LEONARDO_ADC_PRESCALER == 32UL
    return _BV(ADPS2) | _BV(ADPS0);
#elif SCOPE_LEONARDO_ADC_PRESCALER == 64UL
    return _BV(ADPS2) | _BV(ADPS1);
#else
    return _BV(ADPS2) | _BV(ADPS1) | _BV(ADPS0);
#endif
}

static inline void avrSelectAdcChannel(uint8_t channel)
{
    ADMUX = (uint8_t)((ADMUX & 0xE0U) | (channel & 0x07U));
    if (channel & 0x08U) ADCSRB |= _BV(MUX5);
    else ADCSRB &= (uint8_t)~_BV(MUX5);
}

struct AvrTimerSetting { uint16_t divider; uint8_t bits; };
static const AvrTimerSetting avrTimers[] = {
    {1U, _BV(CS10)},
    {8U, _BV(CS11)},
    {64U, (uint8_t)(_BV(CS11) | _BV(CS10))},
    {256U, _BV(CS12)},
    {1024U, (uint8_t)(_BV(CS12) | _BV(CS10))}
};

static uint32_t scopeHardwareSetRate(uint32_t requested)
{
    if (requested < 10UL) requested = 10UL;
    if (requested > SCOPE_PROFILE_MAX_RATE) requested = SCOPE_PROFILE_MAX_RATE;

    const uint32_t conversionRate = requested * 2UL;
    uint32_t actual = 0UL;
    uint16_t top = 0xFFFFU;
    uint8_t cs = avrTimers[4].bits;

    for (uint8_t i = 0U; i < sizeof(avrTimers) / sizeof(avrTimers[0]); ++i)
    {
        const uint32_t div = avrTimers[i].divider;
        uint32_t ticks = (F_CPU + (conversionRate * div) / 2UL) / (conversionRate * div);
        if (ticks < 2UL) ticks = 2UL;
        if (ticks > 65536UL) continue;
        actual = (F_CPU / div / ticks) / 2UL;
        top = (uint16_t)(ticks - 1UL);
        cs = avrTimers[i].bits;
        break;
    }

    scopeAvrTimerCsBits = cs;
    TCCR1B &= (uint8_t)~(_BV(CS12) | _BV(CS11) | _BV(CS10));
    TCNT1 = 0U;
    OCR1A = top;
    OCR1B = top > 2U ? 1U : 0U;
    scopeActualRate = actual;
    return scopeActualRate;
}

static void scopeHardwareInit()
{
    pinMode(SCOPE_LEONARDO_CH1_PIN, INPUT);
    pinMode(SCOPE_LEONARDO_CH2_PIN, INPUT);

    if (SCOPE_LEONARDO_CH1_ADC_CHANNEL <= 7U) DIDR0 |= (uint8_t)_BV(SCOPE_LEONARDO_CH1_ADC_CHANNEL);
    if (SCOPE_LEONARDO_CH2_ADC_CHANNEL <= 7U) DIDR0 |= (uint8_t)_BV(SCOPE_LEONARDO_CH2_ADC_CHANNEL);

    TCCR1A = 0U;
    TCCR1B = _BV(WGM12);
    TIMSK1 = _BV(OCIE1B);

    ADMUX = _BV(REFS0); // AVcc reference
    ADCSRB = _BV(ADTS2) | _BV(ADTS0); // Timer1 Compare Match B
#ifdef ADHSM
    ADCSRB |= _BV(ADHSM);
#endif
    ADCSRA = _BV(ADEN) | _BV(ADATE) | _BV(ADIE) | avrAdcPrescalerBits();
    avrSelectAdcChannel(SCOPE_LEONARDO_CH1_ADC_CHANNEL);
    scopeHardwareSetRate(scopeRequestedRate);
}

static void scopeHardwareStop()
{
    TCCR1B &= (uint8_t)~(_BV(CS12) | _BV(CS11) | _BV(CS10));
    ADCSRA &= (uint8_t)~(_BV(ADIE) | _BV(ADATE));
}

static void scopeHardwareStart()
{
    scopeHardwareStop();
    scopeAvrGeneration[0] = scopeAvrGeneration[1] = 0U;
    scopeAvrConsumed[0] = scopeAvrConsumed[1] = 0U;
    scopeAvrWriteBlock = 0U;
    scopeAvrWriteIndex = 0U;
    scopeAvrCurrentChannel = 0U;
    scopeAvrPendingCh1 = 0U;
    scopeAvrNextBlock = 0U;
    avrSelectAdcChannel(SCOPE_LEONARDO_CH1_ADC_CHANNEL);

    ADCSRA |= _BV(ADEN) | _BV(ADATE) | _BV(ADIE);
    ADCSRA |= _BV(ADIF);
    TIFR1 = _BV(OCF1B);
    TCNT1 = 0U;
    TCCR1B = (uint8_t)(_BV(WGM12) | scopeAvrTimerCsBits);
    ADCSRA |= _BV(ADSC);
}

ISR(TIMER1_COMPB_vect, ISR_NAKED)
{
    asm volatile("reti");
}

ISR(ADC_vect)
{
    const uint16_t value = ADC;

    if (scopeAvrCurrentChannel == 0U)
    {
        scopeAvrPendingCh1 = value;
        scopeAvrCurrentChannel = 1U;
        avrSelectAdcChannel(SCOPE_LEONARDO_CH2_ADC_CHANNEL);
        return;
    }

    const uint8_t block = scopeAvrWriteBlock;
    const uint16_t index = scopeAvrWriteIndex;
    scopeAvrBlocks[block][index].ch1 = scopeAvrPendingCh1;
    scopeAvrBlocks[block][index].ch2 = value;

    uint16_t next = index + 1U;
    if (next >= SCOPE_ACTIVE_CHUNK)
    {
        ++scopeAvrGeneration[block];
        scopeAvrWriteBlock = block ^ 1U;
        next = 0U;
    }
    scopeAvrWriteIndex = next;
    scopeAvrCurrentChannel = 0U;
    avrSelectAdcChannel(SCOPE_LEONARDO_CH1_ADC_CHANNEL);
}

static bool scopeBuildNextPayload(uint8_t *flags)
{
    uint8_t block = scopeAvrNextBlock;
    uint8_t generation = scopeAvrGeneration[block];

    if (generation == scopeAvrConsumed[block])
    {
        const uint8_t other = block ^ 1U;
        if (scopeAvrGeneration[other] == scopeAvrConsumed[other])
            return false;
        *flags |= SCOPE_FLAG_DISCONTINUITY;
        block = other;
        generation = scopeAvrGeneration[block];
    }

    if ((uint8_t)(generation - scopeAvrConsumed[block]) > 1U)
        *flags |= SCOPE_FLAG_DISCONTINUITY;

    uint16_t out = 0U;
    for (uint16_t i = 0U; i < SCOPE_ACTIVE_CHUNK; ++i)
    {
        const uint16_t a = scopeAvrBlocks[block][i].ch1;
        const uint16_t b = scopeAvrBlocks[block][i].ch2;
        scopeTxPayload[out++] = (uint8_t)(a & 0xFFU);
        scopeTxPayload[out++] = (uint8_t)(a >> 8U);
        scopeTxPayload[out++] = (uint8_t)(b & 0xFFU);
        scopeTxPayload[out++] = (uint8_t)(b >> 8U);
    }

    if (scopeAvrGeneration[block] != generation)
    {
        *flags |= SCOPE_FLAG_DISCONTINUITY;
        scopeAvrConsumed[block] = (uint8_t)(scopeAvrGeneration[block] - 1U);
        scopeAvrNextBlock = block;
        return false;
    }

    scopeAvrConsumed[block] = generation;
    scopeAvrNextBlock = block ^ 1U;
    return true;
}

#endif

// ============================================================================
// DIRECT STREAM SERVICE
// ============================================================================

static void scopeRestart()
{
    scopeHardwareStop();
    scopeHardwareSetRate(scopeRequestedRate);
    scopePendingFlags |= SCOPE_FLAG_DISCONTINUITY;
    if (scopeRunning)
        scopeHardwareStart();
}

static void scopeStreamService()
{
    if (!scopeRunning)
        return;

    uint8_t flags = scopePendingFlags;
    if (!scopeBuildNextPayload(&flags))
        return;

    scopePendingFlags = SCOPE_FLAG_DIRECT;
    scopeSendPacket(flags);
}

// ============================================================================
// COMMANDS
// Trigger commands are intentionally accepted but ignored: Windows owns them.
// ============================================================================

static char scopeCommandBuffer[72];
static uint8_t scopeCommandLength = 0U;
static bool scopeCommandOverflow = false;

static void scopeSetRequestedRate(uint32_t rate)
{
    if (rate < 10UL) rate = 10UL;
    if (rate > SCOPE_PROFILE_MAX_RATE) rate = SCOPE_PROFILE_MAX_RATE;
    if (rate == scopeRequestedRate && scopeActualRate != 0UL)
        return;
    scopeRequestedRate = rate;
    scopeRestart();
}

static void scopeApplyProfile(const char *profile)
{
    if (!profile) return;

    uint32_t selected = scopeRequestedRate;
    if (strcmp(profile, "HIGH") == 0)
        selected = SCOPE_ACTIVE_HIGH_RATE;
    else if (strcmp(profile, "NORMAL") == 0 || strcmp(profile, "STANDARD") == 0 ||
             strcmp(profile, "ENGINE") == 0 || strcmp(profile, "AUTO") == 0)
        selected = SCOPE_ACTIVE_STANDARD_RATE;
    else if (strcmp(profile, "LONG") == 0)
        selected = SCOPE_ACTIVE_LONG_RATE;
    else if (strcmp(profile, "MANUAL") == 0)
        return;
    else
        return;

    scopeSetRequestedRate(selected);
}

static void scopeHandleCommand(char *line)
{
    char *token = strtok(line, " \t\r");
    if (!token || strcmp(token, "@SCP") != 0) return;
    char *command = strtok(NULL, " \t\r");
    if (!command) return;

    if (strcmp(command, "RUN") == 0 || strcmp(command, "SINGLE") == 0)
    {
        if (!scopeRunning)
        {
            scopeRunning = true;
            scopeHardwareStart();
            scopePendingFlags |= SCOPE_FLAG_DISCONTINUITY;
        }
        return;
    }

    if (strcmp(command, "STOP") == 0)
    {
        scopeRunning = false;
        scopeHardwareStop();
        return;
    }

    if (strcmp(command, "PROFILE") == 0)
    {
        scopeApplyProfile(strtok(NULL, " \t\r"));
        return;
    }

    if (strcmp(command, "RATE") == 0)
    {
        char *arg = strtok(NULL, " \t\r");
        if (!arg) return;
        const uint32_t rate = strtoul(arg, NULL, 10);
        scopeSetRequestedRate(rate);
        return;
    }

    // Backwards-compatible no-ops. They must never influence acquisition.
    if (strcmp(command, "TRIG_CH") == 0 ||
        strcmp(command, "TRIG_EDGE") == 0 ||
        strcmp(command, "TRIG_LEVEL") == 0 ||
        strcmp(command, "PRETRIGGER") == 0 ||
        strcmp(command, "TRIG_MODE") == 0 ||
        strcmp(command, "ACQ_MODE") == 0)
    {
        return;
    }
}

static void scopeReadCommands()
{
    while (Serial.available() > 0)
    {
        const char c = (char)Serial.read();
        if (c == '\n')
        {
            if (!scopeCommandOverflow)
            {
                scopeCommandBuffer[scopeCommandLength] = '\0';
                scopeHandleCommand(scopeCommandBuffer);
            }
            scopeCommandLength = 0U;
            scopeCommandOverflow = false;
        }
        else if (c != '\r' && !scopeCommandOverflow)
        {
            if (scopeCommandLength < sizeof(scopeCommandBuffer) - 1U)
                scopeCommandBuffer[scopeCommandLength++] = c;
            else
            {
                // Discard the rest of an oversized line. Resetting length here
                // could make the trailing bytes look like a new valid command.
                scopeCommandLength = 0U;
                scopeCommandOverflow = true;
            }
        }
    }
}

void setup()
{
    Serial.begin(SCOPE_SERIAL_BAUD);
    scopeRequestedRate = SCOPE_INITIAL_RATE;
    if (scopeRequestedRate > SCOPE_PROFILE_MAX_RATE)
        scopeRequestedRate = SCOPE_PROFILE_MAX_RATE;
    scopeHardwareInit();
    scopeRunning = true;
    scopeHardwareStart();
}

void loop()
{
    scopeReadCommands();
    scopeStreamService();
}
