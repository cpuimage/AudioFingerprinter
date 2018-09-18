#ifndef _FINGERPRINTER_H_
#define _FINGERPRINTER_H_

#include <stdint.h>
#include <stdbool.h>
#include "stb_fft.h"

#ifndef max
#define max(a, b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a, b)            (((a) < (b)) ? (a) : (b))
#endif

typedef struct {
    int16_t timeBin;
    int16_t frequencyBin;
} Peak;

typedef struct {
    unsigned char *buffer;
    size_t size;
} HashObj;

int CalculateAudioHashes(float *audioBuffer, int audioBufferLen, HashObj **hashOffsets,
                         int sampleRate, float windowSizeSeconds, float overlapRatio, size_t fanValue,
                         float ampMin,
                         size_t peakNeighbourhoodSize, int minHashTimeDelta, int maxHashTimeDelta, bool peakSort);


bool filteredPeaks(float *spectrum, float *detectedPeaks, int16_t specBins, int16_t bins, Peak *peaks,
                   size_t *peaksSize,
                   float ampMin);

bool detectedPeaks(float *dilateBackground, float *erodedBackground, float *output, int16_t specBins, int16_t bins);

bool erodedFilter(float *spectrum, unsigned char *kernel, size_t kernelCols, size_t kernelRows,
                  float *erodedBackground,
                  int16_t specBins, int16_t bins);

bool buildKernel(unsigned char *kernel, size_t size);

bool dilateFilter(float *spectrum, unsigned char *kernel, size_t kernelCols, size_t kernelRows,
                  float *dilateBackground,
                  int16_t specBins,
                  int16_t bins);

bool getPeaks(float *spectrum, int16_t specBins, int16_t bins, Peak *peaks, size_t *peaksSize,
              size_t peakNeighbourhoodSize,
              float ampMin);

int buildHashes(Peak *peaks, size_t peaksSize, HashObj **hashOffsets, bool peakSort,
                size_t fanValue, int minHashTimeDelta, int maxHashTimeDelta);

void buildHash(unsigned char *hashSeed, int16_t freq1, int16_t freq2, int16_t tDelta);

void cleanHash(HashObj *hashOffset, int hashCounts);

#endif