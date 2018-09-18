#include "AudioFingerprinter.h"

#define DR_WAV_IMPLEMENTATION

#include "dr_wav.h"

#define STB_FFT_IMPLEMENTAION

#include "stb_fft.h"
#include <float.h>
#include <math.h>
#include "timing.h"


Peak peakCreate(int16_t timeBin, int16_t frequencyBin) {
    Peak ret = {timeBin, frequencyBin};
    return ret;
}

typedef struct {
    size_t x;
    size_t y;
} Point;

Point pointCreate(size_t x, size_t y) {
    Point ret = {x, y};
    return ret;
}

static void
erodeFloat(const float *src, size_t s_width, float *dst, size_t d_width, size_t d_height, const int *ofsvec, int n) {
    const int *ofs = &ofsvec[0];

    for (int y = 0; y < d_height; y++) {
        const float *sptr = (float *) src + y * s_width;
        float *dptr = dst + y * d_width;

        for (int x = 0; x < d_width; x++) {
            float result = sptr[x + ofs[0]];
            for (int i = 1; i < n; i++)
                result = min(result, sptr[x + ofs[i]]);
            dptr[x] = result;
        }
    }
}

static void
dilateFloat(const float *src, size_t s_width, float *dst, size_t d_width, size_t d_height, const int *ofsvec, int n) {
    const int *ofs = &ofsvec[0];

    for (int y = 0; y < d_height; y++) {
        const float *sptr = (float *) src + y * s_width;
        float *dptr = dst + y * d_width;

        for (int x = 0; x < d_width; x++) {
            float result = sptr[x + ofs[0]];
            for (int i = 1; i < n; i++)
                result = max(result, sptr[x + ofs[i]]);
            dptr[x] = result;
        }
    }
}


static inline
void scalarToRawData(unsigned char s, unsigned char *const buf, const int unroll_to) {
    buf[0] = s;
    for (int i = 1; i < unroll_to; i++)
        buf[i] = buf[i - 1];
}

void
copyMakeBorder(const unsigned char *src, size_t s_width, size_t s_height, unsigned char *dst, size_t top, size_t bottom,
               size_t left,
               size_t right,
               const unsigned char borderValue, int esz
) {

    size_t d_width = s_width + left + right;

    size_t width = s_width * esz, dst_width = d_width * esz;

    unsigned char *valvec = (unsigned char *) calloc((d_width * esz), sizeof(unsigned char));
    if (valvec == NULL)
        return;
    unsigned char *val = &valvec[0];
    scalarToRawData(borderValue, val, (int) (s_width
                                             + left + right));

    left *= esz;
    right *= esz;
    for (int i = 0; i < s_height; i++) {
        const unsigned char *sptr = src + s_width * esz * i;
        unsigned char *dptr = dst + (i + top) * d_width * esz + left;
        for (int j = 0; j < left; j++)
            dptr[j - left] = val[j];
        if (dptr != sptr)
            for (int j = 0; j < width; j++)
                dptr[j] = sptr[j];
        for (int j = 0; j < right; j++)
            dptr[j + width] = val[j];
    }

    for (int i = 0; i < top; i++) {
        unsigned char *dptr = dst + d_width * esz * i;
        for (int j = 0; j < dst_width; j++)
            dptr[j] = val[j];
    }

    for (int i = 0; i < bottom; i++) {
        unsigned char *dptr = dst + d_width * (i + top + s_height);
        for (int j = 0; j < dst_width; j++)
            dptr[j] = val[j];
    }
    free(valvec);
}

void
erodeProc(const float *_src, float *dst, size_t width, size_t height, const unsigned char *kernel, size_t kernelCols,
          size_t kernelRows) {

    unsigned char borderValue = 1;
    Point anchor = pointCreate(kernelCols / 2, kernelRows / 2);
    size_t top = anchor.y;
    size_t bottom = kernelRows - anchor.y - 1;
    size_t left = anchor.x;
    size_t right = kernelCols - anchor.x - 1;
    size_t d_height = height + top + bottom;
    size_t d_width = width + left + right;
    unsigned char *src = (unsigned char *) calloc((d_width * d_height), sizeof(float));
    int *ofs = (int *) calloc((kernelRows * kernelCols), sizeof(int));
    if (src == NULL || ofs == NULL) {
        if (src) free(src);
        if (ofs) free(ofs);
        return;
    }
    copyMakeBorder((unsigned char *) _src, width, height, src, top, bottom,
                   left, right,
                   borderValue, sizeof(float));

    size_t step = d_width;
    int ofsSize = 0;
    for (size_t i = 0; i < kernelRows; i++) {
        const unsigned char *kernelPtr = kernel + i * kernelCols;
        for (size_t j = 0; j < kernelCols; j++) {
            if (kernelPtr[j] != 0) {
                ofs[ofsSize] = (int) (i * step + j);
                ofsSize++;
            }
        }
    }
    if (ofsSize == 0) {
        ofs[ofsSize] = (int) (anchor.y * step + anchor.x);
        ofsSize++;
    }
    erodeFloat((float *) src, step, dst, width, height, ofs, ofsSize);
    free(src);
    free(ofs);
}

void
dilateProc(const float *_src, float *dst, size_t width, size_t height, const unsigned char *kernel, size_t kernelCols,
           size_t kernelRows) {

    unsigned char borderValue = 0;
    Point anchor = pointCreate(kernelCols / 2, kernelRows / 2);
    size_t top = anchor.y;
    size_t bottom = kernelRows - anchor.y - 1;
    size_t left = anchor.x;
    size_t right = kernelCols - anchor.x - 1;
    size_t d_height = height + top + bottom;
    size_t d_width = width + left + right;
    unsigned char *src = (unsigned char *) calloc((d_width * d_height), sizeof(float));
    int *ofs = (int *) calloc((kernelRows * kernelCols), sizeof(int));
    if (src == NULL || ofs == NULL) {
        if (src) free(src);
        if (ofs) free(ofs);
        return;
    }
    copyMakeBorder((unsigned char *) _src, width, height, src, top, bottom,
                   left, right,
                   borderValue, sizeof(float));
    size_t step = d_width;
    int ofsSize = 0;
    for (int i = 0; i < kernelRows; i++) {
        const unsigned char *kernelPtr = kernel + i * kernelCols;
        for (int j = 0; j < kernelCols; j++) {
            if (kernelPtr[j] != 0) {
                ofs[ofsSize] = (int) (i * step + j);
                ofsSize++;
            }

        }
    }
    if (ofsSize == 0) {
        ofs[ofsSize] = (int) (anchor.y * step + anchor.x);
        ofsSize++;
    }
    dilateFloat((float *) src, step, dst, width, height, ofs, ofsSize);
    free(src);
    free(ofs);
}


int CalculateAudioHashes(float *audioBuffer, int audioBufferLen, HashObj **hashOffsets, int sampleRate,
                         float windowSizeSeconds, float overlapRatio, size_t fanValue, float ampMin,
                         size_t peakNeighbourhoodSize, int minHashTimeDelta, int maxHashTimeDelta, bool peakSort) {

    int16_t nfft = (int16_t) roundf(windowSizeSeconds * sampleRate);
    nfft += nfft % 2;
    float *inBuffer = audioBuffer;
    int overlap = (int) (nfft * overlapRatio);
    int16_t bins = ((audioBufferLen - nfft) / overlap);
    int16_t specBins = (int16_t) (nfft / 2 + 1);
    float *spectrum = (float *) calloc((bins * specBins), sizeof(float));
    float *hannWindow = (float *) calloc(nfft, sizeof(float));
    float *buffer = (float *) calloc(nfft, sizeof(float));
    Peak *peaks = (Peak *) calloc((specBins * bins), sizeof(Peak));
    int plan_size = stb_fft_real_plan_dft_1d(nfft, NULL);
    stb_fft_real_plan *plan = (stb_fft_real_plan *) calloc(plan_size, 1);
    cmplx *cmplx_out = (cmplx *) calloc((size_t) nfft, sizeof(cmplx));
    if (hannWindow == NULL || buffer == NULL || spectrum == NULL || peaks == NULL || plan_size == 0 || plan == NULL ||
        cmplx_out == NULL) {
        if (hannWindow) free(hannWindow);
        if (buffer) free(buffer);
        if (spectrum) free(spectrum);
        if (peaks) free(peaks);
        if (plan) free(plan);
        if (cmplx_out) free(cmplx_out);
        return 0;
    }
    float TWOPI = 6.283185307179586476925286766559005768394338798750211641949889f;
    for (int n = 0; n < nfft; n++) {
        float HannMultiplier = 0.5f * (1.0f - cosf(TWOPI * n / (nfft - 1)));
        hannWindow[n] = HannMultiplier;
    }
    float *spectrumPtr = spectrum;
    int specCounts = 0;
    stb_fft_real_plan_dft_1d(nfft, plan);
    for (int i = 0; i < bins; ++i) {
        memcpy(buffer, inBuffer, sizeof(float) * nfft);
        for (int16_t n = 0; n < nfft; ++n) {
            buffer[n] *= hannWindow[n];
        }
        stb_fft_r2c_exec(plan, buffer, cmplx_out);
        specBins = 0;
        for (int k = 0; k < nfft / 2 + 1; ++k) {
            float val = (cmplx_out[k].imag * cmplx_out[k].imag) + (cmplx_out[k].real * cmplx_out[k].real);
            val = 10 * (log10f(val));
            if (fpclassify(val) == FP_SUBNORMAL)
                val = 0;
            spectrumPtr[specBins] = (val);
            specBins++;
        }
        inBuffer += overlap;
        spectrumPtr += specBins;
        specCounts += specBins;
    }
    free(plan);
    free(cmplx_out);
    free(hannWindow);
    free(buffer);
    size_t peaksSize = 0;
    getPeaks(spectrum, specBins, bins, peaks, &peaksSize, peakNeighbourhoodSize, ampMin);
    free(spectrum);
    size_t hash_counts = (size_t) buildHashes(peaks, peaksSize, hashOffsets, peakSort, fanValue, minHashTimeDelta,
                                              maxHashTimeDelta);
    free(peaks);
    return (int) hash_counts;
}


bool filteredPeaks(float *spectrum, float *detectedPeaks, int16_t specBins, int16_t bins, Peak *peaks,
                   size_t *peaksSize,
                   float ampMin) {
    size_t peaksCount = 0;
    for (int16_t i = 0; i < bins; ++i) {
        float *spectrumPtr = spectrum + i * specBins;
        float *detectedPeaksPtr = detectedPeaks + i * specBins;
        for (int16_t j = 0; j < specBins; ++j) {
            if (detectedPeaksPtr[j] > 0) {
                if (spectrumPtr[j] > ampMin) {
                    peaks[peaksCount] = (peakCreate(i, j));
                    peaksCount++;
                }
            }
        }
    }
    *peaksSize = peaksCount;
    return true;
}

bool detectedPeaks(float *dilateBackground, float *erodedBackground, float *output, int16_t specBins, int16_t bins) {
    for (int i = 0; i < bins; ++i) {
        float *out = output + i * specBins;
        float *dilate = dilateBackground + i * specBins;
        float *eroded = erodedBackground + i * specBins;
        for (int j = 0; j < specBins; ++j) {
            out[j] = dilate[j] - eroded[j];
        }
    }
    return true;
}

bool erodedFilter(float *spectrum, unsigned char *kernel, size_t kernelCols,
                  size_t kernelRows, float *erodedBackground, int16_t specBins, int16_t bins) {
    for (int i = 0; i < bins; ++i) {
        float *out = erodedBackground + i * specBins;
        float *in = spectrum + i * specBins;
        for (int j = 0; j < specBins; ++j) {
            if (in[j] == 0) {
                out[j] = 1;
            } else {
                out[j] = 0;
            }
        }
    }
    erodeProc(erodedBackground, erodedBackground, specBins, bins, kernel, kernelCols, kernelRows);
    return true;
}

bool dilateFilter(float *spectrum, unsigned char *kernel, size_t kernelCols, size_t kernelRows,
                  float *dilateBackground, int16_t specBins, int16_t bins) {
    dilateProc(spectrum, dilateBackground, specBins, bins, kernel, kernelCols, kernelRows);
    for (int i = 0; i < bins; ++i) {
        float *out = dilateBackground + i * specBins;
        float *in = spectrum + i * specBins;
        for (int j = 0; j < specBins; ++j) {
            if (out[j] != in[j]) {
                out[j] = 0;
            } else {
                out[j] = 1;
            }
        }
    }
    return true;
}

bool buildKernel(unsigned char *kernel, size_t size) {
    size_t half = (size - 1) / 2;
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            size_t x = i;
            size_t y = j;
            if (i > half) {
                x = (size - 1) - i;
            }
            if (j > half) {
                y = (size - 1) - j;
            }
            if (x + y >= half) {
                kernel[i * size + j] = 1;
            } else {
                kernel[i * size + j] = 0;
            }
        }
    }
    return true;
}

bool getPeaks(float *spectrum, int16_t specBins, int16_t bins,
              Peak *peaks, size_t *peaksSize, size_t peakNeighbourhoodSize, float ampMin) {
    size_t size = 1 + 2 * peakNeighbourhoodSize;
    unsigned char *kernel = (unsigned char *) calloc(size * size, sizeof(unsigned char));
    float *dilateBackground = (float *) calloc((size_t) (specBins * bins), sizeof(float));
    float *erodedBackground = (float *) calloc((size_t) (specBins * bins), sizeof(float));
    float *detectedPeaksResult = (float *) calloc((size_t) (specBins * bins), sizeof(float));
    if (kernel == NULL || dilateBackground == NULL || erodedBackground == NULL || detectedPeaksResult == NULL) {
        if (kernel) free(kernel);
        if (dilateBackground) free(dilateBackground);
        if (erodedBackground) free(erodedBackground);
        if (detectedPeaksResult) free(detectedPeaksResult);
        return false;
    }
    buildKernel(kernel, size);
    dilateFilter(spectrum, kernel, size, size, dilateBackground, specBins, bins);
    erodedFilter(spectrum, kernel, size, size, erodedBackground, specBins, bins);
    detectedPeaks(dilateBackground, erodedBackground, detectedPeaksResult, specBins, bins);
    filteredPeaks(spectrum, detectedPeaksResult, specBins, bins, peaks, peaksSize, ampMin);
    free(kernel);
    free(dilateBackground);
    free(erodedBackground);
    free(detectedPeaksResult);
    return true;
}

int cmpPeak(const void *left, const void *right) {
    const int16_t a = ((Peak *) left)->timeBin, b = ((Peak *) right)->timeBin;
    return (a < b) ? -1 : (a > b);
}

void buildHash(unsigned char *hashSeed, int16_t freq1, int16_t freq2, int16_t tDelta) {
    unsigned char *hashSeedPtr = hashSeed;
    memcpy(hashSeedPtr, &freq1, sizeof(freq1));
    hashSeedPtr += sizeof(freq1);
    memcpy(hashSeedPtr, &freq2, sizeof(freq2));
    hashSeedPtr += sizeof(freq2);
    memcpy(hashSeedPtr, &tDelta, sizeof(tDelta));
}

int buildHashes(Peak *peaks, size_t peaksSize,
                HashObj **hashOffsets, bool peakSort, size_t fanValue,
                int minHashTimeDelta, int maxHashTimeDelta) {
    int hashInit = (int) (peaksSize);
    HashObj *hashOffset = (HashObj *) calloc((size_t) hashInit, sizeof(HashObj));
    if (hashOffset == NULL) return 0;
    if (peakSort) {
        qsort(peaks, peaksSize, sizeof(Peak), cmpPeak);
    }
    int hashCounts = 0;
    size_t fingerprintSize = 6;
    for (size_t i = 0; i < peaksSize; ++i) {
        for (size_t j = 1; j < fanValue; ++j) {
            if (i + j < peaksSize) {
                int16_t time1 = peaks[i].timeBin;
                int16_t time2 = peaks[i + j].timeBin;
                int16_t tDelta = time2 - time1;
                if (tDelta >= minHashTimeDelta && tDelta <= maxHashTimeDelta) {
                    int16_t freq1 = peaks[i].frequencyBin;
                    int16_t freq2 = peaks[i + j].frequencyBin;
                    int offsets = hashCounts;
                    if (hashCounts + 1 > hashInit) {
                        int newSize = hashInit * 2;
                        HashObj *newBuffer = (HashObj *) realloc(hashOffset, newSize * sizeof(HashObj));
                        if (newBuffer != NULL) {
                            hashOffset = newBuffer;
                            hashInit = newSize;
                        } else {
                            cleanHash(hashOffset, hashCounts);
                            return 0;
                        }
                    }
                    unsigned char *hash = (unsigned char *) calloc(fingerprintSize, sizeof(unsigned char));
                    if (hash == NULL) {
                        cleanHash(hashOffset, hashCounts);
                        return 0;
                    }
                    buildHash(hash, freq1, freq2, tDelta);
                    if (offsets <= time1) {
                        for (int x = offsets; x < time1; ++x) {
                            hashCounts++;
                        }
                        HashObj hashObj;
                        hashObj.size = fingerprintSize;
                        hashObj.buffer = hash;
                        hashOffset[hashCounts] = hashObj;
                        hashCounts++;
                    } else {
                        if (hashOffset[time1].buffer == NULL) {
                            hashOffset[time1].buffer = hash;
                            hashOffset[time1].size = fingerprintSize;
                        } else {
                            size_t newHashLength = hashOffset[time1].size + fingerprintSize;
                            unsigned char *buffer = (unsigned char *) realloc(hashOffset[time1].buffer,
                                                                              newHashLength * sizeof(unsigned char));
                            if (buffer != NULL) {
                                hashOffset[time1].buffer = buffer;
                                memcpy(hashOffset[time1].buffer + hashOffset[time1].size, hash, fingerprintSize);
                                hashOffset[time1].size = newHashLength;
                                free(hash);
                            } else {
                                cleanHash(hashOffset, hashCounts);
                                return 0;
                            }
                        }

                    }
                }
            }
        }
    }
    *hashOffsets = hashOffset;
    printf("hashCounts :%d ", hashCounts);
    printf("peaksSize :%d ", peaksSize);

    return hashCounts;
}

void cleanHash(HashObj *hashOffset, int hashCounts) {
    if (hashOffset != NULL) {
        for (int n = 0; n < hashCounts; n++) {
            if (hashOffset[hashCounts].buffer != NULL)
                free(hashOffset[hashCounts].buffer);
        }
        free(hashOffset);
    }
}


float *wavRead_f32(char *filename, uint32_t *sampleRate, uint64_t *totalSampleCount, unsigned int *channels) {
    float *buffer = drwav_open_and_read_file_f32(filename, channels, sampleRate, totalSampleCount);
    if (buffer == NULL) {
        fprintf(stderr, "read file error.\n");
        exit(1);
    }
    if (*channels == 2) {
        float *bufferSave = buffer;
        for (uint64_t i = 0; i < *totalSampleCount; i += 2) {
            *bufferSave++ = ((buffer[i] + buffer[i + 1]) * 0.5f);
        }
        *totalSampleCount = *totalSampleCount >> 1;
        *channels = 1;
    } else if (*channels != 1) {
        drwav_free(buffer);
        buffer = NULL;
        *sampleRate = 0;
        *totalSampleCount = 0;
    }
    return buffer;
}


int main(int argc, char **argv) {
    printf("Audio Fingerprinter\n");
    printf("blog:http://cpuimage.cnblogs.com/\n");
    if (argc < 2) {
        printf("usage: %s filename\n", argv[0]);
        printf("press any key to exit.\n");
        getchar();
        return -1;
    }
    char *filename = argv[1];
    uint32_t sampleRate = 0;
    uint64_t num_samples = 0;
    unsigned int channels = 0;
    float *data_in = wavRead_f32(filename, &sampleRate, &num_samples, &channels);
    if (data_in == 0) return -1;
    float windowSizeSeconds = 0.37f;
    float overlapRatio = 0.5f;
    int fanValue = 15;
    float ampMin = 10.0f;
    size_t peakNeighbourhoodSize = 20;
    int minHashTimeDelta = 0;
    int maxHashTimeDelta = 200;
    bool peakSort = true;
    HashObj *hashOffsets = NULL;
    int hash_count = 0;
    double startTime = now();
    hash_count = CalculateAudioHashes(data_in, num_samples, &hashOffsets, sampleRate,
                                      windowSizeSeconds,
                                      overlapRatio,
                                      (size_t) fanValue,
                                      ampMin,
                                      peakNeighbourhoodSize,
                                      minHashTimeDelta,
                                      maxHashTimeDelta,
                                      peakSort);
    double time_interval = calcElapsed(startTime, now());
    free(data_in);
    printf("[%s] Fingerprinter:\n", filename);
    if (hashOffsets != NULL) {
        for (int x = 0; x < hash_count; x++) {
            for (size_t y = 0; y < hashOffsets[x].size; y++) {
                printf("%d \t", (int) (hashOffsets[x].buffer[y]));
            }
            if (hashOffsets[x].size > 0)
                printf("\n");
            if (hashOffsets[x].buffer)
                free(hashOffsets[x].buffer);
        }
        free(hashOffsets);
    }
    printf("time interval: %f ms\n", (time_interval * 1000));
    printf("press any key to exit.\n");
    getchar();
    return 0;
}


