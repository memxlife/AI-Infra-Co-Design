// CPU memory traffic for the contention test: T threads each copy a private 256 MB buffer in a loop for S seconds;
// prints the total GB/s moved (read + write).
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double secs; static double gbps[64];
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void* run(void* a) {
  long id = (long)a; size_t n = (size_t)256 << 20; char* x = malloc(n); char* y = malloc(n); memset(x, 1, n); memset(y, 2, n);
  double t0 = now(), moved = 0.0, t;
  volatile unsigned long sink = 0;
  while ((t = now()) - t0 < secs) { memcpy(y, x, n); __asm__ volatile("" ::: "memory"); sink += y[(size_t)(moved / 2) % n]; x[0]++; moved += 2.0 * (double)n; }
  gbps[id] = moved / (t - t0) / 1e9; if (sink == 12345) puts(""); free(x); free(y); return 0;
}
int main(int c, char** v) {
  int T = atoi(v[1]); secs = atof(v[2]); pthread_t th[64];
  for (long i = 0; i < T; i++) pthread_create(&th[i], 0, run, (void*)i);
  double s = 0; for (int i = 0; i < T; i++) { pthread_join(th[i], 0); s += gbps[i]; }
  printf("RESULT cpu threads=%d GBs=%.1f\n", T, s); return 0;
}
