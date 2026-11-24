// CPU memory latency and bandwidth on a two-socket server, near and far socket. Memory is placed by first touch from
// threads pinned to one socket (no numactl needed). Latency: one thread, dependent pointer chase over a random
// cycle of 64 B lines, 2 MB pages (madvise), working sets 32 KB to 2 GB. Bandwidth: all cores of a socket reading
// (sum) and writing 8 GB, best of 5.
// gcc -O2 -pthread -o cpu_mem cpu_mem.c ; ./cpu_mem 32 "0-31" "32-63"   (cores per socket, socket-0 and socket-1 cores)
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static int first[2], ncores;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void pin(int cpu) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }
static void* big(size_t b) { void* p = mmap(0, b, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); madvise(p, b, MADV_HUGEPAGE); return p; }

typedef struct { char* p; size_t lo, hi; int cpu, op; double sum; } Job;
static void* work(void* a) {
  Job* j = a; pin(j->cpu);
  if (j->op == 0) memset(j->p + j->lo, 1, j->hi - j->lo);                     // first touch / write
  else { uint64_t s = 0; const uint64_t* q = (uint64_t*)(j->p + j->lo); size_t n = (j->hi - j->lo) / 8;
         for (size_t i = 0; i < n; i += 8) s += q[i] + q[i + 1] + q[i + 2] + q[i + 3] + q[i + 4] + q[i + 5] + q[i + 6] + q[i + 7]; j->sum = s; }
  return 0;
}
static double run(char* p, size_t b, int sock, int op) {                      // all cores of a socket on one buffer
  pthread_t t[256]; Job jb[256]; double t0 = now();
  for (int i = 0; i < ncores; i++) { jb[i] = (Job){p, b / ncores * i, b / ncores * (i + 1), first[sock] + i, op, 0}; pthread_create(&t[i], 0, work, &jb[i]); }
  for (int i = 0; i < ncores; i++) pthread_join(t[i], 0);
  return now() - t0;
}
typedef struct { size_t ws; int home, cpu; double ns; } Lat;
static void* touch(void* a) { Job* j = a; pin(j->cpu); memset(j->p, 0, j->hi); return 0; }
static void* chase_t(void* a) {
  Lat* l = a; size_t m = l->ws / 64; char* p = big(l->ws);
  Job tj = {p, 0, l->ws, first[l->home], 0, 0}; pthread_t th; pthread_create(&th, 0, touch, &tj); pthread_join(th, 0);   // place on home
  pin(l->cpu);
  uint32_t* perm = malloc(m * 4); for (size_t i = 0; i < m; i++) perm[i] = i;
  srand(7); for (size_t i = m - 1; i > 1; i--) { size_t k = 1 + (size_t)rand() % i; uint32_t x = perm[i]; perm[i] = perm[k]; perm[k] = x; }
  for (size_t i = 0; i < m; i++) *(char**)(p + (size_t)perm[i] * 64) = p + (size_t)perm[(i + 1) % m] * 64;
  free(perm);
  char* q = p; long steps = 20000000; for (long s = 0; s < 1000000; s++) q = *(char**)q;
  double t0 = now(); for (long s = 0; s < steps; s++) q = *(char**)q; l->ns = (now() - t0) * 1e9 / steps;
  if (q == 0) printf("x");
  munmap(p, l->ws); return 0;
}

int main(int argc, char** argv) {
  ncores = atoi(argv[1]); first[0] = atoi(argv[2]); first[1] = atoi(argv[3]);
  size_t sets[] = {32 << 10, 1 << 20, 32 << 20, 512ull << 20, 2048ull << 20};
  for (int far = 0; far < 2; far++) {
    printf("{\"test\": \"latency\", \"memory\": \"%s socket\", \"ns\": {", far ? "far" : "near");
    for (int i = 0; i < 5; i++) { Lat l = {sets[i], far, first[0], 0}; pthread_t t; pthread_create(&t, 0, chase_t, &l); pthread_join(t, 0); printf("%s\"%zuKB\": %.1f", i ? ", " : "", sets[i] >> 10, l.ns); }
    printf("}}\n"); fflush(stdout);
  }
  size_t b = 8ull << 30; char* p0 = big(b); char* p1 = big(b);
  run(p0, b, 0, 0); run(p1, b, 1, 0);                                         // place p0 on socket 0, p1 on socket 1
  double r[4] = {1e9, 1e9, 1e9, 1e9};
  for (int k = 0; k < 5; k++) {
    double x;
    if ((x = run(p0, b, 0, 1)) < r[0]) r[0] = x;                               // socket 0 reads its own memory
    if ((x = run(p1, b, 0, 1)) < r[1]) r[1] = x;                               // socket 0 reads socket 1's memory
    if ((x = run(p0, b, 0, 0)) < r[2]) r[2] = x;                               // socket 0 writes its own memory
  }
  // both sockets at once, each on its own memory
  pthread_t t[2]; Job j0 = {0}, j1 = {0}; double best = 1e9;
  for (int k = 0; k < 5; k++) {
    double t0 = now();
    struct { char* p; int s; } a[2] = {{p0, 0}, {p1, 1}};
    for (int s = 0; s < 2; s++) { void* (*f)(void*) = 0; (void)f; }
    pthread_t th[256]; Job jb[256];
    for (int s = 0; s < 2; s++) for (int i = 0; i < ncores; i++) { jb[s * ncores + i] = (Job){a[s].p, b / ncores * i, b / ncores * (i + 1), first[s] + i, 1, 0}; pthread_create(&th[s * ncores + i], 0, work, &jb[s * ncores + i]); }
    for (int i = 0; i < 2 * ncores; i++) pthread_join(th[i], 0);
    double x = now() - t0; if (x < best) best = x;
  }
  (void)t; (void)j0; (void)j1;
  printf("{\"test\": \"bandwidth\", \"GBs\": {\"one socket reads its own memory\": %.0f, \"one socket reads the far memory\": %.0f, \"one socket writes its own memory\": %.0f, \"both sockets read their own memory\": %.0f}}\n",
         b / r[0] / 1e9, b / r[1] / 1e9, b / r[2] / 1e9, 2 * b / best / 1e9);
  return 0;
}
