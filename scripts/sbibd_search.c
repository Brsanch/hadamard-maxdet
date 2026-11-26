/* sbibd_search.c — fast point-wise SBIBD backtracking.
 *
 * Same algorithm as cycle50_h25_sbibd.py / cycle51_h41_sbibd.py:
 *   - Block 1, Block 2 fixed canonical (passed in as initial_blocks).
 *   - For each subsequent block, point-wise DFS picking points lex-asc,
 *     pruning by point cap, pair cap, intersection cap, can-finish bound.
 *   - Symmetry break: each new block must be lex-greater than last placed.
 *
 * Compile:  cc -O3 -Wall -shared -fPIC sbibd_search.c -o sbibd_search.so
 *
 * The library exposes a single function `sbibd_search` which fills out_blocks
 * with a complete SBIBD if one exists from the given (initial_blocks) prefix,
 * or returns 0 if no completion. nodes_out reports the number of search-tree
 * nodes visited (search() entries).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define V_MAX 128
#define K_MAX 64

static int v, k, lam;
static int blocks_arr[V_MAX][K_MAX];
static int n_blocks;
static int point_count[V_MAX];
static int pair_count[V_MAX][V_MAX];
static int in_block_at[V_MAX][V_MAX];
static int inter[V_MAX];
static int partial[K_MAX];
static int n_partial;
static long nodes_visited;
static long progress_every;
static double t_start;
static int verbose;

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int feasibility_prune(void) {
    int remaining = v - n_blocks;
    for (int p = 0; p < v; p++) {
        int need = k - point_count[p];
        if (need < 0 || need > remaining) return 0;
    }
    for (int p = 0; p < v; p++) {
        for (int q = p + 1; q < v; q++) {
            int need = lam - pair_count[p][q];
            if (need < 0) return 0;
            int kpc_p = k - point_count[p];
            int kpc_q = k - point_count[q];
            int cap = remaining;
            if (kpc_p < cap) cap = kpc_p;
            if (kpc_q < cap) cap = kpc_q;
            if (need > cap) return 0;
        }
    }
    return 1;
}

static int can_finish(int slots_remaining) {
    for (int i = 0; i < n_blocks; i++)
        if (inter[i] + slots_remaining < lam) return 0;
    return 1;
}

static int search(void);

static int build_candidate(int start, int remaining) {
    if (remaining == 0) {
        for (int i = 0; i < n_blocks; i++)
            if (inter[i] != lam) return 0;
        /* lex-floor check vs last block placed (blocks_arr[n_blocks - 1]) */
        if (n_blocks > 0) {
            int *last = blocks_arr[n_blocks - 1];
            int lex_ok = 0;
            for (int i = 0; i < k; i++) {
                if (partial[i] > last[i]) { lex_ok = 1; break; }
                if (partial[i] < last[i]) return 0;
            }
            if (!lex_ok) return 0;
        }
        /* Apply candidate as next block */
        for (int i = 0; i < k; i++) {
            int p = partial[i];
            blocks_arr[n_blocks][i] = p;
            point_count[p]++;
            in_block_at[n_blocks][p] = 1;
        }
        for (int i = 0; i < k; i++)
            for (int j = i + 1; j < k; j++) {
                int p = partial[i], q = partial[j];
                pair_count[p][q]++;
                pair_count[q][p]++;
            }
        n_blocks++;
        if (search()) return 1;
        n_blocks--;
        for (int i = 0; i < k; i++) {
            int p = partial[i];
            point_count[p]--;
            in_block_at[n_blocks][p] = 0;
        }
        for (int i = 0; i < k; i++)
            for (int j = i + 1; j < k; j++) {
                int p = partial[i], q = partial[j];
                pair_count[p][q]--;
                pair_count[q][p]--;
            }
        return 0;
    }
    if (!can_finish(remaining)) return 0;
    for (int p = start; p < v; p++) {
        if (point_count[p] >= k) continue;
        int ok = 1;
        for (int qi = 0; qi < n_partial; qi++) {
            int q = partial[qi];
            if (pair_count[p][q] >= lam) { ok = 0; break; }
        }
        if (!ok) continue;
        int bad = 0;
        for (int i = 0; i < n_blocks; i++) {
            if (in_block_at[i][p] && inter[i] + 1 > lam) { bad = 1; break; }
        }
        if (bad) continue;
        partial[n_partial++] = p;
        for (int i = 0; i < n_blocks; i++)
            if (in_block_at[i][p]) inter[i]++;
        if (build_candidate(p + 1, remaining - 1)) return 1;
        n_partial--;
        for (int i = 0; i < n_blocks; i++)
            if (in_block_at[i][p]) inter[i]--;
    }
    return 0;
}

static int search(void) {
    nodes_visited++;
    if (verbose && progress_every > 0 && (nodes_visited % progress_every) == 0) {
        double el = monotonic_seconds() - t_start;
        fprintf(stderr,
                "[c] blocks=%d/%d  nodes=%ld  elapsed=%.1fs\n",
                n_blocks, v, nodes_visited, el);
        fflush(stderr);
    }
    if (n_blocks == v) return 1;
    if (!feasibility_prune()) return 0;
    n_partial = 0;
    for (int i = 0; i < n_blocks; i++) inter[i] = 0;
    return build_candidate(0, k);
}

int sbibd_search(int v_in, int k_in, int lam_in,
                 int *initial_blocks, int n_initial,
                 int *out_blocks,
                 long progress_every_in, int verbose_in,
                 long *nodes_out, double *elapsed_out) {
    v = v_in; k = k_in; lam = lam_in;
    progress_every = progress_every_in;
    verbose = verbose_in;
    if (v > V_MAX || k > K_MAX) return -1;
    memset(point_count, 0, sizeof(point_count));
    memset(pair_count, 0, sizeof(pair_count));
    memset(in_block_at, 0, sizeof(in_block_at));
    n_blocks = 0;
    nodes_visited = 0;
    t_start = monotonic_seconds();

    for (int t = 0; t < n_initial; t++) {
        int *B = initial_blocks + t * k;
        for (int i = 0; i < k; i++) {
            int p = B[i];
            blocks_arr[n_blocks][i] = p;
            point_count[p]++;
            in_block_at[n_blocks][p] = 1;
        }
        for (int i = 0; i < k; i++)
            for (int j = i + 1; j < k; j++) {
                int p = B[i], q = B[j];
                pair_count[p][q]++;
                pair_count[q][p]++;
            }
        n_blocks++;
    }

    int found = search();
    *nodes_out = nodes_visited;
    *elapsed_out = monotonic_seconds() - t_start;
    if (!found) return 0;
    for (int t = 0; t < v; t++)
        for (int i = 0; i < k; i++)
            out_blocks[t * k + i] = blocks_arr[t][i];
    return 1;
}
