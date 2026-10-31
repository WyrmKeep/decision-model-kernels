// scratch: runs the conv input / conv state / ssm_conv + silu graph of a Gated DeltaNet layer on the GPU
// and writes the silu output and the whole cache as raw floats. Run it with and without the fusion
// (mode "fused" / "parent") and compare the two files with cmp.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct cfg {
    int  d_conv;
    int  C;
    int  T;
    int  S;
    bool transposed;
    bool gallocr;
};

static std::vector<float> rnd(size_t n, uint32_t seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) {
        const uint32_t r = g() % 16;
        v[i] = r == 0 ? 0.0f : r == 1 ? -0.0f : r == 2 ? u(g) * 1e-30f : u(g) * (r == 3 ? 50.0f : 1.0f);
    }
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: bitcheck fused|parent out.bin dist-dir [big]\n");
        return 2;
    }
    const bool parent = strcmp(argv[1], "parent") == 0;
    const bool big    = argc > 4;
    ggml_backend_load_all_from_path(argv[3]);
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!be) {
        fprintf(stderr, "no GPU backend\n");
        return 1;
    }

    std::vector<cfg> cfgs;
    for (int d_conv : { 3, 4, 5, 7, 9, 15 }) {
        for (bool tr : { true, false }) {
            for (int T : { 1, 2, 3, 4, 5, 8, 9, 14, 15, 16, 17, 31, 32, 33, 34, 35, 63, 64, 65, 66, 67, 96, 97, 100, 127, 129, 257, 515 }) {
                cfgs.push_back({ d_conv, 128, T, 1, tr, false });
            }
            for (int T : { 1, 7, 33, 70, 200 }) {
                cfgs.push_back({ d_conv, 256, T, 3, tr, false });
                cfgs.push_back({ d_conv, 384, T, 2, tr, true });
            }
        }
    }
    if (big) {
        for (int T : { 1, 2, 5, 64, 515, 1024, 2048 }) {
            cfgs.push_back({ 4, 8192, T, 1, true, false });
            cfgs.push_back({ 4, 8192, T, 3, true, false });
        }
    }

    FILE * f = fopen(argv[2], "wb");
    int idx = 0;
    for (const cfg & c : cfgs) {
        const int n_state = c.d_conv - 1;
        ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx_in = ggml_init(ip);
        ggml_context * ctx    = ggml_init(ip);
        ggml_cgraph *  gf     = ggml_new_graph(ctx);

        ggml_tensor * state = ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, n_state, c.C, c.S);
        ggml_tensor * qkv   = c.transposed ? ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, c.C, c.T, c.S)
                                           : ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, c.T, c.C, c.S);
        ggml_tensor * w     = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, c.d_conv, c.C);
        ggml_tensor * cache = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, n_state * c.C, c.S + 2);
        ggml_backend_buffer_t buf_in = ggml_backend_alloc_ctx_tensors(ctx_in, be);

        ggml_tensor * st = state;
        ggml_tensor * qk = qkv;
        if (c.gallocr) {
            st = ggml_scale(ctx, st, 1.0f);
            qk = ggml_scale(ctx, qk, 1.0f);
        }
        ggml_tensor * xn   = c.transposed ? ggml_transpose(ctx, qk) : qk;
        ggml_tensor * conv = ggml_concat(ctx, st, xn, 0);
        ggml_tensor * last = ggml_view_3d(ctx, conv, n_state, c.C, c.S, conv->nb[1], conv->nb[2], ggml_row_size(conv->type, c.T));
        ggml_tensor * slot = ggml_view_2d(ctx, cache, n_state * c.C, c.S, cache->nb[1], cache->nb[1]);
        ggml_tensor * cpy  = ggml_cpy(ctx, last, slot);
        ggml_tensor * out  = ggml_silu(ctx, ggml_ssm_conv(ctx, conv, w));
        ggml_build_forward_expand(gf, cpy);
        if (parent) {
            // a node between the cpy and the ssm_conv: the graph has no match for the fusion
            ggml_build_forward_expand(gf, ggml_scale(ctx, w, 1.0f));
        }
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t galloc = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        if (c.gallocr) {
            galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            if (!ggml_gallocr_alloc_graph(galloc, gf)) {
                fprintf(stderr, "gallocr failed\n");
                return 1;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        }

        auto set = [&](ggml_tensor * t, uint32_t seed) {
            std::vector<float> v = rnd(ggml_nelements(t), seed + idx * 7919);
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        };
        set(state, 1);
        set(qkv, 2);
        set(w, 3);
        set(cache, 4);

        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed\n");
            return 1;
        }
        ggml_backend_synchronize(be);

        std::vector<float> o(ggml_nelements(out)), k(ggml_nelements(cache));
        ggml_backend_tensor_get(out, o.data(), 0, ggml_nbytes(out));
        ggml_backend_tensor_get(cache, k.data(), 0, ggml_nbytes(cache));
        fwrite(o.data(), sizeof(float), o.size(), f);
        fwrite(k.data(), sizeof(float), k.size(), f);

        if (galloc) {
            ggml_gallocr_free(galloc);
        }
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        ggml_backend_buffer_free(buf_in);
        ggml_free(ctx);
        ggml_free(ctx_in);
        idx++;
    }
    fclose(f);
    fprintf(stderr, "%s: %d configs\n", argv[1], idx);
    return 0;
}
