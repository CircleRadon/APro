#include <thread>
#include <iostream>
#include <stdlib.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include "gp_process.hpp"

static void forward_kernel(float *ptr, int *edge, float *w, float *p, int edge_num, int vertex_num, float sigma){
    struct Edge e[edge_num];

    for (int i=0; i<edge_num; i++){
        e[i].u = edge[i*2];
        e[i].v = edge[i*2+1];
        e[i].w = w[i];
    }
    sort(e, e + edge_num, [](Edge& x, Edge& y) {
        return x.w < y.w;
    });
    auto st = UnionSet(vertex_num, sigma);
    for (int i=0; i<vertex_num; i++){
        st.psum[i] = p[i];
    }
    for(int i = 0; i < edge_num; i++) {
        int u = e[i].u;
        int v = e[i].v;
        float w = e[i].w;
        st.unite(u, v, w);
    }

    int root = [&]() {
        for(int i = 0; i < vertex_num; i++) {
            if(st.find(i) == i) return i;
        }
    }();

    st.dfs(root);

    for (int i = 0; i < vertex_num; i++) {
        ptr[i] = (st.ptag[i]+p[i])/(float)(st.tag[i]+1);
    }
}

at::Tensor gp_forward(
    const at::Tensor& edges,
    const at::Tensor& edge_weight,
    const at::Tensor& prob,
    float sigma){

    auto edges_cpu = edges.cpu();
    auto edge_weight_cpu = edge_weight.cpu();
    auto prob_cpu = prob.cpu();

    int *edge_ = edges_cpu.contiguous().data<int>();
    float *w_ = edge_weight_cpu.contiguous().data<float>();
    float *p_ = prob_cpu.contiguous().data<float>();

    unsigned batch_size = edges.size(0);
    unsigned edge_num = edges.size(1);
    unsigned vertex_num = prob.size(1);

    auto result = at::empty({batch_size,vertex_num}, edge_weight_cpu.options());
    float *ptr_ = result.data<float>();

    std::thread pids[batch_size];
    for (unsigned bs = 0; bs < batch_size; bs++){
        auto ptr_iter = ptr_ + bs*vertex_num;
        auto edge_iter = edge_ + bs*edge_num*2;
        auto w_iter = w_ + bs*edge_num;
        auto p_iter = p_ + bs*vertex_num;
        pids[bs] = std::thread(forward_kernel, ptr_iter, edge_iter, w_iter, p_iter, edge_num, vertex_num, sigma);
    }

    for (unsigned bs = 0; bs < batch_size; bs++){
        pids[bs].join();
    }

    auto result_tensor = result.to(edge_weight.device());
    return result_tensor;
}
