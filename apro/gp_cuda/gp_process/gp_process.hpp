#pragma once
#include <vector>
#include <torch/extension.h>
using namespace std;

struct Edge{
	int u, v;
	float w;
};

struct UnionSet {
	vector<int> fa;
	vector<int> size;
    vector<float> psum;
    vector<float> ptag;
	vector<float> tag;
	vector<vector<int>> g;
	float sigma;

	UnionSet(int n, float s): fa(vector<int>(n, 0)), size(vector<int>(n, 1)), tag(vector<float>(n, 0)), ptag(vector<float>(n, 0)), sigma(s),
	    g(vector<vector<int>>(n)) {
		for(int i = 0; i < n; i++) fa[i] = i;
        psum.resize(n);
	}

	int find(int x) {
		while (x != fa[x]) x = fa[x];
		return x;
    }
	void unite(int x, int y, float add) {
		x = find(x);
		y = find(y);

		assert(x != y);

		if(size[x] < size[y]) swap(x, y);

		tag[x] += size[y] * exp(-add/(sigma*sigma));
		tag[y] += size[x] * exp(-add/(sigma*sigma)) - tag[x];

        ptag[x] += psum[y] * exp(-add/(sigma*sigma));
        ptag[y] += psum[x] * exp(-add/(sigma*sigma)) - ptag[x];

		fa[y] = x;
		size[x] += size[y];
        psum[x] += psum[y];

		g[x].push_back(y);
	}

	void dfs(int u) {
		for(int v: g[u]) {
			tag[v] += tag[u];
            ptag[v] += ptag[u];
			dfs(v);
		}
	}
};


extern at::Tensor gp_forward(
    const at::Tensor & edges,
    const at::Tensor & edge_weight,
    const at::Tensor & prob,
    float sigma);

