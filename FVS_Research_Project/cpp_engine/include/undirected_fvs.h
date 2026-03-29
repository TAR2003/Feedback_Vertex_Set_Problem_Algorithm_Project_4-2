#pragma once

#include "graph_base.h"

class UndirectedFVS : public GraphBase {
public:
    void load(const std::string& path) override;
    void solve();
};
