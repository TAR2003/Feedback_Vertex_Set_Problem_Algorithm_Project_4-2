#pragma once

#include "graph_base.h"

class DirectedFVS : public GraphBase {
public:
    void load(const std::string& path) override;
    void solve();
};
