#pragma once

// Base class for graphs
class GraphBase {
public:
    virtual ~GraphBase() = default;
    virtual void load(const std::string& path) = 0;
};
