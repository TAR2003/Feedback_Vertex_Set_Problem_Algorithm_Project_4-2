#pragma once
/**
 * @file graph_base.h
 * @brief Abstract base class for all graph types in the FVS framework.
 *
 * Design philosophy: Virtual Deletion.
 * Instead of physically removing vertices (which would be O(n) in an adjacency
 * list), we mark them as "inactive" in O(1) via the `active[]` array.
 * Every algorithm checks `active[v]` before processing vertex v.
 * This makes backtracking in BST/IC solvers cheap — just flip the bit back.
 */

#include <vector>
#include <utility>
#include <stdexcept>

class GraphBase
{
public:
    int n;                    ///< Number of vertices (fixed at construction)
    std::vector<bool> active; ///< active[v] = true means vertex v is in graph

    /**
     * @param n Number of vertices (0-indexed: 0 … n-1)
     */
    explicit GraphBase(int n) : n(n), active(n, true) {}

    virtual ~GraphBase() = default;

    /**
     * Logically removes vertex v from the graph.
     * O(1). Does NOT update adjacency lists — caller must do that.
     */
    virtual void deactivate(int v)
    {
        if (v < 0 || v >= n)
            throw std::out_of_range("deactivate: invalid vertex");
        active[v] = false;
    }

    /**
     * Restores a previously deactivated vertex.
     * O(1). Does NOT restore edges — caller must do that.
     */
    virtual void reactivate(int v)
    {
        if (v < 0 || v >= n)
            throw std::out_of_range("reactivate: invalid vertex");
        active[v] = true;
    }

    /// @return true if vertex v is currently in the graph
    bool is_active(int v) const { return v >= 0 && v < n && active[v]; }

    /// @return degree of vertex v in the current (active) graph
    virtual int degree(int v) const = 0;

    /// Add a directed or undirected edge between u and v
    virtual void add_edge(int u, int v) = 0;

    /// Remove edge between u and v (must exist)
    virtual void remove_edge(int u, int v) = 0;
};