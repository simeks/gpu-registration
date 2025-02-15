#pragma once

#include <stk/math/types.h>

/// Interface for using the GCO graph cut solver.

#if defined(__GNUC__) || defined(__clang__)
    #define GCC_VERSION (__GNUC__ * 10000 \
        + __GNUC_MINOR__ * 100 \
        + __GNUC_PATCHLEVEL__)

    // Older versions of GCC do not support push/pop
    #if GCC_VERSION > 40604 // 4.6.4
        #pragma GCC diagnostic push
        //#pragma GCC diagnostic ignored "-Wdeprecated-register"
    #endif

    #pragma GCC diagnostic ignored "-Wparentheses"
#else
    #pragma warning(push)
    #pragma warning(disable: 4512)
    #pragma warning(disable: 4100)
    #pragma warning(disable: 4189)
    #pragma warning(disable: 4701)
    #pragma warning(disable: 4706)
    #pragma warning(disable: 4463)
#endif
namespace gco
{
    // Prevent breaking the build in C++17, where register was removed.
    // The keyword is used within GCO, no idea why, since it is a few
    // decades that it is "exactly as meaningful as whitespace" (cit).
    //#define register

    #include <gco/energy.h>
    #include <gco/graph.cpp>
    #include <gco/maxflow.cpp>
}
#if defined(__GNUC__) || defined(__clang__)
    #if GCC_VERSION > 40604 // 4.6.4
        #pragma GCC diagnostic pop
    #endif
#else
    #pragma warning(pop)
#endif


template<typename T>
class GraphCut
{
public:
    GraphCut(const int3& size);
    ~GraphCut();

    void add_term1(const int3& p, T e0, T e1);
    void add_term1(int x, int y, int z, T e0, T e1);

    void add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11);
    void add_term2(int x1, int y1, int z1,
                   int x2, int y2, int z2,
                   T e00, T e01, T e10, T e11);

    T minimize();

    int get_var(const int3& p);
    int get_var(int x, int y, int z);

private:
    int get_index(int x, int y, int z) const;
    int get_index(const int3& p) const;

    gco::Energy<T, T, T> _e;
    int3 _size;
};

template<typename T>
GraphCut<T>::GraphCut(const int3& size) :
    _e(size.x * size.y * size.z, size.x * size.y * size.z * 3),
    _size(size)
{
    _e.add_variable(size.x * size.y * size.z);
}
template<typename T>
GraphCut<T>::~GraphCut()
{

}

template<typename T>
void GraphCut<T>::add_term1(const int3& p, T e0, T e1)
{
    int index = get_index(p);
    _e.add_term1(index, e0, e1);
}
template<typename T>
void GraphCut<T>::add_term1(int x, int y, int z, T e0, T e1)
{
    int index = get_index(x, y, z);
    _e.add_term1(index, e0, e1);
}

template<typename T>
void GraphCut<T>::add_term2(const int3& p1, const int3& p2, T e00, T e01, T e10, T e11)
{
    int i1 = get_index(p1);
    int i2 = get_index(p2);
    _e.add_term2(i1, i2, e00, e01, e10, e11);
}
template<typename T>
void GraphCut<T>::add_term2(
    int x1, int y1, int z1,
    int x2, int y2, int z2,
    T e00, T e01, T e10, T e11)
{
    int i1 = get_index(x1, y1, z1);
    int i2 = get_index(x2, y2, z2);
    _e.add_term2(i1, i2, e00, e01, e10, e11);
}

template<typename T>
T GraphCut<T>::minimize()
{
    return _e.minimize();
}
template<typename T>
int GraphCut<T>::get_var(const int3& p)
{
    int index = get_index(p.x, p.y, p.z);
    return _e.get_var(index);
}
template<typename T>
int GraphCut<T>::get_var(int x, int y, int z)
{
    int index = get_index(x, y, z);
    return _e.get_var(index);
}

template<typename T>
int GraphCut<T>::get_index(int x, int y, int z) const
{
    return x + y*_size.x + z*_size.x*_size.y;
}
template<typename T>
int GraphCut<T>::get_index(const int3& p) const
{
    return p.x + p.y*_size.x + p.z*_size.x*_size.y;
}
