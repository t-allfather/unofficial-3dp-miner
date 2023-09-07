#include <iostream>
#include <fstream>
#include "buffer_geometry.h"
#include "sphere.h"
#include "scrape.h"
// #include "kernel.cuh"
#include "rock.h"
#include "perlin.h"
#include "simple_deque.h"
#include "narray.h"
#include <chrono>
#include <unistd.h>
#include <deque>
#include "grid.h"
#include "json.h"
#include "Http.h"
#include "main.h"
#include "utils.h"
#include "sha3.h"
#include "math_utils.h"
#include "uint256_t.h"
#include <map>
#include <algorithm>
#include <random>
#include <sstream>
#include "sha256_simd.h"

#include "kernel.cuh"
#include "sha3_gpu.cuh"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#ifndef WS_LIB
#define WS_LIB
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#endif
#else

#include <strings.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>

#endif

using namespace std::chrono;
using namespace std;

int hashrateTotalLast=0;

int gpus = 0;
int gpu_threads = 256;
int gpu_blocks = 64;

std::string host;


int rnd_gpu_step = 0;
vector<int> devices;
cudaDeviceProp devInfo[128];
int useGpu[128] = {0};

int hashLogEnabled = 1;
bool count_dup = false;
int dups = 0;

CpuInfo cpuInfo;
vector<thread> threads;
unsigned long long hashesPerThread[1024];
unsigned long long hashesPerThreadBase[1024];
unsigned long long nullsPerThread[1024];

int blocks_found = 0;

struct GpuJobs
{
    unsigned char *pre_hash_device;
    unsigned char *best_hash_device;
    unsigned char *diffBytes_device;

    int size = 0;
};

struct GpuInfo
{
    int blocks;
    int threads;
};

GpuInfo gpuInfos[16];
int gpusCount = 0;

std::string meta;
unsigned char metaBytes[96];
unsigned char pre_hash[32];
unsigned char best_hash[32];
unsigned char diffBytes[32];
int metaOk = 0;

int TEST_RUN = 0;
int BENCH = 0;

int sphere_stacks = 13;
int sphere_slices = 19;
double sphere_radius = 1.0;
int sphereMode = SPHERE_NORMAL;
bool use_perlin = false;
int use_official_shape = 0;
bool use_avx2 = false;
bool avx2_par = false;
bool use_gpu_sha3 = false;
bool use_gpu = false;

bool test_hashrate = false;

bool logInit = false;
int update_interval = 200;

CmdArgs args;

namespace Color
{
    enum Code
    {
        FG_RED = 31,
        FG_YELLOW = 33,
        FG_PURPLE = 35,
        FG_GREEN = 32,
        FG_BLUE = 34,
        FG_DEFAULT = 39,
        BG_RED = 41,
        BG_GREEN = 42,
        BG_BLUE = 44,
        BG_DEFAULT = 49
    };
    class Modifier
    {
        Code code;

    public:
        Modifier(Code pCode) : code(pCode) {}
        friend std::ostream &
        operator<<(std::ostream &os, const Modifier &mod)
        {
            return os << "\033[" << mod.code << "m";
        }
    };
}

void mainAddThread(void (*f)())
{
    threads.push_back(thread((*f)));
}

std::vector<miner_log_item> log_vector;
void miner_log_main_thread(string tag, string text, string color = "def")
{
    miner_log_item item;
    item.tag = tag;
    item.text = text;
    item.color = color;
    log_vector.push_back(item);
}
void log(string tag, string text, string color = "def")
{
    if (silent)
        return;
#ifdef _WIN32
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    if (tag == "INFO")
    {
        cout << "[";
        SetConsoleTextAttribute(hConsole, FOREGROUND_BLUE);
        cout << tag;
        SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
        cout << "] " << text << endl;
    }
    else if (tag == "ERROR")
    {
        cout << "[";
        SetConsoleTextAttribute(hConsole, FOREGROUND_RED);
        cout << tag;
        SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
        cout << "] " << text << endl;
    }
    else if (tag == "WARN" || tag == "WARNING")
    {
        cout << "[";
        SetConsoleTextAttribute(hConsole, 0x0006);
        cout << tag;
        SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
        cout << "] " << text << endl;
    }
    else if (color != "def")
    {
        if (color == "green")
        {
            cout << "[";
            SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN);
            cout << tag;
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            cout << "] " << text << endl;
        }
        else if (color == "purple")
        {
            cout << "[";
            SetConsoleTextAttribute(hConsole, 0x0005);
            cout << tag;
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            cout << "] " << text << endl;
        }
        else if (color == "red")
        {
            cout << "[";
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED);
            cout << tag;
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            cout << "] " << text << endl;
        }
        else if (color == "yellow")
        {
            cout << "[";
            SetConsoleTextAttribute(hConsole, 0x0006);
            cout << tag;
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            cout << "] " << text << endl;
        }
    }
    else
    {
        cout << "[" + tag + "] " + text << endl;
    }
#else

    Color::Modifier red(Color::FG_RED);
    Color::Modifier yellow(Color::FG_YELLOW);
    Color::Modifier purple(Color::FG_PURPLE);
    Color::Modifier def(Color::FG_DEFAULT);
    Color::Modifier blue(Color::FG_BLUE);
    Color::Modifier green(Color::FG_GREEN);
    if (tag == "INFO")
    {
        cout << "[" << blue << tag << def << "] " << text << endl;
    }
    else if (tag == "ERROR")
    {
        cout << "[" << red << tag << def << "] " << text << endl;
    }
    else if (tag == "WARN" || tag == "WARNING")
    {
        cout << "[" << yellow << tag << def << "] " << text << endl;
    }
    else if (color != "def")
    {
        if (color == "green")
        {
            cout << "[" << green << tag << def << "] " << text << endl;
        }
        else if (color == "purple")
        {
            cout << "[" << purple << tag << def << "] " << text << endl;
        }
        else if (color == "red")
        {
            cout << "[" << red << tag << def << "] " << text << endl;
        }
        else if (color == "yellow")
        {
            cout << "[" << yellow << tag << def << "] " << text << endl;
        }
    }
    else
    {
        cout << "[" << tag << "] " << text << endl;
    }
#endif // _WIN32
}

double GetRandomValue(double min, double max)
{
    return (rand() % 100000000) / 100000000.0 * (max - min) + min;
}

double VaryParameter(double param, double variance, double min, double max)
{
    param += GetRandomValue(-variance * 1.0, +variance * 1.0);
    if (param > max)
        param = max;
    if (param < min)
        param = min;
    return param;
}
int VaryParameter(int param, int variance, int min, int max)
{
    param += (int)GetRandomValue(-variance * 1.0, +variance * 1.0);
    if (param > max)
        param = max;
    if (param < min)
        param = min;
    return param;
}

void varyMesh(RockObjParams &rock_obj_params)
{

    /*
    rock_obj_params.meshNoiseScale     = VaryParameter(1.0,     0.05,      MESH_NOISE_SCALE_MIN,      1.0);
        rock_obj_params.meshNoiseStrength  = VaryParameter(0.5,  0.3,   MESH_NOISE_STRENGTH_MIN,    0.5);
        rock_obj_params.scrapeCount        = VaryParameter(7,        3,          SCRAPE_COUNT_MIN,           SCRAPE_COUNT_MAX);
        rock_obj_params.scrapeMinDist      = VaryParameter(0.8,      SCRAPE_MIN_DIST_VARY,       SCRAPE_MIN_DIST_MIN,        SCRAPE_MIN_DIST_MAX);
        rock_obj_params.scrapeStrength     = VaryParameter(0.05,     0.02,       SCRAPE_STRENGTH_MIN,        SCRAPE_STRENGTH_MAX);
        rock_obj_params.scrapeRadius       = VaryParameter(0.1,       SCRAPE_RADIUS_VARY,         SCRAPE_RADIUS_MIN,          0.5);
    rock_obj_params.scale[0] = VaryParameter(1.0, 0.1, SCALE_MIN, SCALE_MAX);
    rock_obj_params.scale[1] = VaryParameter(1.0, 0.1, SCALE_MIN, SCALE_MAX);
    rock_obj_params.scale[2] = VaryParameter(1.2, 0.1, SCALE_MIN, 1.2);

    */

    // Good

    rock_obj_params.meshNoiseScale = VaryParameter(1.0, 0.05, MESH_NOISE_SCALE_MIN, 1.0);
    rock_obj_params.meshNoiseStrength = VaryParameter(0.5, 0.3, MESH_NOISE_STRENGTH_MIN, 0.5);
    rock_obj_params.scrapeCount = VaryParameter(7, 3, SCRAPE_COUNT_MIN, SCRAPE_COUNT_MAX);
    rock_obj_params.scrapeMinDist = VaryParameter(0.8, SCRAPE_MIN_DIST_VARY, SCRAPE_MIN_DIST_MIN, SCRAPE_MIN_DIST_MAX);
    rock_obj_params.scrapeStrength = VaryParameter(0.05, 0.02, SCRAPE_STRENGTH_MIN, SCRAPE_STRENGTH_MAX);
    rock_obj_params.scrapeRadius = VaryParameter(0.1, SCRAPE_RADIUS_VARY, SCRAPE_RADIUS_MIN, 0.5);
    rock_obj_params.scale[0] = VaryParameter(1.0, 0.1, SCALE_MIN, SCALE_MAX);
    rock_obj_params.scale[1] = VaryParameter(1.0, 0.1, SCALE_MIN, SCALE_MAX);
    rock_obj_params.scale[2] = VaryParameter(1.2, 0.1, SCALE_MIN, 1.2);
}

int totalRocks = 0;
int nullRocks = 0;
int repeats = 0;
std::map<std::string, int> frecv_hash;
int usePrecalcLines = 0;

RockObjParams random_hash(unsigned char *trans_, int trans_len, Vec3Float64 *positions_out, unsigned int *indicies_out, Vec3Float64 *normals_out, unsigned char *output_hash, int &output_len, int &positions_out_len, int &indicies_out_len, std::mt19937 &m_gen, std::uniform_int_distribution<uint32_t> &m_distribution, RockObjParams *pre_rock_obj_params = NULL, Sphere *spherePrecompute = NULL, CellRet *adjacentVerticesPrecompute = NULL)
{
    short grid_size = 8;
    short n_sections = 12;
    unsigned char trans[4] = {0, 0, 0, 0};
    bool trans_null = true;
    if (trans_len == 4)
    {
        trans_null = false;
        trans[0] = trans_[0];
        trans[1] = trans_[1];
        trans[2] = trans_[2];
        trans[3] = trans_[3];
    }
    RockObjParams rock_obj_params;
    if (pre_rock_obj_params == NULL)
    {
        varyMesh(rock_obj_params);
    }
    else
    {
        rock_obj_params = *pre_rock_obj_params;
    }

    /*
    cout << "scrapeCount: " << rock_obj_params.scrapeCount << endl;
    cout << "scrapeMinDist: " << rock_obj_params.scrapeMinDist << endl;
    cout << "scrapeStrength: " << rock_obj_params.scrapeRadius << endl;
    cout << "meshNoiseStrength: " << rock_obj_params.meshNoiseStrength << endl;
    cout << "meshNoiseScale: " << rock_obj_params.meshNoiseScale << endl;
    cout << "scale 0: "<< rock_obj_params.scale[0] << endl;
    cout << "scale 1: "<< rock_obj_params.scale[1] << endl;
    cout << "scale 2: "<< rock_obj_params.scale[2] << endl;
    */

    Sphere sphereStack;
    Sphere *sphere = &sphereStack;
    if (spherePrecompute == NULL)
    {
        CreateSphere(sphere, sphere_radius, sphere_stacks, sphere_slices, sphereMode);
    }
    else
    {
        sphere->len = spherePrecompute->len;
        sphere->len_indicies = spherePrecompute->len_indicies;

        memcpy(sphere->indices, spherePrecompute->indices, sizeof(Vec3Uint) * sphere->len_indicies);
        memcpy(sphere->normals, spherePrecompute->normals, sizeof(Vec3Float64) * sphere->len);
        memcpy(sphere->vertices, spherePrecompute->vertices, sizeof(Vec3Float64) * sphere->len);
    }

        CellRet adjacentVertices;
        if (adjacentVerticesPrecompute == NULL)
        {
            GetNeighbours(sphere->len, sphere->len_indicies, sphere->indices, &adjacentVertices);
        }
        else
        {
            adjacentVertices.size = adjacentVerticesPrecompute->size;
            adjacentVertices.size2 = adjacentVerticesPrecompute->size2;
            adjacentVertices.vec = adjacentVerticesPrecompute->vec;
            // memcpy(&adjacentVertices,adjacentVerticesPrecompute,sizeof(CellRet));
        }

        unsigned int scrapeIndices[33];
        int scrapeIndicesLen = 0;

        for (int i = 0; i < rock_obj_params.scrapeCount; i++)
        {
            int attempts = 0;

            // find random position which is not too close to the other positions.
            while (true)
            {
                int randIndex = m_distribution(m_gen) % sphere->len;
                Vec3Float64 p = sphere->vertices[randIndex];

                bool tooClose = false;
                // check that it is not too close to the other vertices.
                for (int j = 0; j < scrapeIndicesLen; j++)
                {
                    Vec3Float64 q = sphere->vertices[scrapeIndices[j]];

                    double dist = (p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z);
                    if (dist < rock_obj_params.scrapeMinDist)
                    {
                        tooClose = true;
                        break;
                    }
                }
                attempts = attempts + 1;

                // if we have done too many attempts, we let it pass regardless.
                // otherwise, we risk an endless loop.
                if (tooClose && attempts < 100)
                {
                    continue;
                }
                else
                {
                    scrapeIndices[scrapeIndicesLen] = randIndex;
                    scrapeIndicesLen++;
                    break;
                }
            }
        }

        bool *traversed = (bool *)malloc(sphere->len);

        if (pre_rock_obj_params == NULL)
        {
            rock_obj_params.scrapeIndicesLen = scrapeIndicesLen;
            memcpy(rock_obj_params.scrapeIndices, scrapeIndices, scrapeIndicesLen * 4);
        }
        else
        {
            scrapeIndicesLen = rock_obj_params.scrapeIndicesLen;
            memcpy(scrapeIndices, rock_obj_params.scrapeIndices, scrapeIndicesLen * 4);
        }

        std::deque<int> stack;

        // now we scrape at all the selected positions.
        for (int i = 0; i < scrapeIndicesLen; i++)
        {
            memset(traversed, 0, sphere->len);
            stack.clear();
            scrapeMainStd(scrapeIndices[i], sphere->vertices, sphere->normals, &adjacentVertices, rock_obj_params.scrapeStrength, rock_obj_params.scrapeRadius, traversed, stack);
        }

        if (adjacentVerticesPrecompute == NULL)
            free(adjacentVertices.vec);
        free(traversed);
    

    if (use_perlin == false)
    {

        for (int i = 0; i < sphere->len; i++)
        {
            Vec3Float64 &pI = sphere->vertices[i];
            pI.x *= rock_obj_params.scale[0];
            pI.y *= rock_obj_params.scale[1];
            pI.z *= rock_obj_params.scale[2];
        }
    }
    else
    {
        Perlin perlin;
        perlin.Init(m_distribution(m_gen) % INT32_MAX);
        for (int i = 0; i < sphere->len; i++)
        {
            Vec3Float64 p = sphere->vertices[i];
            double noise = rock_obj_params.meshNoiseStrength * perlin.Noise(rock_obj_params.meshNoiseScale * p.x, rock_obj_params.meshNoiseScale * p.y, rock_obj_params.meshNoiseScale * p.z);

            Vec3Float64 &pI = sphere->vertices[i];
            pI.x += noise;
            pI.y += noise;
            pI.z += noise;

            pI.x *= rock_obj_params.scale[0];
            pI.y *= rock_obj_params.scale[1];
            pI.z *= rock_obj_params.scale[2];

            double EPSILON = std::numeric_limits<double>::epsilon();

            pI.x = round(((pI.x + EPSILON) * 100.0)) / 100.0;
            pI.y = round(((pI.y + EPSILON) * 100.0)) / 100.0;
            pI.z = round(((pI.z + EPSILON) * 100.0)) / 100.0;
        }
    }

    BufferGeometry geo = BufferGeometry(sphere->vertices, (unsigned int *)sphere->indices, sphere->normals, sphere->len, sphere->len_indicies * 3, sphere);
    geo.fixIndicies();

    if (positions_out != NULL && normals_out != NULL && indicies_out != NULL)
    {
        for (int i = 0; i < geo.len; i++)
        {
            positions_out[i].x = geo.positions[i].x;
            positions_out[i].y = geo.positions[i].y;
            positions_out[i].z = geo.positions[i].z;
            normals_out[i].x = geo.normals[i].x;
            normals_out[i].y = geo.normals[i].y;
            normals_out[i].z = geo.normals[i].z;
        }
        for (int i = 0; i < geo.len_indices; i++)
        {
            indicies_out[i] = geo.indices[i];
        }
        positions_out_len = geo.len;
        indicies_out_len = geo.len_indices;
    }

    Vec3Float64 f1;
    Vec3Float64 f2;
    Vec3Float64 f3;

    Vec3Float64 g0;
    Vec3Float64 g1;
    Vec3Float64 g2;

    Vec3Float64 cross;
    double integral[10];

    double integral_sum[10] = {0};
    double coefficients[] = {1. / 6., 1. / 24., 1. / 24., 1. / 24., 1. / 60., 1. / 60., 1. / 60., 1. / 120., 1. / 120., 1. / 120.};

    for (int i = 0; i < geo.len_indices; i += 3)
    {
        Vec3Float64 tp1 = geo.positions[geo.indices[i + 1]];
        Vec3Float64 tp2 = geo.positions[geo.indices[i + 2]];
        Vec3Float64 tp3 = geo.positions[geo.indices[i + 0]];
        f1.x = tp1.x + tp2.x + tp3.x;
        f1.y = tp1.y + tp2.y + tp3.y;
        f1.z = tp1.z + tp2.z + tp3.z;

        f2.x = tp1.x * tp1.x +
               tp2.x * tp2.x +
               tp1.x * tp2.x +
               tp2.x * f1.x;

        f2.y = tp1.y * tp1.y +
               tp2.y * tp2.y +
               tp1.y * tp2.y +
               tp2.y * f1.y;

        f2.z = tp1.z * tp1.z +
               tp2.z * tp2.z +
               tp1.z * tp2.z +
               tp2.z * f1.z;

        f3.x = tp1.x * tp1.x * tp1.x +
               tp1.x * tp1.x * tp2.x +
               tp1.x * tp2.x * tp2.x +
               tp2.x * tp2.x * tp2.x +
               tp3.x * f2.x;

        f3.y = tp1.y * tp1.y * tp1.y +
               tp1.y * tp1.y * tp2.y +
               tp1.y * tp2.y * tp2.y +
               tp2.y * tp2.y * tp2.y +
               tp3.y * f2.y;

        f3.z = tp1.z * tp1.z * tp1.z +
               tp1.z * tp1.z * tp2.z +
               tp1.z * tp2.z * tp2.z +
               tp2.z * tp2.z * tp2.z +
               tp3.z * f2.z;

        g0.x = f2.x + (tp1.x + f1.x) * tp1.x;
        g0.y = f2.y + (tp1.y + f1.y) * tp1.y;
        g0.z = f2.z + (tp1.z + f1.z) * tp1.z;

        g1.x = f2.x + (tp2.x + f1.x) * tp2.x;
        g1.y = f2.y + (tp2.y + f1.y) * tp2.y;
        g1.z = f2.z + (tp2.z + f1.z) * tp2.z;

        g2.x = f2.x + (tp3.x + f1.x) * tp3.x;
        g2.y = f2.y + (tp3.y + f1.y) * tp3.y;
        g2.z = f2.z + (tp3.z + f1.z) * tp3.z;

        double d1[3];
        d1[0] = tp2.x - tp1.x;
        d1[1] = tp2.y - tp1.y;
        d1[2] = tp2.z - tp1.z;

        double d2[3];
        d2[0] = tp3.x - tp2.x;
        d2[1] = tp3.y - tp2.y;
        d2[2] = tp3.z - tp2.z;

        cross.x = d1[1] * d2[2] - d1[2] * d2[1];
        cross.y = d1[2] * d2[0] - d1[0] * d2[2];
        cross.z = d1[0] * d2[1] - d1[1] * d2[0];

        integral[0] = cross.x * f1.x;

        integral[1] = cross.x * f2.x;
        integral[2] = cross.y * f2.y;
        integral[3] = cross.z * f2.z;

        integral[4] = cross.x * f3.x;
        integral[5] = cross.y * f3.y;
        integral[6] = cross.z * f3.z;

        for (int j = 0; j < 3; j++)
        {
            int triangle_i = (j + 1) % 3;
            if (j == 0)
            {
                integral[7] = cross.x * (tp1.y * g0.x +
                                         tp1.y * g1.x +
                                         tp1.y * g2.x);
            }
            else if (j == 1)
            {
                integral[8] = cross.y * (tp1.z * g0.y +
                                         tp1.z * g1.y +
                                         tp1.z * g2.y);
            }
            else if (j == 2)
            {
                integral[9] = cross.z * (tp1.x * g0.z +
                                         tp1.x * g1.z +
                                         tp1.x * g2.z);
            }
        }

        for (int j = 0; j < 10; j++)
            integral_sum[j] += integral[j];
    }

    double integrated[10];
    for (int j = 0; j < 10; j++)
        integrated[j] = integral_sum[j] * coefficients[j];
    double volume = integrated[0];
    double center_mass[3];
    if (volume > 0.0001)
    {
        center_mass[0] = integrated[1] / volume;
        center_mass[1] = integrated[2] / volume;
        center_mass[2] = integrated[3] / volume;
    }
    else
    {
        center_mass[0] = 0;
        center_mass[1] = 0;
        center_mass[2] = 0;
    }

    double inertia[3][3];

    inertia[0][0] = integrated[5] + integrated[6] - volume * (center_mass[1] * center_mass[1] + center_mass[2] * center_mass[2]);

    inertia[1][1] = integrated[4] + integrated[6] -
                    volume * (center_mass[0] * center_mass[0] + center_mass[2] * center_mass[2]);

    inertia[2][2] = integrated[4] + integrated[5] -
                    volume * (center_mass[0] * center_mass[0] + center_mass[1] * center_mass[1]);

    inertia[0][1] = integrated[7] -
                    volume * center_mass[0] * center_mass[1];

    inertia[1][2] = integrated[8] -
                    volume * center_mass[1] * center_mass[2];

    inertia[0][2] = integrated[9] -
                    volume * center_mass[0] * center_mass[2];

    inertia[2][0] = inertia[0][2];
    inertia[2][1] = inertia[1][2];
    inertia[1][0] = inertia[0][1];

    double m[3][3] = {{1 * inertia[0][0], -1 * inertia[0][1], -1 * inertia[0][2]},
                      {-1 * inertia[1][0], 1 * inertia[1][1], -1 * inertia[1][2]},
                      {-1 * inertia[2][0], -1 * inertia[2][1], 1 * inertia[2][2]}};

    Eigen eigen;
    eigen.Create(m);
    eigen.Solve();

    double components[3] = {
        eigen.d[1],
        eigen.d[0],
        eigen.d[2]};
    double vectors[3][3] = {
        eigen.v[0][0],
        eigen.v[1][0],
        eigen.v[2][0],
        eigen.v[0][1],
        eigen.v[1][1],
        eigen.v[2][1],
        eigen.v[0][2],
        eigen.v[1][2],
        eigen.v[2][2],
    };

    double pit[4][4] = {
        vectors[0][0], vectors[0][1], vectors[0][2], -center_mass[0],
        vectors[1][0], vectors[1][1], vectors[1][2], -center_mass[1],
        vectors[2][0], vectors[2][1], vectors[2][2], -center_mass[2],
        0, 0, 0, 1};

    double b[4][4]; // pit matrix inverse = pit ^ -1
    double bb[4][4];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            b[j][i] = pit[i][j];

    double determinant = pit[0][0] * (pit[1][1] * pit[2][2] - pit[2][1] * pit[1][2]) -
                         pit[0][1] * (pit[1][0] * pit[2][2] - pit[1][2] * pit[2][0]) +
                         pit[0][2] * (pit[1][0] * pit[2][1] - pit[1][1] * pit[2][0]);

    double invdet = 1 / determinant;

    b[0][0] = (pit[1][1] * pit[2][2] - pit[2][1] * pit[1][2]) * invdet;
    b[0][1] = (pit[0][2] * pit[2][1] - pit[0][1] * pit[2][2]) * invdet;
    b[0][2] = (pit[0][1] * pit[1][2] - pit[0][2] * pit[1][1]) * invdet;
    b[1][0] = (pit[1][2] * pit[2][0] - pit[1][0] * pit[2][2]) * invdet;
    b[1][1] = (pit[0][0] * pit[2][2] - pit[0][2] * pit[2][0]) * invdet;
    b[1][2] = (pit[1][0] * pit[0][2] - pit[0][0] * pit[1][2]) * invdet;
    b[2][0] = (pit[1][0] * pit[2][1] - pit[2][0] * pit[1][1]) * invdet;
    b[2][1] = (pit[2][0] * pit[0][1] - pit[0][0] * pit[2][1]) * invdet;
    b[2][2] = (pit[0][0] * pit[1][1] - pit[1][0] * pit[0][1]) * invdet;

    b[0][3] = 0;
    b[1][3] = 0;
    b[2][3] = 0;
    b[3][3] = 1;
    b[3][0] = 0;
    b[3][1] = 0;
    b[3][2] = 0;

    if (trans_null == false)
    {
        double v[4];
        for (int i = 0; i < 4; i++)
        {
            double vl = ((double)trans[i]);
            v[i] = vl * 45.0 / 256.0;
        }
        Vec3Float64 axis;
        axis.x = v[0];
        axis.y = v[1];
        axis.z = v[2];
        double n = 1.0 / sqrt(Vec3Float64::dot(&axis, &axis));
        axis.x = axis.x * n;
        axis.y = axis.y * n;
        axis.z = axis.z * n;
        v[3] = v[3] * 360.0 / 256.0;

        double sn, cs;
        sn = sin(v[3] * M_PI / 180.0);
        cs = cos(v[3] * M_PI / 180.0);
        double _lsubc = 1.0 - cs;
        bb[0][0] = _lsubc * axis.x * axis.x + cs;
        bb[0][1] = _lsubc * axis.x * axis.y + sn * axis.z;
        bb[0][2] = _lsubc * axis.x * axis.z - sn * axis.y;
        bb[0][3] = 0;

        bb[1][0] = _lsubc * axis.x * axis.y - sn * axis.z;
        bb[1][1] = _lsubc * axis.y * axis.y + cs;
        bb[1][2] = _lsubc * axis.y * axis.z + sn * axis.x;
        bb[1][3] = 0;

        bb[2][0] = _lsubc * axis.x * axis.z + sn * axis.y;
        bb[2][1] = _lsubc * axis.y * axis.z - sn * axis.x;
        bb[2][2] = _lsubc * axis.z * axis.z + cs;
        bb[2][3] = 0;

        bb[3][0] = 0;
        bb[3][1] = 0;
        bb[3][2] = 0;
        bb[3][3] = 1;
    }

    Vec3Float64 v_min, v_max;
    v_min.x = 10000000;
    v_min.y = 10000000;
    v_min.z = 10000000;
    v_max.x = -10000000;
    v_max.y = -10000000;
    v_max.z = -10000000;

    // Shift + Translate + Rotate
    for (int i = 0; i < geo.len; i++)
    {

        geo.positions[i].x = geo.positions[i].x + pit[0][3];
        geo.positions[i].y = geo.positions[i].y + pit[1][3];
        geo.positions[i].z = geo.positions[i].z + pit[2][3];

        double t1 =
            geo.positions[i].x * b[0][0] +
            geo.positions[i].y * b[1][0] +
            geo.positions[i].z * b[2][0] +
            1 * b[3][0];
        double t2 =
            geo.positions[i].x * b[0][1] +
            geo.positions[i].y * b[1][1] +
            geo.positions[i].z * b[2][1] +
            1 * b[3][1];
        double t3 =
            geo.positions[i].x * b[0][2] +
            geo.positions[i].y * b[1][2] +
            geo.positions[i].z * b[2][2] +
            1 * b[3][2];

        if (trans_null == false)
        {

            double tt1 =
                t1 * bb[0][0] +
                t2 * bb[1][0] +
                t3 * bb[2][0] +
                1 * bb[3][0];
            double tt2 =
                t1 * bb[0][1] +
                t2 * bb[1][1] +
                t3 * bb[2][1] +
                1 * bb[3][1];
            double tt3 =
                t1 * bb[0][2] +
                t2 * bb[1][2] +
                t3 * bb[2][2] +
                1 * bb[3][2];

            if (tt1 > v_max.x)
                v_max.x = tt1;
            if (tt1 < v_min.x)
                v_min.x = tt1;

            if (tt2 > v_max.y)
                v_max.y = tt2;
            if (tt2 < v_min.y)
                v_min.y = tt2;

            if (tt3 > v_max.z)
                v_max.z = tt3;
            if (tt3 < v_min.z)
                v_min.z = tt3;

            geo.positions[i].x = tt1;
            geo.positions[i].y = tt2;
            geo.positions[i].z = tt3;
        }
        else
        {
            if (t1 > v_max.x)
                v_max.x = t1;
            if (t1 < v_min.x)
                v_min.x = t1;

            if (t2 > v_max.y)
                v_max.y = t2;
            if (t2 < v_min.y)
                v_min.y = t2;

            if (t3 > v_max.z)
                v_max.z = t3;
            if (t3 < v_min.z)
                v_min.z = t3;

            geo.positions[i].x = t1;
            geo.positions[i].y = t2;
            geo.positions[i].z = t3;
        }
    }

    double step = (v_max.z - v_min.z) / (1.0 + n_sections);
    Vec2F64 cntr[1024];
    int cntrs_len[13] = {0};
    for (int x = 0; x < 13; x++)
        cntrs_len[x] = 0;
    int psum = 0;
    int goFurther = 1;

    for (int n = 0; n < n_sections; n++)
    {
        double z_sect = v_min.z + (n + 1.0) * step;
        int cntr_len = 0;

        get_contour_opt(&geo, z_sect, cntr + psum, cntr_len);
        if (cntr_len == 0)
        {
            goFurther = 0;
            break;
        }
        cntrs_len[n + 1] = cntr_len;
        psum += cntr_len;
    }

    if (goFurther)
    {
        unsigned char out_hash[512];
        int out_hash_len = 0;
        find_top_std_3(cntr, cntrs_len, 10, n_sections, grid_size, v_min, v_max, out_hash, out_hash_len);

        totalRocks++;
        if (out_hash_len < 0)
        {
            output_len = -1;
        }
        else if (out_hash_len == 0)
        {
            output_len = 0;
        }
        else
        {
            memcpy(output_hash, out_hash, 32);
            output_len = 32;
        }
        if (out_hash_len <= 0)
        {
            nullRocks++;
        }

        if (BENCH == 1)
        {
            for (int i = 0; i < 1 && i < out_hash_len; i++)
            {
                string hs = "";
                for (int j = 0; j < 32; j++)
                {
                    unsigned char c = out_hash[i * 32 + j];
                    char c1, c2;
                    c1 = c / 16;
                    if (c1 < 10)
                    {
                        c1 = '0' + c1;
                    }
                    else
                    {
                        c1 = 'a' + (c1 - 10);
                    }
                    c2 = c % 16;
                    if (c2 < 10)
                    {
                        c2 = '0' + c2;
                    }
                    else
                    {
                        c2 = 'a' + (c2 - 10);
                    }
                    hs += c1;
                    hs += c2;
                }

                if (frecv_hash.find(hs) != frecv_hash.end())
                {
                    repeats++;
                }
                else
                {
                    frecv_hash.insert(make_pair(hs, 0));
                }
            }
        }
    }
    else
    {
        output_len = 0;
        totalRocks++;
        nullRocks++;
    }

    return rock_obj_params;
}

int tests = 5000;
void benchmarkThread(uint64_t threadId, int &count, int &v)
{
    bool status = setThreadAffinity(cpuInfo.affinity[threadId]);
    if (status == false)
    {
        if (!silent)
            cout << "Failed to set affinity at thread " << threadId << endl;
        return;
    }
    Vec3Float64 positions[602];
    Vec3Float64 normals[602];
    unsigned int indicies[1806 * 3];
    unsigned char hash[32];
    int hash_len = 0;
    int a, b;
    std::mt19937 m_gen;
    std::uniform_int_distribution<uint32_t> m_distribution{0, std::numeric_limits<uint32_t>::max()};
    std::uniform_int_distribution<uint64_t> m_distribution_ll{0, std::numeric_limits<uint64_t>::max()};

    std::chrono::system_clock::time_point now_time = std::chrono::system_clock::now();
    auto epoch = now_time.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count() % 0xfffffffffffffffeULL;

    unsigned char aaa[4] = {54, 126, 226, 199};

    int aa = 0;
    int rnd = 0;
    if (args.get("rand-rotation") != "NULL_ARG")
    {
        aa = 4;
        rnd = 1;
    }
    if (args.get("x") != "NULL_ARG")
    {
        aaa[0] = tryParseInt(args.get("x"));
        aa++;
    }
    if (args.get("y") != "NULL_ARG")
    {
        aaa[1] = tryParseInt(args.get("y"));
        aa++;
    }
    if (args.get("z") != "NULL_ARG")
    {
        aaa[2] = tryParseInt(args.get("z"));
        aa++;
    }
    if (args.get("a") != "NULL_ARG")
    {
        aaa[3] = tryParseInt(args.get("a"));
        aa++;
    }
    m_gen = std::mt19937(us);

    Sphere sphere;
    CellRet adjacentVertices;
    CreateSphere(&sphere, sphere_radius, sphere_stacks, sphere_slices, SPHERE_NORMAL);
    adjacentVertices.size = sphere.len + 1;

    GetNeighbours(sphere.len, sphere.len_indicies, sphere.indices, &adjacentVertices);

    auto t1 = getTime();
    for (int i = 0; i < tests; i++)
    {
        if (aa == 4 && rnd == 1)
        {
            aaa[0] = rand() % 256;
            aaa[1] = rand() % 256;
            aaa[2] = rand() % 256;
            aaa[3] = rand() % 256;
        }
        random_hash(aaa, aa, NULL, NULL, NULL, hash, hash_len, a, b, m_gen, m_distribution, NULL, &sphere, &adjacentVertices);
    }
    auto t2 = getTime();
    double d = tests / ((double)getMs(t1, t2) / 1000);
    count += d;
    if (!silent)
    {
        log("Thread " + to_string(threadId), to_string(d) + " H/s", "green");
    }
    v++;
}

void benchmarkThreadGpu(uint64_t threadId, int &count, int &v)
{

    Vec3Float64 positions[602];
    Vec3Float64 normals[602];
    unsigned int indicies[1806 * 3];
    unsigned char hash[32];
    int hash_len = 0;
    int a, b;
    std::mt19937 m_gen;
    std::uniform_int_distribution<uint32_t> m_distribution{0, std::numeric_limits<uint32_t>::max()};
    std::uniform_int_distribution<uint64_t> m_distribution_ll{0, std::numeric_limits<uint64_t>::max()};

    std::chrono::system_clock::time_point now_time = std::chrono::system_clock::now();
    auto epoch = now_time.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count() % 0xfffffffffffffffeULL;

    unsigned char aaa[4] = {54, 126, 226, 199};

    int aa = 0;
    int rnd = 0;
    if (args.get("rand-rotation") != "NULL_ARG")
    {
        aa = 4;
        rnd = 1;
    }
    if (args.get("x") != "NULL_ARG")
    {
        aaa[0] = tryParseInt(args.get("x"));
        aa++;
    }
    if (args.get("y") != "NULL_ARG")
    {
        aaa[1] = tryParseInt(args.get("y"));
        aa++;
    }
    if (args.get("z") != "NULL_ARG")
    {
        aaa[2] = tryParseInt(args.get("z"));
        aa++;
    }
    if (args.get("a") != "NULL_ARG")
    {
        aaa[3] = tryParseInt(args.get("a"));
        aa++;
    }
    m_gen = std::mt19937(us);

    Sphere sphere;
    CellRet adjacentVertices;
    CreateSphere(&sphere, sphere_radius, sphere_stacks, sphere_slices, SPHERE_NORMAL);
    adjacentVertices.size = sphere.len + 1;

    GetNeighbours(sphere.len, sphere.len_indicies, sphere.indices, &adjacentVertices);

    initGpuData(threadId, gpuInfos[threadId].blocks, gpuInfos[threadId].threads, sphere_stacks, sphere_slices);

    int total_tests = gpuInfos[threadId].blocks * gpuInfos[threadId].threads * tests;

    unsigned char *outhash = new unsigned char[total_tests / tests * 32];
    int *outlen = new int[total_tests / tests];

    map<string, int> mm;
    auto t1 = getTime();

    unsigned char *best_hash = new unsigned char[32];
    unsigned char *pre_hash = new unsigned char[32];
    unsigned char *diffBytes = new unsigned char[32];
    unsigned char *cmpBytes = new unsigned char[32];
    memset(best_hash, 0, 32);
    memset(pre_hash, 0, 32);
    memset(diffBytes, 0, 32);
    memset(cmpBytes, 0, 32);
    for (int i = 0; i < tests; i++)
    {
        doGpuBatch(threadId - 100, gpuInfos[threadId].blocks, gpuInfos[threadId].threads, outhash, outlen, best_hash, pre_hash, diffBytes, cmpBytes);

        for (int j = 0; j < gpuInfos[threadId].blocks * gpuInfos[threadId].threads; j++)
        {
            string h = toHex(outhash + 32 * j, 32);
            if (outlen[j] > 0)
            {
                auto it = mm.find(h);
                if (it != mm.end())
                {
                    repeats++;
                }
                else
                {
                    mm.insert(make_pair(h, 0));
                }
            }
            else
            {
                nullRocks++;
            }
        }
    }

    auto t2 = getTime();
    double d = total_tests / ((double)getMs(t1, t2) / 1000);
    count += d;
    if (!silent)
    {
        log("Thread " + to_string(threadId), to_string(d) + " H/s", "green");
    }
    v++;
}

void benchmark()
{
    BENCH = 1;
    srand(time(NULL));

    if (use_gpu == false)
    {
        int count = 0;
        int count2[512];
        for (int i = 0; i < 512; i++)
            count2[i] = 0;

        if (!silent)
        {
            log("INFO", "Benchmark on " + to_string(cpuInfo.cores) + " threads");
        }
        int v = 0;
        for (int i = 0; i < cpuInfo.cores; i++)
        {
            threads.push_back(thread(benchmarkThread, i, std::ref(count2[i]), std::ref(v)));
        }
        while (v < cpuInfo.cores)
        {
            waitMs(100);
        }

        count = 0;
        for (int i = 0; i < cpuInfo.cores; i++)
        {
            count += count2[i];
        }
        log("Total", to_string(count) + " H/s", "yellow");
        log("Total good", to_string((double)(totalRocks - nullRocks - repeats) / (double)totalRocks * count), "green");
        log("WARN", "Repeats: " + to_string(repeats));
        log("WARN", "Null/Total " + to_string(nullRocks) + "/" + to_string(totalRocks));
        waitMs(100);
        _Exit(0);
    }
    else
    {

        int count = 0;
        int count2[512];
        for (int i = 0; i < 512; i++)
            count2[i] = 0;
        if (!silent)
        {
            log("INFO", "Benchmark on " + to_string(gpusCount) + " GPUs");
        }
        int v = 0;
        for (int i = 0; i < gpusCount; i++)
        {
            threads.push_back(thread(benchmarkThreadGpu, i, std::ref(count2[i]), std::ref(v)));
        }
        while (v < gpusCount)
        {
            waitMs(100);
        }
        count = 0;
        for (int i = 0; i < gpusCount; i++)
        {
            count += count2[i];
        }
        log("Total", to_string(count) + " H/s", "yellow");
        log("Total good", to_string((double)(totalRocks - nullRocks - repeats) / (double)totalRocks * count), "green");
        log("WARN", "Repeats: " + to_string(repeats));
        log("WARN", "Null/Total " + to_string(nullRocks) + "/" + to_string(totalRocks));
        waitMs(100);
        _Exit(0);
    }
}

void help()
{
    printf("Options:\n");
    printf("--host [text] : ip:port\n");
    printf("--use-sha : some optimizations\n");
    printf("--threads : number of cpu threads\n");
    printf("--affinity : cpu thread affinity (advanced)\n");
    printf("--gpu : gpu mining toggle\n");
}

void getNodeMeta()
{
    std::string mth = "poscan_getMeta";
    string sendData = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"" + mth + "\"}";
    HttpHeader header = HttpRequest(host, "POST", "/", sendData, "", "", "");
    string metaNew = header.data();
    Json json = Json(metaNew);
    metaNew = json.get("result");

    unsigned char diffTest[32] = {0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (metaNew != meta)
    {
        meta = metaNew;
        auto b = fromHex(meta);
        int best_hash_loaded = 0;
        if (b.size() >= 96)
        {
            for(int i=0;i<32;i++){
                diffBytes[i] = b[i];
            }
            for(int i=32;i<64;i++){
                pre_hash[i-32] = b[i];
            }
            for(int i=64;i<96;i++){
                best_hash[i-64] = b[i];
            }
            metaOk = 1;
            if (args.get("no-job-info") == "NULL_ARG")
            {
                if (!silent)
                    cout << "New metadata "<<endl;
                if (metaNew.size() <= 96)
                    cout << metaNew << endl;
            }
        }
        else
        {
            metaOk = 0;
            if (!silent)
                cout << "Fail to decode metadata data:" << metaNew << endl;
        }
    }
}

void metaLoop()
{
    while (true)
    {
        getNodeMeta();
        waitMs(update_interval);
    }
}

struct Solution
{
    unsigned char hash[32];
    unsigned char pre[4];
    RockObjParams objParams;
};

void sendGpuSolution(unsigned char *hash, int iterIndex, RockObjParams param, unsigned char *seal, unsigned char *besthash)
{
    Vec3Float64 positions[602];
    Vec3Float64 normals[602];
    unsigned int indicies[1806 * 3];
    unsigned char hashObj[32];
    memset(hashObj, 0, 32);
    int hash_len = 0;
    int pos_len = 0, indicies_len = 0;
    std::mt19937 m_gen;
    std::uniform_int_distribution<uint32_t> m_distribution{0, std::numeric_limits<uint32_t>::max()};
    std::uniform_int_distribution<uint64_t> m_distribution_ll{0, std::numeric_limits<uint64_t>::max()};

    std::chrono::system_clock::time_point now_time = std::chrono::system_clock::now();
    auto epoch = now_time.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count() % 0xfffffffffffffffeULL;

    m_gen = std::mt19937(us);

    random_hash(besthash, 4, positions, indicies, normals, hashObj, hash_len, pos_len, indicies_len, m_gen, m_distribution, &param, NULL, NULL);

    int cmp = 0;
    for (int i = 0; i < 32; i++)
    {
        if (hashObj[i] != hash[i])
        {
            cmp = 1;
            break;
        }
    }

    if (cmp)
    {
        log("ERROR", "Expected object hash is different from actual object hash");
        printf("Expected object hash: ");
        for (int i = 0; i < 32; i++)
        {
            printf("%d ", hash[i]);
        }
        printf("\n");
        printf("Object hash: ");
        for (int i = 0; i < 32; i++)
        {
            printf("%d ", hashObj[i]);
        }
        printf("\n");

        log("WARN", "Still submiting...");
    }

    cout << "Obj hash: " << toHex(hashObj, 32) << endl;
    cout << "Besthash: " << toHex(besthash, 4) << endl;

    std::time_t t = std::time(0); // get time now
    std::tm *now = std::localtime(&t);
    std::cout << (now->tm_year + 1900) << '-'
              << (now->tm_mon + 1) << '-'
              << now->tm_mday
              << " " << now->tm_hour << ":" << now->tm_min << ":" << now->tm_sec
              << "\n";

    miner_log("NICE", "Preparing submit of object to host " + (host), "green");
    string objFile = "";
    BufferGeometry geo(positions, indicies, normals, pos_len, indicies_len);
    geo.ComputeVertexNormals();
    objFile = geo.parse();
    string rpl = "";
    for (int i = 0; i < objFile.size(); i++)
    {
        if (objFile[i] == '\n')
        {
            rpl += "\\n";
        }
        else
        {
            rpl += objFile[i];
        }
    }
    string submit_data = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"poscan_pushMiningObject\",\"params\":[1,\"" + rpl + "\",\"" + toHex(seal, 32) + "\",\"" + toHex(hash, 32) + "\"]}";

    if (test_hashrate == false)
    {
        HttpHeader header = HttpRequest(host, "POST", "/", submit_data, "", "", "");
        string res = header.data();
        miner_log("INFO", "Node response to submit: " + res);
    }
    else
    {
        cout << "Seal -> " << toHex(seal, 32) << endl;
    }
}

std::map<string, int> map_hashes;

void gpuMain(uint64_t threadId)
{
    initGpuData(threadId, gpuInfos[threadId].blocks, gpuInfos[threadId].threads, sphere_stacks, sphere_slices);
    miner_log("GPU #" + to_string(threadId), "Initialized", "green");

    Vec3Float64 *positions = new Vec3Float64[602];
    Vec3Float64 normals[602];
    unsigned int indicies[1806 * 3];
    int pos_len = 0;
    int indicies_len = 0;
    unsigned char hash[32];
    for (int i = 0; i < 32; i++)
    {
        hash[i] = 0;
    }
    int hash_len = 0;
    unsigned char sealPre[64];
    unsigned char seal[32];
    unsigned char hh[96];
    unsigned char hash_final[32];

    std::mt19937 m_gen;
    std::uniform_int_distribution<uint32_t> m_distribution{0, std::numeric_limits<uint32_t>::max()};
    std::uniform_int_distribution<uint64_t> m_distribution_ll{0, std::numeric_limits<uint64_t>::max()};

    std::chrono::system_clock::time_point now_time = std::chrono::system_clock::now();
    auto epoch = now_time.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count() % 0xfffffffffffffffeULL;

    m_gen = std::mt19937(us);

    uint256_t max256 = 1;
    for (int ii = 0; ii < 31; ii++)
    {
        max256 = max256 * 256;
    }
    uint256_t max256_t = max256 - 1;
    max256 = max256 * 255 + max256_t;
    SHA3 sha3_1;

    unsigned char oldDiff[32];
    unsigned char oldDiffDev[32];
    for (int i = 0; i < 32; i++)
    {
        oldDiff[i] = 0;
        oldDiffDev[i] = 0;
    }
    vector<uint8_t> vec;
    vector<uint8_t> vec2;

    Sphere sphere;
    CellRet adjacentVertices;
    CreateSphere(&sphere, sphere_radius, sphere_stacks, sphere_slices, SPHERE_NORMAL);
    adjacentVertices.size = sphere.len + 1;

    GetNeighbours(sphere.len, sphere.len_indicies, sphere.indices, &adjacentVertices);


    while (true)
    {
        int run = -1;
        if(metaOk){
            run = 0;
        }
        if (run == -1)
        {
            waitMs(100);
            continue;
        }
        bool c = false;
        for (int i = 0; i < 32; i++)
        {
            if (oldDiff[i] != diffBytes[i])
            {
                c = true;
                oldDiff[i] = diffBytes[i];
            }
        }
        if (c)
        {
            uint256_t diff = 0;
            unsigned long long pow256 = 1;
            for (int i = 0; i < 8; i++)
            {
                diff = diff + diffBytes[i] * pow256;
                pow256 = pow256 * 256;
            }

            if (diff == 0)
                continue;
            uint256_t a = max256 / diff;

            vec = a.export_bits();
        }



                auto sols = doGpuBatch(threadId, gpuInfos[threadId].blocks, gpuInfos[threadId].threads, NULL, NULL,
                                       best_hash, pre_hash, diffBytes, vec.data());

                hashesPerThread[threadId] += gpuInfos[threadId].threads * gpuInfos[threadId].blocks;

                for (int j = 0; j < sols.size() && j < 1; j++)
                {
                    unsigned char hashObj[32];
                    random_hash(pre_hash, 4, positions, indicies, normals, hashObj, hash_len, pos_len, indicies_len, m_gen, m_distribution, &sols[j].param, NULL, NULL);
                    memcpy(hash, hashObj, 32);

                    /*
68 207 179 82
correct: 85 82 240 218 150 215 131 23 34 120 200 153 40 166 253 204 209 204 62 38 129 16 177 127 169 53 127 84 246 190 26 78
gpu: 85 82 240 218 150 215 131 23 34 120 200 153 40 166 253 204 209 204 62 38 129 16 177 127 169 53 127 84 246 190 26 78
pre_hash = 76c8d7b8e894b992138dabf8a9172e2b3b54e117b421146a5553c47c1a99bf6d
obj_hash = 5552f0da96d783172278c89928a6fdccd1cc3e268110b17fa9357f54f6be1a4e
best_hash = 44cfb35286e098937d96728812e8c147525e8f2df58e34b4464a195b9510607f
diff_bytes = 1b01000000000000000000000000000000000000000000000000000000000000
*/

                    miner_log("NICE", "Preparing submit of object to host " + (host), "green");
                    string objFile = "";
                    BufferGeometry geo(positions, indicies, normals, pos_len, indicies_len);
                    geo.ComputeVertexNormals();
                    objFile = geo.parse();

                    string rpl = "";
                    for (int i = 0; i < objFile.size(); i++)
                    {
                        if (objFile[i] == '\n')
                        {
                            rpl += "\\n";
                        }
                        else
                        {
                            rpl += objFile[i];
                        }
                    }

                    cout << "pre_hash = " << toHex(pre_hash, 32) << endl;
                    cout << "obj_hash = " << toHex(hash, 32) << endl;
                    cout << "best_hash = " << toHex(best_hash, 32) << endl;
                    cout << "diff_bytes = " << toHex(diffBytes, 32) << endl;
                    string submit_data = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"poscan_pushMiningObject\",\"params\":[\"" + rpl + "\",\"" + toHex(hash, 32) + "\"]}";
                    HttpHeader header = HttpRequest(host, "POST", "/", submit_data, "", "", "");
                    string res = header.data();
                    miner_log("INFO", "Node response to submit: " + res);
                }

    }
}

void cpuMain(uint64_t threadId)
{
    bool status = setThreadAffinity(cpuInfo.affinity[threadId]);
    if (status == false)
    {
        if (!silent)
            cout << "Failed to set affinity at thread " << threadId << endl;
        return;
    }

    Vec3Float64 *positions = new Vec3Float64[602];
    Vec3Float64 normals[602];
    unsigned int indicies[1806 * 3];
    int pos_len = 0;
    int indicies_len = 0;
    unsigned char hash[32];
    for (int i = 0; i < 32; i++)
    {
        hash[i] = 0;
    }
    int hash_len = 0;
    unsigned char sealPre[64];
    unsigned char seal[32];
    unsigned char hh[96];
    unsigned char hash_final[32];

    std::mt19937 m_gen;
    std::uniform_int_distribution<uint32_t> m_distribution{0, std::numeric_limits<uint32_t>::max()};
    std::uniform_int_distribution<uint64_t> m_distribution_ll{0, std::numeric_limits<uint64_t>::max()};

    std::chrono::system_clock::time_point now_time = std::chrono::system_clock::now();
    auto epoch = now_time.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count() % 0xfffffffffffffffeULL;

    m_gen = std::mt19937(us);

    uint256_t max256 = 1;
    for (int ii = 0; ii < 31; ii++)
    {
        max256 = max256 * 256;
    }
    uint256_t max256_t = max256 - 1;
    max256 = max256 * 255 + max256_t;
    SHA3 sha3_1;

    unsigned char oldDiff[32];
    for (int i = 0; i < 32; i++)
    {
        oldDiff[i] = 0;
    }
    vector<uint8_t> vec;

    Sphere sphere;
    CellRet adjacentVertices;
    CreateSphere(&sphere, sphere_radius, sphere_stacks, sphere_slices, SPHERE_NORMAL);
    adjacentVertices.size = sphere.len + 1;

    GetNeighbours(sphere.len, sphere.len_indicies, sphere.indices, &adjacentVertices);



    while (true)
    {
        int run = -1;
        if(metaOk){
            run = 0;
        }
        if (run == -1)
        {
            waitMs(100);
            continue;
        }

        RockObjParams objReturn;
        objReturn = random_hash(pre_hash, 4, positions, indicies, normals, hash, hash_len, pos_len, indicies_len, m_gen, m_distribution, NULL, &sphere, &adjacentVertices);

        if (count_dup)
        {
            std::string ss = "";
            if (hash_len > 0)
            {
                for (int i = 0; i < 32; i++)
                {
                    ss += hash[i];
                }
                auto it = map_hashes.find(ss);
                if (it != map_hashes.end())
                {
                    dups++;
                }
                else
                {
                    map_hashes.insert(make_pair(ss, 0));
                }
            }
        }

        hashesPerThreadBase[threadId]++;

        bool c = false;
        for (int i = 0; i < 32; i++)
        {
            if (oldDiff[i] != diffBytes[i])
            {
                c = true;
                oldDiff[i] = diffBytes[i];
            }
        }
        if (c)
        {
            uint256_t diff = 0;
            unsigned long long pow256 = 1;
            for (int i = 0; i < 8; i++)
            {
                diff = diff + diffBytes[i] * pow256;
                pow256 = pow256 * 256;
            }

            if (diff == 0)
                continue;
            uint256_t a = max256 / diff;

            vec = a.export_bits();
        }


                hashesPerThread[threadId]++;
                if (hash_len <= 0)
                {
                    nullsPerThread[threadId]++;
                    continue;
                }
                memcpy(sealPre, pre_hash, 32);
                memcpy(sealPre + 32, hash, 32);
               
                    sha3_1(sealPre, 64, seal);


                memcpy(hh, diffBytes, 32);
                memcpy(hh + 32, pre_hash, 32);
                memcpy(hh + 64, seal, 32);


                    sha3_1(hh, 96, hash_final);
  
                int cmp = 1;

                for (int ii = 0; ii < vec.size(); ii++)
                {
                    if (vec[ii] > hash_final[ii])
                    {
                        cmp = 1;
                        break;
                    }
                    else if (vec[ii] < hash_final[ii])
                    {
                        cmp = -1;
                        break;
                    }
                }
                bool valid = (cmp > 0);
                if (valid)
                {
                    miner_log("NICE", "Preparing submit of object to host " + (host), "green");
                    string objFile = "";
                    BufferGeometry geo(positions, indicies, normals, pos_len, indicies_len);
                    geo.ComputeVertexNormals();
                    objFile = geo.parse();
                    string rpl = "";
                    for (int i = 0; i < objFile.size(); i++)
                    {
                        if (objFile[i] == '\n')
                        {
                            rpl += "\\n";
                        }
                        else
                        {
                            rpl += objFile[i];
                        }
                    }
                    cout << "pre_hash = " << toHex(pre_hash, 32) << endl;
                    cout << "obj_hash = " << toHex(hash, 32) << endl;
                    cout << "best_hash = " << toHex(best_hash, 32) << endl;
                    cout << "diff_bytes = " << toHex(diffBytes, 32) << endl;
                    string submit_data = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"poscan_pushMiningObject\",\"params\":[\"" + rpl + "\",\"" + toHex(hash, 32) + "\",\"" + toHex(pre_hash, 32) + "\"]}";

                    HttpHeader header = HttpRequest(host, "POST", "/", submit_data, "", "", "");
                    string res = header.data();
                    miner_log("INFO", "Node response to submit: " + res);

                }

    }
}


void hashrate()
{
    int delay = 5000;
    int delay_custom = tryParseInt(args.get("hashrate-delay"));
    if (delay_custom > 500)
    {
        delay = delay_custom;
    }
    while (true)
    {
        auto t1 = getTime();
        waitMs(delay);
        auto t2 = getTime();
        int d = delay;
        delay = getMs(t1, t2);
        string text = "";
        unsigned long long h_total = 0;
        unsigned long long h_total_bad = 0;

            for (int i = 0; i < cpuInfo.cores; i++)
            {
                text += "core #" + to_string(i) + " (affinity " + to_string(cpuInfo.affinity[i]) + "): " + to_string(hashesPerThread[i] / (delay / 1000)) + " h/s\n";
                h_total += hashesPerThread[i];
                hashesPerThread[i] = 0;
                h_total_bad += nullsPerThread[i];
                nullsPerThread[i] = 0;
            }


        double good = (h_total - h_total_bad - dups) / (delay / 1000.0);
        if (count_dup)
        {
            map_hashes.clear();
            dups = 0;
        }
        double hh = h_total / (delay / 1000.0);
        unsigned long long hh_int = h_total / (delay / 1000);
        std::string hs = to_string(hh);
        std::string hs_int = to_string(hh_int);
        string ht = "H/s";
        hashrateTotalLast = hh_int;
        if (hh >= 1000000)
        {
            hh = hh / 1000000;
            ht = "MH/s";
        }
        else if (hh >= 1000)
        {
            hh = hh / 1000;
            ht = "KH/s";
        }
        hs = to_string(hh);
        unsigned long long diff = 0;
        unsigned long long pow256 = 1;
        for (int i = 0; i < 8; i++)
        {
            diff = diff + diffBytes[i] * pow256;
            pow256 = pow256 * 256;
        }

        if (use_gpu_sha3 == false)
        {
            for (int i = 0; i < cpuInfo.cores; i++)
            {
                hashesPerThreadBase[i] = 0;
            }

            miner_log("INFO", "Hashrate total: " + hs + " " + ht + "   GOOD: " + to_string(good) + " H/s   Diff: " + to_string(diff));

        }
        else
        {
            h_total = 0;
            for (int i = 0; i < cpuInfo.cores; i++)
            {
                h_total += hashesPerThreadBase[i];
                hashesPerThreadBase[i] = 0;
            }
            hh = h_total / (delay / 1000.0);
            miner_log("INFO", "Hashrate total: " + hs + " " + ht + "   CPU_BASE: " + to_string((int)hh) + " H/s   Diff: " + to_string(diff));
        }

    }
}

void initGpu()
{
    gpusCount = 0;
    cudaGetDeviceCount(&gpusCount);
    cout << "gpu_count = " << gpusCount << endl;
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime" << endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    for (int i = 0; i < gpusCount; i++)
    {
        cudaDeviceProp pr;
        cudaGetDeviceProperties(&pr, i);
        log("GPU #" + to_string(i), std::string(pr.name), "green");
        gpuInfos[i].blocks = pr.multiProcessorCount;
        gpuInfos[i].threads = 256;
    }
}


int main(int argc, char *argv[])
{

    srand(time(NULL));

    args = CmdArgs(argv, argc, false, false);
    cpuInfo = getCpuInfo();

    if (!silent)
    {
        log("INFO", "CPU: " + cpuInfo.cpuName);
        log("INFO", "Threads: " + to_string(cpuInfo.cores));
    }

    if (args.get("threads") != "NULL_ARG")
    {
        int threadCount = tryParseInt(args.get("threads"));
        if (threadCount == 0)
        {
            exitMsg("Invalid thread count: 0");
        }
        cpuInfo.cores = threadCount;
    }

    int xxx = 0;
    while (cpuInfo.affinity.size() < cpuInfo.cores)
    {
        cpuInfo.affinity.push_back(-1);
        xxx++;
    }

    if (args.get("affinity") != "NULL_ARG")
    {
        vector<string> list = split(args.get("affinity"), " ");
        if (list.size() > cpuInfo.cores)
        {
            exitMsg("Invalid affinity");
        }
        for (int i = 0; i < list.size(); i++)
        {
            cpuInfo.affinity[i] = tryParseInt(list[i]);
        }
    }

    // cudaGetDeviceCount(&gpus);


    if (args.m_args.size() == 0)
    {
        help();
        return 0;
    }



    if (args.get("sp-stacks") != "NULL_ARG")
    {
        int s = tryParseInt(args.get("sp-stacks"));
        if (s < 0 || s > 30)
        {
            log("ERROR", "Invalid range, accepted: [4..30]");
        }
        else
        {
            sphere_stacks = s;
        }
    }

    if (args.get("sp-radius") != "NULL_ARG")
    {
        sphere_radius = tryParseFloat(args.get("sp-radius"));
    }


    if (args.get("sp-slices") != "NULL_ARG")
    {
        int s = tryParseInt(args.get("sp-slices"));
        if (s < 0 || s > 30)
        {
            log("ERROR", "Invalid range, accepted: [4..30]");
        }
        else
        {
            sphere_slices = s;
        }
    }

    if (args.get("obj-mode") != "NULL_ARG")
    {
        if (args.get("obj-mode") == "sphere")
        {
            sphereMode = SPHERE_NORMAL;
        }
        else if (args.get("obj-mode") == "random")
        {
            sphereMode = SPHERE_RANDOM;
            log("WARN", "Obj mode changed to random");
        }
    }
    else
    {
        sphereMode = SPHERE_NORMAL;
    }

    if (args.get("use-noise") != "NULL_ARG")
    {
        use_perlin = true;
    }

    {
        Sphere sp;
        CreateSphere(&sp, 1, sphere_stacks, sphere_slices, sphereMode);
        log("INFO", "Sphere sizes: verts=" + to_string(sp.len) + " indicies=" + to_string(sp.len_indicies * 3));
    }

    if (args.get("tests") != "NULL_ARG")
    {
        int s = tryParseInt(args.get("tests"));
        tests = s;
    }


    log("INFO", "Sphere stacks: " + to_string(sphere_stacks));
    log("INFO", "Sphere slices: " + to_string(sphere_slices));
    log("INFO", "Sphere radius: " + to_string(sphere_radius));

    if (args.get("use-sha") != "NULL_ARG")
    {
        sha256Mode = SHA256_MODE_SHA;
        log("INFO", "Using sha256 special simd");
    }
    else
    {
        sha256Mode = SHA256_MODE_NORMAL;
    }

    log("INFO", "Using avx2: " + to_string(use_avx2));


    if (args.get("gpu") != "NULL_ARG")
    {
        use_gpu = true;
        log("INFO", "Using GPU");
        initGpu();
    }


    if (args.get("benchmark") != "NULL_ARG" || args.get("bench") != "NULL_ARG")
    {

        benchmark();
        return 0;
    }


    if (args.get("count-dup") != "NULL_ARG")
    {
        count_dup = true;
    }

    if (args.get("test") != "NULL_ARG")
    {
        auto start = high_resolution_clock::now();
        repeats = 0;
        TEST_RUN = 1;
        totalRocks = 0;


        Vec3Float64 positions[SPHERE_MAX_SIZE];
        Vec3Float64 normals[SPHERE_MAX_SIZE];
        unsigned int indicies[SPHERE_MAX_IND_SIZE];
        unsigned char hash[32];
        int hash_len = 0;
        int a, b;
        unsigned char trans[4];
        unsigned int trans_len = 0;
        if (args.get("x") != "NULL_ARG")
        {
            trans[0] = tryParseInt(args.get("x"));
            trans_len++;
        }
        if (args.get("y") != "NULL_ARG")
        {
            trans[1] = tryParseInt(args.get("y"));
            trans_len++;
        }
        if (args.get("z") != "NULL_ARG")
        {
            trans[2] = tryParseInt(args.get("z"));
            trans_len++;
        }
        if (args.get("a") != "NULL_ARG")
        {
            trans[3] = tryParseInt(args.get("a"));
            trans_len++;
        }
        std::mt19937 m_gen;
        std::uniform_int_distribution<uint32_t> m_distribution{0, std::numeric_limits<uint32_t>::max()};
        std::uniform_int_distribution<uint64_t> m_distribution_ll{0, std::numeric_limits<uint64_t>::max()};

        std::chrono::system_clock::time_point now_time = std::chrono::system_clock::now();
        auto epoch = now_time.time_since_epoch();
        auto us = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count() % 0xfffffffffffffffeULL;

        m_gen = std::mt19937(us);
        for (int tr = 0; tr < 1; tr++)
        {
            cout << "Try: " << tr << endl;
            cout << "Rot: " << trans_len << endl;
            random_hash(trans, trans_len, positions, indicies, normals, hash, hash_len, a, b, m_gen, m_distribution);
            cout << "Hash len: " << hash_len << "\n";
            if (hash_len == 32)
            {
                for (int j = 0; j < 32; j++)
                {
                    unsigned char c = hash[j];
                    char c1, c2;
                    c1 = c / 16;
                    if (c1 < 10)
                    {
                        c1 = '0' + c1;
                    }
                    else
                    {
                        c1 = 'a' + (c1 - 10);
                    }
                    c2 = c % 16;
                    if (c2 < 10)
                    {
                        c2 = '0' + c2;
                    }
                    else
                    {
                        c2 = 'a' + (c2 - 10);
                    }
                    cout << c1 << c2;
                }
                cout << endl;
            }
            else
            {
            }
            hash_len = 0;
        }

        //
        // geo.roundDecimals();
        cout << "repeats: " << repeats << endl;
        cout << nullRocks << "/" << totalRocks << endl;

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << duration.count() << endl;

        cout << "done" << endl;

        // test();
        return 0;
    }


    if (args.get("host") != "NULL_ARG")
    {
        host = args.get("host");
    }


    if (host.size() == 0)
    {
        printf("Host is empty\n");
        help();
        return 0;
    }

    initTcp();


    if (use_gpu == false)
    {
        for (int i = 0; i < cpuInfo.cores; i++)
        {
            threads.push_back(thread(cpuMain, i));
            waitMs(rand() % 50);
        }
    }
    else
    {
        for (int i = 0; i < gpusCount; i++)
        {
            threads.push_back(thread(gpuMain, i));
            waitMs(rand() % 50);
        }
    }

    int meta_step = 5;

    if (args.get("update-interval") != "NULL_ARG")
    {
        update_interval = tryParseInt(args.get("update-interval"));
    }
    if (args.get("updates-per-thread") != "NULL_ARG")
    {
        meta_step = tryParseInt(args.get("updates-per-thread"));
        if (meta_step < 1)
        {
            printf("Updates per thread should be over 0\n");
            return 0;
        }
    }
    else
    {
        meta_step = 5;
    }

    threads.push_back(thread(metaLoop));

    threads.push_back(thread(hashrate));


    logInit = true;
    while (true)
    {
        waitMs(25);
        for (int i = 0; i < log_vector.size(); i++)
        {
            log(log_vector[i].tag, log_vector[i].text, log_vector[i].color);
        }
        log_vector.clear();
    }
    return 0;
}
