#include "grid.h"
#include <math.h>
#include "buffer_geometry.h"
#include "simple_deque.h"
#include "simple_deque_type.cu"
#include "sha256.h"
#include <map>
#include <unordered_map>
#include <algorithm>
#include "utils.h"
#include <queue>
#include <stack>
#include "sha256_simd.h"
#include <bitset>

#include <immintrin.h>
#include <x86intrin.h>

// template <size_t rows, size_t cols>
void chk_add(int (&z)[8][10], int n, int i, int j)
{
    if ((i >= 0 && i < n) && (j >= 0 && j < n))
    {
        bool f = false;
        for (int o = 0; o < z[i][9]; o++)
        {
            if (z[i][o] == j)
            {
                f = true;
                break;
            }
        }
        if (f == false)
        {
            z[i][z[i][9]] = j;
            z[i][9]++;
        }
    }
}

void chk_add(int (&z)[8][8], int n, int i, int j)
{
    if ((i >= 0 && i < n) && (j >= 0 && j < n))
    {
        z[i][j] = 1;
    }
}

inline bool chk_zone(int i, int j, int (&z)[8][10], PolyLine *line)
{
    if (i < 0 || i >= 8 || j < 0 || j >= 8)
    {
        return false;
    }

    Vec2Int first = line->nodes[0];
    if (first.x == i && first.y == j && line->len > 5)
    {
        return true;
    }

    if (line->allowed[i][j] != 0)
        return false;

    bool f = false;
    for (int x = 0; x < z[i][9]; x++)
    {
        if (z[i][x] == j)
        {
            f = true;
            break;
        }
    }
    if (f == false)
    {
        return false;
    }
    /*
    if(line->allowed[i][j] != 0){
        printf("not ok\n");
        return false;
    } else {
        printf("let %d %d\n",i,j);
        for(int x=0;x<8;x++){
            for(int y=0;y<8;y++){
                printf("%d ",line->allowed[x][y]);
            }
            printf("\n");
        }
        printf("\n");
        for(int x=0;x<line->len;x++){
            printf("(%d,%d) ",line->nodes[x].x,line->nodes[x].y);
        }
        printf("\n");
    }
    sleep(1);
    */
    /*
    for(int p_index = 0;p_index < line->len;p_index++){
        int pi = line->nodes[p_index].x;
        int pj = line->nodes[p_index].y;
        if(abs(pi-i) < 2 && abs(pj-j) < 2){
            return false;
        }
    }
    */
    return true;
}

inline bool chk_zone(int i, int j, int (&z)[8][8], PolyLine *line)
{
    if (i < 0 || i >= 8 || j < 0 || j >= 8)
    {
        return false;
    }

    Vec2Int first = line->nodes[0];
    if (first.x == i && first.y == j && line->len > 5)
    {
        return true;
    }

    if (z[i][j] != 1)
        return false;

    /*
    for(int x=0;x<line->len;x++){
        Vec2Int p = line->nodes[x];
        if(abs(p.x-i) < 2 && abs(p.y-j) < 2){
            return false;
        }
    }
    */

    if (line->allowed[i][j] != 0)
        return false;

    return true;
}

void near_points(int (&zone)[8][10], PolyLine *line, Vec2Int start_point, Vec2Int *v, int &v_len)
{

#define tp int

    tp dist = 2; // DISTANCE const = 2
    tp min_i = start_point.x - dist;
    tp min_j = start_point.y - dist + 1;
    tp max_i = start_point.x + dist;
    tp max_j = start_point.y + dist - 1;

    for (tp ii = min_i; ii <= max_i; ii++)
    {
        tp jj = min_j - 1;
        if (chk_zone(ii, jj, zone, line))
        {
            v[v_len].x = ii;
            v[v_len].y = jj;
            v_len++;
        }
    }
    // 5

    for (int jj = min_j; jj <= max_j; jj++)
    {
        tp ii = max_i;
        if (chk_zone(ii, jj, zone, line))
        {
            v[v_len].x = ii;
            v[v_len].y = jj;
            v_len++;
        }
    }

    // 3

    for (tp ii = min_i; ii <= max_i; ii++)
    {
        tp jj = max_j + 1;
        if (chk_zone(ii, jj, zone, line))
        {
            v[v_len].x = ii;
            v[v_len].y = jj;
            v_len++;
        }
    }

    // 5

    for (tp jj = min_j; jj <= max_j; jj++)
    {
        tp ii = min_i;
        if (chk_zone(ii, jj, zone, line))
        {
            v[v_len].x = ii;
            v[v_len].y = jj;
            v_len++;
        }
    }

    // 3
}

void near_points(int (&zone)[8][8], PolyLine *line, Vec2Int start_point, Vec2Int *v, int &v_len)
{

    int dist = 2; // DISTANCE const = 2
    int min_i = start_point.x - dist;
    int min_j = start_point.y - dist + 1;
    int max_i = start_point.x + dist;
    int max_j = start_point.y + dist - 1;

    for (int ii = min_i; ii <= max_i; ii++)
    {
        int jj = min_j - 1;
        if (chk_zone(ii, jj, zone, line))
        {
            v[v_len].x = ii;
            v[v_len].y = jj;
            v_len++;
        }
    }
    // 5

    for (int jj = min_j; jj <= max_j; jj++)
    {
        int ii = max_i;
        if (chk_zone(ii, jj, zone, line))
        {
            v[v_len].x = ii;
            v[v_len].y = jj;
            v_len++;
        }
    }

    // 3

    for (int ii = min_i; ii <= max_i; ii++)
    {
        int jj = max_j + 1;
        if (chk_zone(ii, jj, zone, line))
        {
            v[v_len].x = ii;
            v[v_len].y = jj;
            v_len++;
        }
    }

    // 5

    for (int jj = min_j; jj <= max_j; jj++)
    {
        int ii = min_i;
        if (chk_zone(ii, jj, zone, line))
        {
            v[v_len].x = ii;
            v[v_len].y = jj;
            v_len++;
        }
    }

    // 3
}

struct TP64
{
    double val;
    Vec2F64 p;
};

double calc_sco(Vec2F64 *cn, int cn_len, PolyLine *line, Vec3Float64 v_min, Vec3Float64 v_max)
{
    double l = 0.0;
    /*
    Vec2F64 * line2 = new Vec2F64[line->len];
    for(int i=0;i<line->len;i++){
        line2[i].x = line->nodes[i].x + 0.5;
        line2[i].y = line->nodes[i].y + 0.5;
    }
    */

    Vec2F64 p1; // line2[0];
    p1.x = line->nodes[0].x + 0.5;
    p1.y = line->nodes[0].y + 0.5;

    TP64 ll[40]; // = new TP64[100];//line->len];
    ll[0].val = 0;
    ll[0].p.x = p1.x;
    ll[0].p.y = p1.y;

    for (int i = 1; i < line->len; i++)
    {
        Vec2F64 p2; // line2[i];
        p2.x = line->nodes[i].x + 0.5;
        p2.y = line->nodes[i].y + 0.5;
        l = l + ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
        ll[i].p.x = p2.x;
        ll[i].p.y = p2.y;
        ll[i].val = l;
        p1.x = p2.x;
        p1.y = p2.y;
    }
    // delete[] line2;

    double tot_len = ll[line->len - 1].val;
    double dl = tot_len / (double)cn_len;
    unsigned int m = 0;
    Vec2F64 p = ll[0].p;

    double sum = 0;
    sum += (cn[0].x - ll[0].p.x) * (cn[0].x - ll[0].p.x) + (cn[0].y - ll[0].p.y) * (cn[0].y - ll[0].p.y);
    // Vec2F64 * res = new Vec2F64[cn_len];
    // res[0].x = ll[0].p.x;
    // res[0].y = ll[0].p.y;
    // int res_len = 1;

    for (int k = 1; k < cn_len; k++)
    {
        double r = k * dl;
        while (m < line->len)
        {
            double l = ll[m].val;
            if (r < l)
            {
                // cur_path = r;
                break;
            }
            p = ll[m].p;
            m += 1;
        }
        //     # px = (p2[0] - p1[0]) / l * dl # TODO!!!
        //     # py = (p2[1] - p1[1]) / l * dl # TODO!!!

        double dd = r - ll[m - 1].val;
        // let (mut dx, mut dy): (f64, f64) = (0.0, 0.0);

        double dx, dy;
        if (abs(ll[m].p.x - ll[m - 1].p.x) > 1.0e-10)
        {
            double kk = (ll[m].p.y - ll[m - 1].p.y) / (ll[m].p.x - ll[m - 1].p.x);
            dx = dd / sqrt(1.0 + kk * kk); // sqrt(1.0 + kk * kk);
            dy = kk * dx;
        }
        else
        {
            dx = 0.0;
            dy = dd;
        }
        dx = p.x + dx;
        dy = p.y + dy;
        sum += (cn[k].x - dx) * (cn[k].x - dx) + (cn[k].y - dy) * (cn[k].y - dy);
    }
    // delete[] ll;
    return sum / (double)cn_len;
}

void ff(PolyLine *pl, Deque_F64_PolyLine *top_in_cntr, Vec2F64 *cntr, int cntrl_len, Vec3Float64 v_min, Vec3Float64 v_max)
{
    double d = calc_sco(cntr, cntrl_len, pl, v_min, v_max);
    int len = top_in_cntr->len;
    if (len > 0)
    {
        if (d < top_in_cntr->get(len - 1)->val || len <= 10)
        {
            if (len == 10)
            {
                top_in_cntr->deletefront();
            }
            top_in_cntr->insertrear(d, pl);
            top_in_cntr->sort_by_val();
        }
    }
    else
    {

        top_in_cntr->insertrear(d, pl);
    }
};

void ff(PolyLine *pl, std::deque<T64_PolyLine> &top_in_cntr, Vec2F64 *cntr, int cntrl_len, Vec3Float64 v_min, Vec3Float64 v_max)
{
    double d = calc_sco(cntr, cntrl_len, pl, v_min, v_max);
    int len = top_in_cntr.size();
    if (len > 0)
    {
        if (d < top_in_cntr.back().val || len <= 10)
        {
            if (len == 10)
            {
                top_in_cntr.pop_front();
            }
            T64_PolyLine temp;
            temp.val = d;
            temp.line.len = pl->len;
            memcpy(temp.line.nodes, pl->nodes, pl->len * sizeof(Vec2Int));
            auto it = top_in_cntr.end() - 1;
            auto sol = it;
            int steps = 0;
            while (d > (*it).val)
            {
                steps++;
                sol = it;
                it--;
                if (sol == top_in_cntr.begin())
                {
                    break;
                }
            }
            if (steps == 0)
                top_in_cntr.push_back(temp);
            else
                top_in_cntr.insert(sol, temp);
        }
    }
    else
    {

        T64_PolyLine temp;
        temp.val = d;
        temp.line.len = pl->len;
        memcpy(temp.line.nodes, pl->nodes, pl->len * sizeof(Vec2Int));
        top_in_cntr.push_back(temp);

      
    }
};

void ff_3(PolyLine *pl, std::deque<T64_PolyLine> &top_in_cntr, Vec2F64 *cntr, int cntrl_len, Vec3Float64 v_min, Vec3Float64 v_max)
{

    double d = calc_sco(cntr, cntrl_len, pl, v_min, v_max);
    int len = top_in_cntr.size();

    if (len == 10)
    {
        double maxval = 0;
        int index = -1;
        for (int i = 0; i < 10; i++)
        {
            if (maxval < top_in_cntr[i].val)
            {
                maxval = top_in_cntr[i].val;
                index = i;
            }
        }
        // T64_PolyLine temp;
        top_in_cntr[index].val = d;
        top_in_cntr[index].line.len = pl->len;
        memcpy(top_in_cntr[index].line.nodes, pl->nodes, pl->len * sizeof(Vec2Int));

    }
    else
    {

        T64_PolyLine temp;
        temp.val = d;
        temp.line.len = pl->len;
        memcpy(temp.line.nodes, pl->nodes, pl->len * sizeof(Vec2Int));
        top_in_cntr.push_back(temp);

    }
};

#define NEIB_MODE_NORMAL

const int d2x[] = {0, 1, 2, 2, 2, 2, 2, 1, 0, -1, -2, -2, -2, -2, -2, -1};
const int d2y[] = {-2, -2, -2, -1, 0, 1, 2, 2, 2, 2, 2, 1, 0, -1, -2, -2};

int paths_len;

void complete_line_3(int &lev, int (&zone)[8][8], PolyLine *line, deque<T64_PolyLine> &top_in_cntr, Vec2F64 *cntr, int cntrl_len, Vec3Float64 v_min, Vec3Float64 v_max)
{
    /*
    lev ++;
    if(lev > 6)
    {
        lev = -10000;
    }
    */
    lev++;

    Vec2Int start_point = line->nodes[line->len - 1];
    Vec2Int first_point = line->nodes[0];
    Vec2Int v[(8 - 1) * 4];
    int v_len = 0;
    near_points(zone, line, start_point, v, v_len);

    if (start_point.x == 0 && start_point.y == 0)
    {
        int old_lev = lev;
        lev = v_len * 128 + lev;
        if (v_len < 2)
        {
            // printf("skip\n");
            return;
        }
        else if (v_len == 2)
        {
            Vec2Int p1, p2;
            p1 = v[0];
            p2 = v[1];
            if (abs(p1.x - p2.x) < 2 && abs(p1.y - p2.y) < 2)
            {
                return;
            }
        }
        else
        {
            int good[8];
            for (int i = 0; i < 8; i++)
                good[i] = 1;
            for (int i = 0; i < v_len; i++)
            {
                int g = 0;
                for (int j = 0; j < v_len; j++)
                {
                    if (i == j)
                        continue;
                    Vec2Int p1, p2;
                    p1 = v[i];
                    p2 = v[j];
                    if (abs(p1.x - p2.x) >= 2 || abs(p1.y - p2.y) >= 2)
                    {
                        g = 1;
                        break;
                    }
                }
                if (g == 0)
                {
                    good[i] = 0;
                    // printf("not good %d\n",g);
                }
            }
            int good_len = 0;
            int stp_ = 0;
            for (int i = 0; i < v_len; i++)
            {
                if (good[i] == 0)
                {
                    v[i].x = -1;
                }
                else
                {
                    good_len++;
                }
            }
            lev = good_len * 128 + old_lev;

            // printf("%d\n",v_len);
        }
    }

    if (lev / 128 == 2 && lev % 128 == 2)
    {
        for (int i = 0; i < v_len; i++)
        {
            if (v[i].x == -1)
                continue;
            char enc = v[i].x * 8 + v[i].y;
            if (
                (v[i].x == 0 && v[i].y == 2) ||
                (v[i].x == 1 && v[i].y == 2) ||
                (v[i].x == 2 && v[i].y == 2) ||
                (v[i].x == 2 && v[i].y == 0) ||
                (v[i].x == 2 && v[i].y == 1))
            {
                v[i].x = -1;
            }
        }
    }

    /*
    1 * x
    * * x
    x x x
    */

    if (v_len > (8 - 1) * 4)
    {
        printf("ANOTHER OVERFLOW\n");
        return;
    }
    for (int i = 0; i < v_len; i++)
    {
        Vec2Int p = v[i];
        if (p.x == -1)
            continue;

        if (p.x == first_point.x && p.y == first_point.y)
        {

            // Normal push
            line->push(p);

            ff_3(line, top_in_cntr, cntr, cntrl_len, v_min, v_max);

            line->pop();
            continue;
        }

        line->push(p);

        complete_line_3(lev, zone, line, top_in_cntr, cntr, cntrl_len, v_min, v_max);

        line->pop();
    }
    lev--;
}

map<string, int> map_zones;
map<string, vector<T64_PolyLine_Compress>> map_zones_not_empty;
int reps = 0;
int zones_loaded = 0;
int precalc_uses = 0;

int bigFrv[8][8];

void find_top_std_3(Vec2F64 *cntrs, int *cntrs_len, unsigned int depth,
                    unsigned int n_sect, unsigned int grid_size, Vec3Float64 v_min, Vec3Float64 v_max, unsigned char *out_hashes, int &out_hashes_len)
{

#define USE_STD_DEQUE

#ifdef USE_STD_DEQUE
    deque<T64_PolyLine> top_in_cntr_deque;
#endif

    double width = v_max.x - v_min.x;
    double height = v_max.y - v_min.y;
    const int N = 2;
    out_hashes_len = 0;
    if (cntrs_len == 0)
    {
        return;
    }
    Tuple_select_top_all ss[12][10];
    int ss_len = 0;
    int ss_lens[12];
    for (int i = 0; i < 12; i++)
    {
        ss_lens[i] = 0;
    }

    /*
        calc ss
    */
    ss_len = 0;

    int zone[8][10];
    int zoneF[8][8];

    double grid_dx = width / (double)grid_size;
    double grid_dy = height / (double)grid_size;
    double dx = 0.1 * grid_dx;
    double dy = 0.1 * grid_dy;
    short _len = 12;
    int psum = 0;

    for (int j = 0; j < 12; j++)
    {

        Vec2F64 *cntr = cntrs + psum;
        int cntr_len = cntrs_len[j + 1];

        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                zoneF[i][j] = 0;

        /*
            fn line_zone
        */
        for (int x = 0; x < cntr_len; x++)
        {
            double px, py;
            px = cntr[x].x;
            py = cntr[x].y;
            int ii = (int)((px - v_min.x) / grid_dx);
            int jj = (int)((py - v_min.y) / grid_dy);
            if (ii > grid_size || jj > grid_size)
                continue;
            if (ii == grid_size)
            {
                ii = grid_size - 1;
            }
            if (jj == grid_size)
            {
                jj = grid_size - 1;
            }

            zoneF[ii][jj] = 1;

            double xx = round(px) - px;
            double yy = round(py) - py;

            if (abs(xx) < dx)
            {
                if (xx >= 0.0)
                {
                    chk_add(zoneF, grid_size, ii + 1, jj);
                }
                else
                {
                    chk_add(zoneF, grid_size, ii - 1, jj);
                }
            }
            if (abs(yy) < dy)
            {
                if (yy >= 0.0)
                {
                    chk_add(zoneF, grid_size, ii, jj + 1);
                }
                else
                {
                    chk_add(zoneF, grid_size, ii, jj - 1);
                }
            }
            if (abs(yy) < dy && abs(xx) < dx)
            {
                if (xx >= 0.0 && yy >= 0.0)
                {
                    chk_add(zoneF, grid_size, ii + 1, jj + 1);
                }
                if (xx >= 0.0 && yy < 0.0)
                {
                    chk_add(zoneF, grid_size, ii + 1, jj - 1);
                }
                if (xx < 0.0 && yy >= 0.0)
                {
                    chk_add(zoneF, grid_size, ii - 1, jj + 1);
                }
                if (xx < 0.0 && yy < 0.0)
                {
                    chk_add(zoneF, grid_size, ii - 1, jj - 1);
                }
            }
        }

        /*
            fn complete_line
        */
        PolyLine line;
        line.len = 0;
        Vec2Int point0;
        point0.x = 0;
        point0.y = 0;
        line.push(point0);

        /*
            fn NeiborNodes:new
        */
        int lev = 0;

#ifdef USE_STD_DEQUE

        top_in_cntr_deque.clear();
#else
        Deque_F64_PolyLine top_in_cntr(200);
#endif

        // cout << "Path count: " << paths_count << endl;
        // return;

        lev = 0;
        complete_line_3(lev, zoneF, &line, top_in_cntr_deque, cntr, cntr_len, v_min, v_max);

#ifdef USE_STD_DEQUE
        int dsize = top_in_cntr_deque.size();
        dsize = min(dsize, 2);
        // printf("%d\n",dsize);
        for (int ii = 0; ii < dsize; ii++)
        {
#else
        for (int ii = 0; ii < top_in_cntr.len; ii++)
        {
#endif

#ifdef USE_STD_DEQUE
            T64_PolyLine dq = top_in_cntr_deque.front();
            T64_PolyLine *a = &dq;
#else
            T64_PolyLine *a = top_in_cntr.get(ii);
#endif
            // printf("val=%lf\n",a->val);
            ss[j][ss_lens[j]].val = a->val;

            for (int jj = 0; jj < a->line.len; jj++)
            {
                int xx = a->line.nodes[jj].x;
                int yy = a->line.nodes[jj].y;
                a->line.nodes[jj].x = __builtin_bswap32(xx);
                a->line.nodes[jj].y = __builtin_bswap32(yy);
                // int_to_be(xx,(unsigned char*)a->line.nodes[jj].x);
                // int_to_be(yy,(unsigned char*)a->line.nodes[jj].y);
            }
            if (sha256Mode == SHA256_MODE_NORMAL)
            {
                CUDA_SHA256_CTX sha_ctx;
                cuda_sha256_init(&sha_ctx);
                cuda_sha256_update(&sha_ctx, (unsigned char *)a->line.nodes, 8 * a->line.len);
                cuda_sha256_final(&sha_ctx, ss[j][ss_lens[j]].hash);
            }
            else if (sha256Mode == SHA256_MODE_SHA)
            {
                sha256_sha_full((unsigned char *)a->line.nodes, 8 * a->line.len, ss[j][ss_lens[j]].hash);
            }
            ss_lens[j]++;
            top_in_cntr_deque.pop_front();
        }

        ss_len++;
        psum += cntrs_len[j + 1];
    }

    int okk = 0;
    for (int j = 0; j < 12; j++)
    {
        if (ss_lens[j] > 0)
        {
            okk = 1;
        }
    }

    F64_Hash384 best_totals2[10];
    best_totals2[0].hash_len = 0;
    int len_best_totals2 = 0;

    double expect = 0;
    for (int i = 0; i < 12; i++)
    {
        // printf("lev = %d\n",i);
        if (ss_lens[i] > 0)
        {
            if (ss[i][0].val < ss[i][1].val)
            {
                memcpy(best_totals2[0].hash + best_totals2[0].hash_len, ss[i][0].hash, 32);
            }
            else
            {
                memcpy(best_totals2[0].hash + best_totals2[0].hash_len, ss[i][1].hash, 32);
            }
            best_totals2[0].hash_len += 32;
            expect += min(ss[i][0].val, ss[i][1].val);
        }
    }

    for (int i = 0; i < 1; i++)
    {
        F64_Hash384 *hash = &best_totals2[i]; // best_totals.get(i);
        CUDA_SHA256_CTX sha_ctx;
        cuda_sha256_init(&sha_ctx);
        cuda_sha256_update(&sha_ctx, (unsigned char *)hash->hash, hash->hash_len);
        cuda_sha256_final(&sha_ctx, out_hashes + (out_hashes_len * 32));
        out_hashes_len++;
    }
}
