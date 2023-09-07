#include "cuda_runtime.h"
#include "grid_cuda.cuh"
#include <math.h>
#include "buffer_geometry.h"
#include "simple_deque.h"
#include "simple_deque_type.cu"
#include "sha256.h"
#include "utils.h"


//template <size_t rows, size_t cols>
__device__ void chk_add_cuda(int (&z)[8][10], int n, int i, int j) {
	if( (i>=0 && i < n) && (j>=0 && j < n)) {
        bool f = false;
        for(int o=0;o<z[i][9];o++){
            if(z[i][o] == j){
                f = true;
                break;
            }
        }
        if(f==false){
            z[i][z[i][9]] = j;
            z[i][9]++;
        }
	}
}

__device__ void chk_add_cuda(int (&z)[8][8], int n, int i, int j) {
	if( (i>=0 && i < n) && (j>=0 && j < n)) {
        z[i][j]=1;
	}
}

 __device__ inline bool chk_zone_cuda(int i,int j,int (&z)[8][10],PolyLine * line){
    if(i < 0 || i >= 8 || j < 0 || j >= 8) {
        return false;
    }

    Vec2Int first = line->nodes[0];
    if(first.x == i && first.y == j && line->len > 5){
        return true;
    }

    if(line->allowed[i][j] != 0)
    return false;

    bool f = false;
    for(int x=0;x<z[i][9];x++){
        if(z[i][x] == j)
        {
            f=true;
            break;
        }
    }
    if(f==false){
        return false;
    }

    return true;
}

 __device__ inline bool chk_zone_cuda(int i,int j,int (&z)[8][8],PolyLine * line){
    if(i < 0 || i >= 8 || j < 0 || j >= 8) {
        return false;
    }

    if(line->nodes[0].x == i && line->nodes[0].y == j && line->len > 5){
        return true;
    }

    if(line->allowed[i][j] != 0)
    return false;

    if(z[i][j] == 0)
        return false;

    return true;
}

 __device__ void near_points_cuda(int (&zone)[8][10],PolyLine * line,Vec2Int start_point,Vec2Int * v,int &v_len){

        int dist = 2; // DISTANCE const = 2
        int min_i = start_point.x - dist;
        int min_j = start_point.y - dist + 1;
        int max_i = start_point.x + dist;
        int max_j = start_point.y + dist - 1;

        for(int ii=min_i; ii<=max_i; ii++) {
            int jj = min_j - 1;
            if(chk_zone_cuda(ii, jj, zone, line)) {
                v[v_len].x = ii;
                v[v_len].y = jj;
                v_len ++;
            }
        }
        // 5 aici

        for(int jj=min_j;jj<=max_j;jj++){
            int ii = max_i;
            if(chk_zone_cuda(ii, jj, zone, line)) {
                v[v_len].x = ii;
                v[v_len].y = jj;
                v_len ++;
            }
        }

        // 3 aici

        for(int ii=min_i; ii<=max_i; ii++) {
            int jj = max_j + 1;
            if(chk_zone_cuda(ii, jj, zone, line)){
                v[v_len].x = ii;
                v[v_len].y = jj;
                v_len ++;
            }
        }

        // 5

        for(int jj=min_j;jj<=max_j;jj++){
            int ii = min_i;
            if(chk_zone_cuda(ii, jj, zone, line)) {
                v[v_len].x = ii;
                v[v_len].y = jj;
                v_len ++;
            }
        }

        // 3
}

 __device__ void near_points_cuda(int (&zone)[8][8],PolyLine * line,Vec2Int start_point,Vec2Int * v,int &v_len){    
        int dist = 2; // DISTANCE const = 2
        int min_i = start_point.x - dist;
        int min_j = start_point.y - dist + 1;
        int max_i = start_point.x + dist;
        int max_j = start_point.y + dist - 1;

        for(int ii=min_i; ii<=max_i; ii++) {
            int jj = min_j - 1;
            if(chk_zone_cuda(ii, jj, zone, line)) {
                v[v_len].x = ii;
                v[v_len].y = jj;
                v_len ++;
            }
        }
        // 5 aici

        for(int jj=min_j;jj<=max_j;jj++){
            int ii = max_i;
            if(chk_zone_cuda(ii, jj, zone, line)) {
                v[v_len].x = ii;
                v[v_len].y = jj;
                v_len ++;
            }
        }

        // 3 aici

        for(int ii=min_i; ii<=max_i; ii++) {
            int jj = max_j + 1;
            if(chk_zone_cuda(ii, jj, zone, line)){
                v[v_len].x = ii;
                v[v_len].y = jj;
                v_len ++;
            }
        }

        // 5

        for(int jj=min_j;jj<=max_j;jj++){
            int ii = min_i;
            if(chk_zone_cuda(ii, jj, zone, line)) {
                v[v_len].x = ii;
                v[v_len].y = jj;
                v_len ++;
            }
        }

        // 3
        

}

struct TP64{
    double val;
    Vec2F64 p;
};

__device__ double calc_sco_cuda(Vec2F64 * cn,int cn_len, PolyLine * line,Vec3Float64 v_min,Vec3Float64 v_max){
        double l = 0.0;

        /*
        Vec2F64 * line2 = new Vec2F64[line->len];
        for(int i=0;i<line->len;i++){
            line2[i].x = line->nodes[i].x + 0.5;
            line2[i].y = line->nodes[i].y + 0.5;
        }
        */

        Vec2F64 p1; //line2[0];
        p1.x = line->nodes[0].x + 0.5;
        p1.y = line->nodes[0].y + 0.5;
        
        //TP64 ll[100];// = new TP64[100];//line->len];
        TP64 * ll = (TP64*)malloc(30 * sizeof(TP64));
        ll[0].val = 0;
        ll[0].p.x = p1.x;
        ll[0].p.y = p1.y;

        for(int i=1;i<line->len;i++){
            Vec2F64 p2; // line2[i];
            p2.x = line->nodes[i].x + 0.5;
            p2.y = line->nodes[i].y + 0.5;
            l = l + ( (p1.x - p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) );
            ll[i].p.x = p2.x;
            ll[i].p.y = p2.y;
            ll[i].val = l;
            p1.x = p2.x;
            p1.y = p2.y;
        }
        //delete[] line2;

        double tot_len = ll[line->len - 1].val;
        double dl = tot_len / (double)cn_len;
        unsigned int m = 0;
        Vec2F64 p = ll[0].p;



        double sum = 0;
        sum += (cn[0].x - ll[0].p.x) * (cn[0].x - ll[0].p.x) + (cn[0].y - ll[0].p.y) * (cn[0].y - ll[0].p.y);
        //Vec2F64 * res = new Vec2F64[cn_len];
        //res[0].x = ll[0].p.x;
        //res[0].y = ll[0].p.y;
        //int res_len = 1;

        for(int k=1;k<cn_len;k++){
            double r = k * dl;
            while(m < line->len) {
                double l = ll[m].val;
                if(r < l){
                    // cur_path = r;
                    break;
                }
                p = ll[m].p;
                m += 1;
            }
            //     # px = (p2[0] - p1[0]) / l * dl # TODO!!!
            //     # py = (p2[1] - p1[1]) / l * dl # TODO!!!

            double dd = r - ll[m-1].val;
            //let (mut dx, mut dy): (f64, f64) = (0.0, 0.0);

            double dx,dy;
            if(abs(ll[m].p.x - ll[m-1].p.x) > 1.0e-10){
                double kk = (ll[m].p.y - ll[m-1].p.y) / (ll[m].p.x - ll[m-1].p.x);
                dx = dd / sqrt(1.0 + kk * kk);
                dy = kk * dx;
            } else {
                dx = 0.0;
                dy = dd;
            }
            dx = p.x + dx;
            dy = p.y + dy;
            sum += (cn[k].x - dx) * (cn[k].x - dx) + (cn[k].y - dy) * (cn[k].y - dy);
        }
        //delete[] ll;
        free(ll);
        return sum/ (double)cn_len;
}

 __device__ void ff_cuda(PolyLine * pl,T64_PolyLine * top_in_cntr,int& top_in_cntr_len,Vec2F64 * cntr,int cntrl_len,Vec3Float64 v_min,Vec3Float64 v_max) {
                double d = calc_sco_cuda(cntr,cntrl_len,pl,v_min,v_max);
                int len = top_in_cntr_len;
                //printf("ff_cuda %d %d\n",cntrl_len,pl->len);


                 if(len == 10){
                        double maxval=0;
                        int index=-1;
                       for(int i=0;i<10;i++){
                        if(maxval < top_in_cntr[i].val){
                            maxval = top_in_cntr[i].val;
                            index = i;
                        }
                       }
                        //T64_PolyLine temp;
                        top_in_cntr[index].val = d;
                        top_in_cntr[index].line.len = pl->len;
                        memcpy(top_in_cntr[index].line.nodes, pl->nodes , pl->len * 8);
                        //copyAVX2((char*)temp.line.nodes,(char*)pl->nodes,pl->len * sizeof(Vec2Int));
                        //top_in_cntr.push_back(temp);
                        
                        
                        /*
                        sort(top_in_cntr.begin(),top_in_cntr.end(),[](T64_PolyLine &a,T64_PolyLine &b){
                            return a.val > b.val;
                        });
                        */
                    
                } else {
                    
                    top_in_cntr[top_in_cntr_len].val = d;
                    top_in_cntr[top_in_cntr_len].line.len = pl->len;
                    memcpy(top_in_cntr[top_in_cntr_len].line.nodes, pl->nodes , pl->len * 8);
                    top_in_cntr_len += 1;
                    
                    
                    //top_in_cntr.push_back((d, pl.clone()));
                }
            
};



__device__ void complete_line_cuda(int &lev,int (&zone)[8][8],PolyLine * line,T64_PolyLine * top_in_cntr,int &top_in_cntr_len,Vec2F64 * cntr,int cntrl_len,Vec3Float64 v_min,Vec3Float64 v_max){
    lev ++;
    
    Vec2Int start_point = line->nodes[line->len-1];
    Vec2Int first_point = line->nodes[0];
    //Vec2Int v[(8 - 1) * 4];
    Vec2Int v[16];// = (Vec2Int*)malloc(sizeof(Vec2Int) * (8-1) * 4);
    int v_len = 0;

    near_points_cuda(zone,line,start_point,v,v_len);


    if(start_point.x == 0 && start_point.y == 0){
        int old_lev = lev;
        lev = v_len*128 + lev;
        if(v_len<2){
           //printf("skip\n");
           return;
        } else if(v_len == 2){
            Vec2Int p1,p2;
            p1 = v[0];
            p2 = v[1];
            if(abs(p1.x-p2.x) < 2 && abs(p1.y-p2.y) < 2){
                return;
            }
        } else {
            Vec2Int p1,p2;
            int good[8];
            for(int i=0;i<8;i++)
            good[i]=1;
            for(int i=0;i<v_len;i++){
                int g = 0;
                for(int j=0;j<v_len;j++){
                    if(i == j)
                        continue;
                    p1,p2;
                    p1 = v[i];
                    p2 = v[j];
                    if(abs(p1.x-p2.x) >= 2 || abs(p1.y-p2.y) >= 2){
                        g=1;
                        break;
                    }
                }
                if(g==0){
                    good[i] = 0;
                    //printf("not good %d\n",g);
                }
            }
            int good_len=0;
            for(int i=0;i<v_len;i++){
                if(good[i] == 0){
                    v[i].x = -1;
                } else 
                good_len++;
            }
            lev = good_len*128 + old_lev;
            //printf("%d\n",v_len);
        }
    }

     if(lev/128==2 && lev%128 == 2){
        for(int i=0;i<v_len;i++){
            if(v[i].x == -1)
                continue;
            char enc = v[i].x * 8 + v[i].y;
            if(
                (v[i].x == 0 && v[i].y == 2) ||
                (v[i].x == 1 && v[i].y == 2) ||
                (v[i].x == 2 && v[i].y == 2) ||
                (v[i].x == 2 && v[i].y == 0) ||
                (v[i].x == 2 && v[i].y == 1)
            ){
                v[i].x = -1;
            }
        }
    }
    

    /*
    1 * x
    * * x
    x x x
    */

    if(v_len >= 16){
        printf("ANOTHER OVERFLOW\n");
        return;
    }
    
    Vec2Int p;
    for(int i=0;i<v_len;i++){
        p = v[i];
        if(p.x == -1)
            continue;
        //printf("vecin: %d,%d\n",p.x,p.y);
        if(p.x == first_point.x && p.y == first_point.y) {
            // println!("line_buf: {:?}", self.line_buf.nodes);
            line->push(p);
            /*
            for(int j=0;j<line->len;j++){
                printf("(%d,%d) ",line->nodes[j].x,line->nodes[j].y);
            }
            printf("\n");
            */
            //printf("hehe %d %d\n",line->len,line->nodes[1].x);
            
            
            //if(line->len > 20)
            //printf("ok\n");
            ff_cuda(line,top_in_cntr,top_in_cntr_len,cntr,cntrl_len, v_min, v_max);
           
            //(*f)(&self.line_buf);
            
            line->pop();
            continue;
        }
        line->push(p);
        //sleep(1);
        complete_line_cuda(lev,zone,line,top_in_cntr,top_in_cntr_len,cntr,cntrl_len,v_min,v_max);
        //printf("pop\n");
        line->pop();
    }
    //lev--;
   // free(v);
}



//#define IN_DEBUG

__device__ void find_top_std_2_cuda(Vec2F64 * cntrs,int * cntrs_len,unsigned int depth,
    unsigned int n_sect, unsigned int grid_size,Vec3Float64 v_min,Vec3Float64 v_max , unsigned char * out_hashes, int &out_hashes_len){

    /*
    printf("Settings:\n");
    printf("grid size: %d\n",grid_size);
    printf("n_sect: %d\n",n_sect);
    printf("\n");
    */
   

    double width = v_max.x - v_min.x;
    double height = v_max.y - v_min.y;
    const int N = 2;
    out_hashes_len = 0;
    if(cntrs_len == 0){
        return;
    }
    Tuple_select_top_all ss[12][2];

    T64_PolyLine top_in_cntr[10];
    int top_in_cntr_len = 0;

    int ss_len=0;
    int ss_lens[12];
    for(int i=0;i<12;i++){
        ss_lens[i]=0;
    }

    /*
        calc ss
    */
    ss_len = 0;

    //int zone[8][10];
    int zoneF[8][8];

    double grid_dx = width / (double)grid_size;
    double grid_dy = height / (double)grid_size;
    double dx = 0.1 * grid_dx;
    double dy = 0.1 * grid_dy;
    short _len = 12;
    int psum = 0;

    for(int j=0;j<12;j++){
        top_in_cntr_len = 0;
        
        Vec2F64 * cntr = cntrs + psum;
        int cntr_len = cntrs_len[j+1];

        /*
        for(int i=0;i<8;i++){
            for(int jj=0;jj<10;jj++)
                zone[i][jj] = -1;
            zone[i][9] = 0;
        }
        */

        for(int i=0;i<8;i++)
            for(int j=0;j<8;j++)
                zoneF[i][j]=0;
        /*
            fn line_zone
        */
        for(int x = 0;x < cntr_len;x++){
            double px,py;
            px = cntr[x].x;
            py = cntr[x].y;
            int ii = (int)((px - v_min.x) / grid_dx);
			int jj = (int)((py - v_min.y) / grid_dy);
            if (ii > grid_size || jj > grid_size)
            continue;
			if(ii == grid_size){
				ii = grid_size - 1;
			}
			if(jj == grid_size) {
				jj = grid_size - 1;
			}

            zoneF[ii][jj]=1;

			double xx = round(px) - px;
		    double yy = round(py) - py;
            /*
            if(abs(xx) < dx) {
				if(xx >= 0.0) {
					chk_add(zone, grid_size, ii + 1, jj);
				} else {
					chk_add(zone, grid_size, ii - 1, jj);
				}
			}
			if (abs(yy) < dy){
				if(yy >= 0.0) {
					chk_add(zone, grid_size, ii, jj + 1);
				} else {
					chk_add(zone, grid_size, ii, jj - 1);
				}
			}
            if(abs(yy) < dy && abs(xx) < dx) {
				if (xx >= 0.0 && yy >= 0.0) {
					chk_add(zone, grid_size, ii + 1, jj + 1);
				}
				if(xx >= 0.0 && yy < 0.0){
					chk_add(zone, grid_size, ii + 1, jj - 1);
				}
				if(xx < 0.0 && yy >= 0.0) {
					chk_add(zone, grid_size, ii - 1, jj + 1);
				}
				if(xx < 0.0 && yy < 0.0) {
					chk_add(zone, grid_size, ii - 1, jj - 1);
				}
			}
            */
           if(abs(xx) < dx) {
				if(xx >= 0.0) {
					chk_add_cuda(zoneF, grid_size, ii + 1, jj);
				} else {
					chk_add_cuda(zoneF, grid_size, ii - 1, jj);
				}
			}
			if (abs(yy) < dy){
				if(yy >= 0.0) {
					chk_add_cuda(zoneF, grid_size, ii, jj + 1);
				} else {
					chk_add_cuda(zoneF, grid_size, ii, jj - 1);
				}
			}
            if(abs(yy) < dy && abs(xx) < dx) {
				if (xx >= 0.0 && yy >= 0.0) {
					chk_add_cuda(zoneF, grid_size, ii + 1, jj + 1);
				}
				if(xx >= 0.0 && yy < 0.0){
					chk_add_cuda(zoneF, grid_size, ii + 1, jj - 1);
				}
				if(xx < 0.0 && yy >= 0.0) {
					chk_add_cuda(zoneF, grid_size, ii - 1, jj + 1);
				}
				if(xx < 0.0 && yy < 0.0) {
					chk_add_cuda(zoneF, grid_size, ii - 1, jj - 1);
				}
			}
            /*
            bool ff = false;
            for(int o=0;o<zone[ii][9];o++){
                if(zone[ii][o] == jj){
                    ff= true;
                    break;
                }
            }
            if(ff==false){
                zone[ii][zone[ii][9]] = jj;
                zone[ii][9]++;
            }

			double xx = round(px) - px;
		    double yy = round(py) - py;
            if(abs(xx) < dx) {
				if(xx >= 0.0) {
					chk_add_cuda(zone, grid_size, ii + 1, jj);
				} else {
					chk_add_cuda(zone, grid_size, ii - 1, jj);
				}
			}
			if (abs(yy) < dy){
				if(yy >= 0.0) {
					chk_add_cuda(zone, grid_size, ii, jj + 1);
				} else {
					chk_add_cuda(zone, grid_size, ii, jj - 1);
				}
			}
            if(abs(yy) < dy && abs(xx) < dx) {
				if (xx >= 0.0 && yy >= 0.0) {
					chk_add_cuda(zone, grid_size, ii + 1, jj + 1);
				}
				if(xx >= 0.0 && yy < 0.0){
					chk_add_cuda(zone, grid_size, ii + 1, jj - 1);
				}
				if(xx < 0.0 && yy >= 0.0) {
					chk_add_cuda(zone, grid_size, ii - 1, jj + 1);
				}
				if(xx < 0.0 && yy < 0.0) {
					chk_add_cuda(zone, grid_size, ii - 1, jj - 1);
				}
			}*/
        }
    
        
        /*
            fn complete_line
        */
        PolyLine line;
        line.len = 0;
        line.push(0,0);

        
        /*
            fn NeiborNodes:new
        */
        int lev=0;

    

        int precalc =0;

    lev=0;
    complete_line_cuda(lev,zoneF, &line, top_in_cntr,top_in_cntr_len, cntr, cntr_len,v_min,v_max);
        #ifdef IN_DEBUG
        printf("init  %d\n",top_in_cntr_len);
        #endif
        for(int ii=0;ii<top_in_cntr_len && ii < 2;ii++){
            T64_PolyLine * a = &top_in_cntr[ii];
            //printf("ss len of %d -> %d\n",j,ss_lens[j]);
            ss[j][ss_lens[j]].val= a->val;

            #ifdef IN_DEBUG
            printf("val: %lf\n",a->val);
            #endif
            for(int jj=0;jj<a->line.len;jj++){
                #ifdef IN_DEBUG
                printf("(%d,%d) ",a->line.nodes[jj].x,a->line.nodes[jj].y);
                #endif
                int xx = a->line.nodes[jj].x;
                int yy = a->line.nodes[jj].y;
                a->line.nodes[jj].x = (xx<<24) | (xx<<8 & 0xff0000) | (xx>>8 & 0xff00) | (xx>>24);
                a->line.nodes[jj].y = (yy<<24) | (yy<<8 & 0xff0000) | (yy>>8 & 0xff00) | (yy>>24);
                /*
                a->line.nodes[jj].x = int_to_be(xx);
                a->line.nodes[jj].y = int_to_be(yy);
                */
                //int_to_be(xx,(unsigned char*)a->line.nodes[jj].x);
                //int_to_be(yy,(unsigned char*)a->line.nodes[jj].y);
            }
            #ifdef IN_DEBUG
            printf("\n");
            #endif

           
           
            CUDA_SHA256_CTX sha_ctx;
            cuda_sha256_init(&sha_ctx);
            cuda_sha256_update(&sha_ctx,(unsigned char*)a->line.nodes,8 * a->line.len);
            cuda_sha256_final(&sha_ctx,ss[j][ss_lens[j]].hash);
            
          
            ss_lens[j]++;
            //printf("\n");
        }
        ss_len++;
        

        psum += cntrs_len[j+1];

    }
    

    int okk=0;
    for(int j=0;j<12;j++){
        if(ss_lens[j]>0){
            okk=1;
        }

    }

    #ifdef IN_DEBUG
    for(int j=0;j<12;j++){
        printf("ss #%d\n",j);
        for(int jj=0;jj<ss_lens[j];jj++){
            for(int jjj=0;jjj<32;jjj++)
                printf("%d ",ss[j][jj].hash[jjj]);
            printf("\n");
        }
        printf("\n");
    }
    #endif
    
    if(okk==0){
        out_hashes_len=-1;
        return;
    }



    F64_Hash384 best_totals2[1];
    best_totals2[0].hash_len=0;
    int len_best_totals2=0;

    #define BEST_HASH_MODE_TURBO



    
    double expect = 0;
    for(int i=0;i<12;i++){
        //printf("lev = %d\n",i);
        if(ss_lens[i] > 0){
            if(ss[i][0].val < ss[i][1].val){
                memcpy(best_totals2[0].hash + best_totals2[0].hash_len,ss[i][0].hash,32);
            } else {
                memcpy(best_totals2[0].hash + best_totals2[0].hash_len,ss[i][1].hash,32);
            }
            best_totals2[0].hash_len += 32;
            expect += min(ss[i][0].val,ss[i][1].val);
        }
        /*
        for(int j=0;j<ss_lens[i];j++){
            printf("%lf,",ss[i][j].val);
        }
        printf("\n\n");
        */
    }
    //printf("expect = %lf\n",expect);

    for(int i=0;i<1;i++){
        F64_Hash384 * hash = &best_totals2[i]; //best_totals.get(i);
        CUDA_SHA256_CTX sha_ctx;
            cuda_sha256_init(&sha_ctx);
            /*
            */
            
            cuda_sha256_update(&sha_ctx,(unsigned char*)hash->hash,hash->hash_len);
            cuda_sha256_final(&sha_ctx,out_hashes+(out_hashes_len*32));
            //printf("%lf\n",hash->val);
            out_hashes_len++;
    }



}
