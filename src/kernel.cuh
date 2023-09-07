void test();

struct GPUSolution{
    RockObjParams param;
    unsigned char obj_hash[32];
};

void initGpuData(int x,int blocks,int threads,int sp_stacks,int sp_slices);
vector<GPUSolution> doGpuBatch(int x,int blocks,int threads,unsigned char * outhash,int * outlen,unsigned char * best_hash,unsigned char * pre_hash,unsigned char * diffBytes,unsigned char * cmpBytes);
