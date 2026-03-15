#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#define MAX_DOCS 1000000
#define MAX_LEN 256
#define MAX_NODES 1000000
#define MAX_PARAMS 200000
#define MAX_LAYER 4
#define MAX_EMBD 64
#define MAX_BLOCK_SIZE 64
#define MAX_VOCAB_SIZE 512
#define MAX_FF (MAX_EMBD * 4)
//docs 数据缓存
char *docs[MAX_DOCS];
int num_docs = 0;
//tokenizer 变量
char uchars[256];
int vocab_size = 0;
int BOS = 0;
int charset[256] = {0};
//统一打印错误
void die(const char *msg){
    fprintf(stderr, "%s\n", msg);
    exit(1);
}
//确定是否有训练语料
int file_exists(const char *filename){
    FILE *f = fopen(filename, "r");
    if(f){
        fclose(f);
        return 1;
    }
    return 0;
}
//没有训练语料就自动下载并检查是否成功
void download_file(void) {
    int ret = system(
        "curl -L "
        "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt "
        "-o input.txt");
    if (ret != 0) {
    die("download failed");
    }
}
//将训练语料写入docs
void load_docs(void) {
    FILE *f = fopen("input.txt", "r");
    char buffer[MAX_LEN];
    if (!f) {
        perror("open failed");
        return;
    }
    while (fgets(buffer, MAX_LEN, f)) {
        buffer[strcspn(buffer, "\n")] = '\0';
        if (strlen(buffer) == 0) {
            continue;
        }
        if (num_docs >= MAX_DOCS) {
            fprintf(stderr, "MAX_DOCS reached\n");
            fclose(f);
            return;
        }
        docs[num_docs] = strdup(buffer);
        if (!docs[num_docs]) {
            perror("strdup failed");
            fclose(f);
            return;
        }
        num_docs++;
    }
    fclose(f);
}
//将语料随机打乱
void shuffle_docs(){
    for(int i = num_docs-1; i > 0 ; i--){
        int j = rand() % (i+1);

        char *tmp = docs[i];
        docs[i] = docs[j];
        docs[j] = tmp;
    }
}
//构建tokenizer第一步:构建无重复词表
void build_vocab(void) {
    int i, j;
    for (i = 0; i < num_docs; i++) {
        char *s = docs[i];
        for (j = 0; s[j]; j++) {
            unsigned char c = (unsigned char)s[j];
            charset[c] = 1;
        }
    }
    vocab_size = 0;
    for (i = 0; i < 256; i++) {
        if (charset[i]) {
            uchars[vocab_size++] = (char)i;
        }
    }
    if (vocab_size + 1 > MAX_VOCAB_SIZE) {
        fprintf(stderr, "vocab too large for MAX_VOCAB_SIZE\n");
        exit(1);
    }
    BOS = vocab_size;
    vocab_size += 1;
}
//创建value
typedef struct Value{
    double data;
    double grad;
    struct Value *children[2];
    double local_grads[2];
    int n_children;
    int topo_mark;
} Value;
//为反向传播做准备
Value *topo[MAX_NODES];
int topo_size = 0;
int current_topo_mark = 1;
//自动垃圾回收
Value *all_values[MAX_NODES];
int all_values_size = 0;
//创建Value节点带报错
Value *value_create(
    double data, 
    int n_children,
    Value *c0,
    Value *c1,
    double g0,
    double g1){
        Value *v;
        if (all_values_size >= MAX_NODES){
            die("MAX_NODES reached while creating Value");
        }
        v = (Value *)malloc(sizeof(Value));
        if (!v){
            die("malloc failed in value_create");
        }
        v ->data = data;
        v ->grad = 0.0;
        v ->children[0] = c0;
        v ->children[1] = c1;
        v ->local_grads[0] = g0;
        v ->local_grads[1] = g1;
        v ->n_children = n_children;
        v ->topo_mark = 0;
        all_values[all_values_size++] = v;
        return v;
    }
//创建一个叶子节点,没children
Value *value_leaf(double data){
    return value_create(data,0 , NULL, NULL, 0.0, 0.0);
}
//实现加法
Value *value_add(Value *self, Value *other){
    return value_create(self->data + other->data, 2, self, other, 1.0, 1.0);
}
//实现乘法
Value *value_mul(Value *self, Value *other) {
    return value_create(self->data * other->data, 2, self, other, other->data, self->data);
}
//实现幂运算
Value *value_pow(Value *self , double other){
    return value_create(pow(self->data, other), 1, self, NULL, other * pow(self->data, other - 1.0), 0.0);
}
//实现对数log
Value *value_log(Value *self){
    return value_create(log(self->data), 1, self, NULL, 1.0 / self->data, 0.0);
}
//实现指数exp
Value *value_exp(Value *self){
    double out = exp(self->data);
    return value_create(out, 1, self, NULL, out, 0.0);
}
//创建ReLU
Value *value_relu(Value *self){
    double out = self->data > 0.0 ? self->data : 0.0;
    double grad = self->data > 0.0 ? 1.0 : 0.0;
    return value_create(out, 1, self, NULL, grad, 0.0); 
}
//变成相反数
Value *value_neg(Value *self){
    return value_mul(self, value_leaf(-1.0));
}
//实现右加(right add)
Value *value_radd_scalar(double other, Value *self){
    return value_add(self, value_leaf(other));
}
//实现减法
Value *value_sub(Value *self, Value *other){
    return value_add(self, value_neg(other));
}
//实现右减
Value *value_rsub_scalar(double other, Value *self) {
    return value_add(value_leaf(other), value_neg(self));
}
//实现右乘
Value *value_rmul_scalar(double other, Value *self) {
    return value_mul(self , value_leaf(other));
}
//实现相除
Value *value_div(Value *self, Value *other){
    return value_mul(self, value_pow(other, -1.0));
}
//实现右除
Value *value_rdiv_scalar(double other, Value *self) {
    return value_mul(value_leaf(other), value_pow(self, -1.0));
}
//把图里的所有节点按顺序收集起来(DFS)
void build_topo(Value *v) {
    int i;
    if (v->topo_mark == current_topo_mark) {
        return;
    }
    v->topo_mark = current_topo_mark;
    if (topo_size >= MAX_NODES) {
        die("MAX_NODES reached in bulid_topo");
    }
    for (i = 0; i < v->n_children ; i++) {
        build_topo(v->children[i]);
    }
    topo[topo_size++] = v;
}
//开始从最后的loss进行反向传播，把每个节点的 grad 都算出来
void value_backward(Value *self) {
    int i, j;
    topo_size = 0;
    if(current_topo_mark == INT_MAX) {
        for (i = 0; i < all_values_size; i++){
            all_values[i]->topo_mark = 0;
        }
        current_topo_mark = 1;
    }else{
        current_topo_mark++;
    }
    build_topo(self);
    self->grad = 1.0;  
    for(i = topo_size - 1; i >= 0; i--){
        Value *v = topo[i];
        for(j = 0;j < v->n_children; j++){
            v->children[j]->grad += v->local_grads[j] * v->grad;
        }
    }
}
//每一步结束就后就清垃圾
void free_temp_values(int start_index) {
    int i;
    if (start_index < 0 || start_index > all_values_size) {
        die("invalid start_index in free_temp_values");
    }
    for (i = start_index; i < all_values_size; i ++) {
        free(all_values[i]);
    }
    all_values_size = start_index;
}
//程序结束时全清理
void free_all_values(void){
    free_temp_values(0);
}
//开始设置模型大小超参数
int n_layer = 1;
int n_embd = 16;
int block_size = 16;
int n_head = 4;
int head_dim = 0;
//用一个简单的二维矩阵容器装权重
typedef struct Matrix {
    int rows;
    int cols;
    Value **data;
}Matrix;
//把所有权重矩阵打包成一个模型
typedef struct Model {
    Matrix wte;
    Matrix wpe;
    Matrix lm_head;
    Matrix attn_wq[MAX_LAYER];
    Matrix attn_wk[MAX_LAYER];
    Matrix attn_wv[MAX_LAYER];
    Matrix attn_wo[MAX_LAYER];
    Matrix mlp_fc1[MAX_LAYER];
    Matrix mlp_fc2[MAX_LAYER];
}Model;
//实现KV缓存：保存每层历史keys/values
typedef struct KVCache {
    Value *keys[MAX_LAYER][MAX_BLOCK_SIZE][MAX_EMBD];
    Value *values[MAX_LAYER][MAX_BLOCK_SIZE][MAX_EMBD];
    int len[MAX_LAYER];
} KVCache;
//开始保存模型权重，参数列表，Adam需要的缓冲区
Model model;
Value *params[MAX_PARAMS];
int num_params = 0;
double *adam_m = NULL;
double *adam_v = NULL;
//抽一个 0~1 的随机数(用于高斯采样)
double rand_uniform01(void) {
    return (rand() + 1.0) / ((double)RAND_MAX + 2.0);
}
//生成权重用的随机数
double rand_normal(double std){
    double u1 = rand_uniform01();
    double u2 = rand_uniform01();
    double r = sqrt(-2.0 * log(u1));
    double theta = 6.28318530717958647692 * u2;
    return (r * cos(theta)) * std;
}
//取矩阵里的一个数
Value *matrix_at(Matrix *m, int r, int c){
    return m->data[r * m->cols + c];
}
//创建一张随机权重表
Matrix matrix_create(int rows, int cols, double std){
    Matrix m;
    int i;
    size_t n = (size_t)rows * (size_t)cols;
    m.rows = rows;
    m.cols = cols;
    m.data = (Value **)malloc(n * sizeof(Value *));
    if (!m.data){
        die("malloc failed in matrix_create");
    }
    for (i = 0; i < (int)n; i++){
        m.data[i] = value_leaf(rand_normal(std));
    }
    return m;
}
//把权重都登记到params
void matrix_append_params(Matrix *m){
    int i;
    int n = m->rows * m->cols;
    for (i = 0; i < n; i++){
        if(num_params >= MAX_PARAMS){
            die("MAX_PARAMS reached");
        }
        params[num_params++] = m->data[i];
    }
}
//开始释放这张表的指针内存
void matrix_free(Matrix *m){
    free(m->data);
    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
}
//开始把所有权重随机初始化
void model_init(Model *m){
    int li;
    m->wte = matrix_create(vocab_size, n_embd, 0.08);
    m->wpe = matrix_create(block_size, n_embd, 0.08);
    m->lm_head = matrix_create(vocab_size, n_embd, 0.08);
    for (li = 0;li < n_layer; li++){
        m->attn_wq[li] = matrix_create(n_embd, n_embd, 0.08);
        m->attn_wk[li] = matrix_create(n_embd, n_embd, 0.08);
        m->attn_wv[li] = matrix_create(n_embd, n_embd, 0.08);
        m->attn_wo[li] = matrix_create(n_embd, n_embd, 0.08);
        m->mlp_fc1[li] = matrix_create(4 * n_embd, n_embd, 0.08);
        m->mlp_fc2[li] = matrix_create(n_embd, 4 * n_embd, 0.08);
    }
}
//把所有的权重指针排成一条队
void model_collect_params(Model *m){
    int li;
    num_params = 0;
    matrix_append_params(&m->wte);
    matrix_append_params(&m->wpe);
    matrix_append_params(&m->lm_head);

    for(li = 0 ; li < n_layer ; li++){
        matrix_append_params(&m->attn_wq[li]);
        matrix_append_params(&m->attn_wk[li]);
        matrix_append_params(&m->attn_wv[li]);
        matrix_append_params(&m->attn_wo[li]);
        matrix_append_params(&m->mlp_fc1[li]);
        matrix_append_params(&m->mlp_fc2[li]);
    }   
}
//释放模型里每个矩阵的指针数组（不释放 Value，本文件最后统一释放）
void model_free_matrices(Model *m) {
    int li;
    matrix_free(&m->wte);
    matrix_free(&m->wpe);
    matrix_free(&m->lm_head);
    for(li = 0; li < n_layer; li++){
        matrix_free(&m->attn_wq[li]);
        matrix_free(&m->attn_wk[li]);
        matrix_free(&m->attn_wv[li]);
        matrix_free(&m->attn_wo[li]);
        matrix_free(&m->mlp_fc1[li]);
        matrix_free(&m->mlp_fc2[li]);
    }
}
//做一次"矩阵乘向量" ;线性层: y = W x (用 Value 运算来搭计算图)
void linear(Value **x, int x_len, Matrix *w , Value **out){
    int r , c ;
    if(x_len != w->cols || x_len <= 0){
        die("liner shape mismatch");
    }
    for(r = 0; r < w->rows; r++){
        Value *acc = value_mul(matrix_at(w, r, 0), x[0]);
        for(c=1; c < w->cols ; c++){
            acc = value_add(acc, value_mul(matrix_at(w, r, c), x[c]));
        }
        out[r] = acc;
    }
}
//把分数变成概率 (把logits变成概率)(每个都>=0, 且总和为1 )
void softmax(Value **logits, int n, Value **probs){
    int i;
    double max_val;
    Value *exps[MAX_VOCAB_SIZE];
    Value *total;
    if (n <= 0 || n > MAX_VOCAB_SIZE) {
        die("softmax size invalid");
    }
    max_val = logits[0] -> data;
    for (i = 1 ; i < n ; i++ ){
        if (logits[i]->data > max_val){
            max_val = logits[i]->data;
        }
    }
    for (i = 0 ; i < n; i++){
        exps[i] = value_exp(value_sub(logits[i], value_leaf(max_val)));
    }
    total = exps[0];
    for (i = 1 ; i < n ; i++){
        total = value_add(total, exps[i]);
    }
    for (i = 0; i <n ; i++){
        probs[i] = value_div(exps[i], total);
    }
}
//把向量整数缩放一下,让数值更稳定(把这一排数"调匀")
void rmsnorm(Value **x, int n , Value **out){
    int i;
    Value *ms;
    Value *scale;
    if (n <= 0){
        die("rmsnorm size  invalid");
    }
    ms = value_mul(x[0],x[0]);
    for (i = 1; i < n; i++){
        ms = value_add(ms, value_mul(x[i],x[i]));
    }
    ms = value_div(ms, value_leaf((double)n));
    scale = value_pow(value_add(ms, value_leaf(1e-5)), -0.5);
    for(i = 0; i< n; i++){
        out[i] = value_mul(x[i],scale);
    }
}
//每个新序列开始前清空KVcache(等价于重新创建 key/values 的空列表)
//把历史缓存清空
void kv_cache_reset(KVCache *cache){
    int li;
    for (li = 0 ; li < n_layer; li++){
        cache->len[li] = 0;
    }
}
//做GTP前向:输入token_id和位置pos_id,输入下一个字符的logits.(给下一个字符打分)
void gpt(Model *m, int token_id, int pos_id, KVCache *cache, Value **logits_out){
    int i, li, h , j , t;
    Value *x[MAX_EMBD];
    Value *x_norm[MAX_EMBD];
    Value *x_residual[MAX_EMBD];
    Value *q[MAX_EMBD];
    Value *k[MAX_EMBD];
    Value *v[MAX_EMBD];
    Value *x_attn[MAX_EMBD];
    Value *mlp_hidden[MAX_FF];    
    if(token_id < 0 || token_id >= vocab_size){
        die("token_id out of range");
    }
    if(pos_id < 0 || pos_id >= block_size){
        die("pos_id out of range ");
    }
    for(i=0; i< n_embd ; i++){
        x[i] = value_add(matrix_at(&m->wte, token_id, i), matrix_at(&m->wpe, pos_id,i));
    }
    rmsnorm(x, n_embd, x_norm);
    for(i = 0; i < n_embd; i++){
        x[i] = x_norm[i];
    }
    for(li = 0; li < n_layer; li++){
        for (i = 0; i < n_embd; i++){
            x_residual[i] = x[i];
        }
        rmsnorm(x, n_embd, x_norm);
        for(i = 0; i < n_embd; i++){
            x[i] = x_norm[i];
        }
        linear(x, n_embd, &m->attn_wq[li], q);
        linear(x, n_embd, &m->attn_wk[li], k);
        linear(x, n_embd, &m->attn_wv[li], v);
        t = cache->len[li];
        if(t >= block_size || t>=MAX_BLOCK_SIZE){
            die("KV cache is full");
        }
        for (i=0; i< n_embd; i++){
            cache->keys[li][t][i] = k[i];
            cache->values[li][t][i] = v[i];
        }
        cache->len[li] = t + 1;
        for (h = 0 ; h < n_head; h++){
            int hs = h * head_dim;
            int T = cache->len[li];
            Value *attn_logits[MAX_BLOCK_SIZE];
            Value *attn_weights[MAX_BLOCK_SIZE];
            double inv_sqrt_head = 1.0 / sqrt((double)head_dim);
            for (t = 0; t < T; t++){
                Value *score = value_mul(q[hs], cache->keys[li][t][hs]);
                for(j = 1; j < head_dim; j++){
                    score = value_add(score, value_mul(q[hs + j], cache->keys[li][t][hs + j]));
                }
                attn_logits[t] = value_mul(score , value_leaf(inv_sqrt_head));
            }
            softmax(attn_logits , T, attn_weights);
            for(j = 0; j< head_dim; j++){
                Value *sum_head = value_mul(attn_weights[0],cache->values[li][0][hs + j]);
                for (t = 1; t < T; t++) {
                    sum_head = value_add(sum_head, value_mul(attn_weights[t], cache->values[li][t][hs + j]));
                }
                x_attn[hs + j] = sum_head;
            
            }
        }
        linear(x_attn, n_embd, &m->attn_wo[li], x_norm);
        for(i = 0; i< n_embd; i++){
            x[i] = value_add(x_norm[i], x_residual[i]);
        }
        for (i = 0; i < n_embd; i++){
            x_residual[i] = x[i];
        }    
        rmsnorm(x, n_embd, x_norm);
        for (i = 0; i< n_embd ; i++){
            x[i] = x_norm[i];
        }
        linear(x , n_embd, &m->mlp_fc1[li], mlp_hidden);
        for(i= 0; i < 4 * n_embd ; i++){
            mlp_hidden[i] = value_relu(mlp_hidden[i]);
        }
        linear(mlp_hidden, 4 * n_embd , &m->mlp_fc2[li], x_norm);
        for(i = 0; i< n_embd ; i++){
            x[i] = value_add(x_norm[i], x_residual[i]);
        }
    }
    linear(x , n_embd , &m->lm_head, logits_out);
}
// 把一行文本编码成 tokens，并在两端加 BOS
int encode_doc_tokens(const char *doc, int *tokens, int max_tokens) {
    int i;
    int n = 0;
    if (max_tokens < 2) {
        die("max_tokens too small");
    }
    tokens[n++] = BOS;
    for (i = 0; doc[i]; i++) {
        int k;
        int tok = -1;
        for (k = 0; k < vocab_size - 1; k++) {
            if ((unsigned char)uchars[k] == (unsigned char)doc[i]) {
                tok = k;
                break;
            }
        }
        if (tok < 0) {
            continue;
        }
        if (n >= max_tokens - 1) {
            break;
        }
        tokens[n++] = tok;
    }
    tokens[n++] = BOS;
    return n;
}

//按概率抽一个下标(编号)
int sample_from_probs(Value **probs, int n){
    int i;
    double r = rand_uniform01();
    double cdf = 0.0;
    for (i = 0; i < n; i++){
        cdf += probs[i]->data;
        if(r<=cdf) {
            return i;
        }
    }
    return n - 1;
}
//进入主流程! : 跑完整流程
//(准备数据 -> 初始化模型/优化器 -> 训练 -> 推理 -> 释放资源)
int main(void) {
    int i;
    int num_steps =         1000;
    int step;
    double beta1 =          0.85;
    double beta2 =          0.99;
    double eps_adam =       1e-8;
    double learning_rate =  0.01;
    //让随机数每次都一样,保证每次运行更可复现
    srand(42);
    //开始准备训练数据
    if (!file_exists("input.txt")){
        download_file();
    }
    load_docs();
    shuffle_docs();
    printf("num docs: %d\n", num_docs);
    //开始构建tokenizer , 统计字符表并加上BOS
    build_vocab();
    printf("vocab size: %d\n", vocab_size);
    //开始检查超参数是否合法 , 并计算 head_dim
    if(n_layer <= 0 || n_layer  > MAX_LAYER) die("n_layer out of range");
    if(n_embd <= 0 || n_embd > MAX_EMBD) die("n_embd out of range");
    if(block_size <= 0 || block_size > MAX_BLOCK_SIZE) die("block_size out of range");
    if(n_head <= 0 || n_head > n_embd) die("n_head out of range");
    if(n_embd % n_head != 0) die("n_embd must be divisible by n_head");
    if(vocab_size > MAX_VOCAB_SIZE) die("vocab_size exceeds Max_vocab_size");
    if(4 * n_embd > MAX_FF) die("4*n_embd exceeds MAX_FF");
    head_dim = n_embd / n_head;
    //初始化模型权重, 并收集所有参数到 params 列表(把权重建出来并排成一条队)
    model_init(&model);
    model_collect_params(&model);
    printf("num params: %d\n", num_params);
    //初始化 Adam 的两个缓冲区 (每个参数各有一份)
    //给每个参数准备两本 "账本"
    adam_m = (double *)calloc((size_t)num_params, sizeof(double));
    adam_v = (double *)calloc((size_t)num_params, sizeof(double));
    if(!adam_m || !adam_v){
        die("calloc failed for adam buffers");
    }
    //重复训练很多步(取一条 doc -> 前向算 loss -> backward -> Adam 更新)
    for(step = 0 ; step < num_steps; step++ ){
        int graph_start = all_values_size;
        char *doc = docs[step % num_docs];
        int tokens[MAX_LEN + 2];
        int token_count = encode_doc_tokens(doc, tokens, MAX_LEN + 2);
        int n = block_size;
        KVCache cache;
        Value *losses[MAX_BLOCK_SIZE];
        Value *loss;
        double lr_t;
        //查看多少个字符(不超过 block_size，也不超过 tokens-1)
        if (n > token_count - 1){
            n = token_count - 1;
        }    
        if (n <= 0){
            free_temp_values(graph_start);
            continue;
        }
        //每个新的序列都要清空 keys/values 的缓存长度
        kv_cache_reset(&cache);
        //逐位置计算 loss_t , 并收集到 losses[]  (计算每一个位置的损失)
        for (i = 0; i < n ; i++){
            int token_id = tokens[i];
            int target_id = tokens[i + 1];
            Value *logits[MAX_VOCAB_SIZE];
            Value *probs[MAX_VOCAB_SIZE];
            gpt(&model, token_id , i , &cache, logits);
            softmax(logits , vocab_size , probs);
            losses[i] = value_neg(value_log(probs[target_id]));
        }  
        //开始计算平均损失: loss = (1/n) * sum(losses)
        loss = losses[0];
        for(i = 1; i < n ; i++){
            loss = value_add(loss , losses[i]);
        }
        loss = value_mul(loss, value_leaf(1.0 / (double)n));
        //开始把梯度传回到所有参数
        value_backward(loss);
        //Adam更新 , 用梯度更新每一个参数 , 并清零 grad
        lr_t = learning_rate * (1.0 - (double)step / (double)num_steps);
        for (i = 0 ; i < num_params ; i++){
            double g = params[i] ->grad;
            double m_hat;
            double v_hat;
            adam_m[i] = beta1 * adam_m[i] + (1.0 - beta1) * g;
            adam_v[i] = beta2 * adam_v[i] + (1.0 - beta2) * g * g ;
            m_hat = adam_m[i] / (1.0 - pow(beta1, (double)(step + 1)));
            v_hat = adam_v[i] / (1.0 - pow(beta2, (double)(step + 1)));
            params[i]->data -= lr_t * m_hat / (sqrt(v_hat) + eps_adam);
            params[i]->grad = 0.0;
        }
        printf("step %4d / %4d | loss %.4f\r", step + 1, num_steps, loss->data);
        fflush(stdout);
        //释放本 step 新建的临时计算图节点，只保留参数节点(丢掉临时节点)
        free_temp_values(graph_start);   
    }
    //开始推理生成新名字(从 BOS 开始一位一位的采样字符)(生成一些新名字)
    printf("\n--- inference (new , hallucinated names) ---\n");
    for (step = 0 ; step < 20 ; step++) {
        int graph_start = all_values_size;
        KVCache cache;
        int token_id = BOS;
        char sample[MAX_BLOCK_SIZE + 1];
        int sample_len = 0;
        //每个 sample 都是一个新序列，所以缓存要清空
        kv_cache_reset(&cache);
        //逐位置生成下一个 token，直到生成 BOS 或达到 block_size
        for(i = 0; i < block_size; i++){
            Value *logits[MAX_VOCAB_SIZE];
            Value *probs[MAX_VOCAB_SIZE];
            int j;
            gpt(&model, token_id, i , &cache, logits);
            //开始控制随机程度
            for(j = 0; j < vocab_size; j++) {
                logits[j] = value_div(logits[j], value_leaf(0.5));
            }
            softmax(logits, vocab_size, probs);
            //按概率抽下一个 token，并把 token 转回字符追加到 sample
            token_id = sample_from_probs(probs, vocab_size);
            if (token_id == BOS){
                break;
            }     
            if (sample_len < MAX_BLOCK_SIZE) {
                sample[sample_len++] = uchars[token_id];
            }
        }
        sample[sample_len] = '\0';
        printf("sample %2d: %s\n", step + 1, sample);
        //推理这一步也会创建临时 Value，结束后同样要释放
        free_temp_values(graph_start);
    }
    //释放所有资源：docs 字符串、Adam 缓冲、模型矩阵指针、以及所有 Value
    for(i = 0; i < num_docs ; i++){
        free(docs[i]);
    }
    free(adam_m);
    free(adam_v);
    model_free_matrices(&model);
    free_all_values();
    return 0;
}