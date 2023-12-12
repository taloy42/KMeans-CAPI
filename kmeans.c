/*
the kmeans algorithm and its integration into C-API
*/
#include <stdlib.h>
#include <assert.h>
#include <Python.h>

#define MAX 2147483647

typedef double T;

struct LL{
    T* data;
    struct LL* next;
};
typedef struct LL LL;


static LL* init_d(int);
static int add(LL**, T*);
static void del_head(LL**);


static T distance(const T *,const T *,int);
static void addInPlace(T*,const T*,int);
static double updateCentroid(T** centroid, LL** cluster,int d);
static int findClosestCentroid(T* v, T** centroids,int K, int d);
static void divideInPlace(T*,double,int);

static int* K_MEANS(T** observations,T** centroids,int K,int N, int d,int MAX_ITER);



/*AUXILIARY*/
static T distance(const T *p,const T *q,int d){
/**
* calculate the euclidian distance between 2 points
* p: point
* q: point
* d: dimension of p and q
**/
    int i;
    T cur;
    T sum = 0;
    for(i=0;i<d;i++){
        cur = p[i]-q[i];
        sum+=cur*cur;
    }
    return sum;
}

static void addInPlace(T* x, const T* y,int d){
/**
* add vector y to vector x inplace ( of x )
* x: point
* y: point
* d: dimension of x and y
**/
    int i=0;
    for(;i<d;++i){
        x[i]+=y[i];
    }
}

static void divideInPlace(T* r, double a,int d){
/**
* divide vector by scalar inplace
* r: point
* a: scalar
* d: dimension of r
**/
    int i=0;
    for(;i<d;++i){
        r[i]/=a;
    }
}

static int findClosestCentroid(T* v, T** centroids,int K, int d){
/**
* find centroids that is closest to vector
* v: vector
* centroids: list of centroids
* K: length of centroids
* d: dimension of all vectors
* returns: index of the wanted centroid
**/
    double min=MAX,dist;
    int min_index=0;
    int i;

    for (i = 0; i<K ; i++) {
        dist = distance(v,centroids[i],d);
        if (dist<min){
            min = dist;
            min_index = i;
        }
    }

    return min_index;
}

static double updateCentroid(T** centroid, LL** cluster, int d){
/**
* update the centroid of a cluster
* centroid: vector
* cluster: list of vectors
* d: dimension of centroid
* returns: whether the centroid changed or not
**/
    Py_ssize_t i,size=0;
    T equal;
    T* v= (T*)calloc(d,sizeof(T));
    //assert(v!=NULL);
    if (v==NULL){
        return -1;
    }

    for (i=0;i<d;++i){
        v[i]=0;
    }


    while((*cluster)->next!=0){
        addInPlace(v,(*cluster)->data,d);
        del_head(cluster);
        size++;
    }
    divideInPlace(v,size,d);
    equal=distance(*centroid,v,d);
    if(size!=0) {
        *centroid = v;
    }
    else{
        equal=0;
    }
    return equal;
}



/*LINKED LIST*/
static LL* init_d(int d){
/**
* initialize a linked list with vector of size d
* d: dimension of vector
* returns: pointer to the head of the list
**/
    LL *list = (LL *) malloc(sizeof(LL));
    //assert(list!=NULL);
    if (list==NULL){
        return NULL;
    }

    list->next = 0;
    list->data = (T*)calloc(d,sizeof(T));
    return list;
}
static int add(LL **l, T* e){
/**
* append vector e to linked list l
* l: linked list
* e: vector
**/
    LL *new = (LL*)malloc(sizeof(LL));
    //assert(new!=NULL);
    if (new==NULL){
        return 1;
    }
    new->data=e;
    new->next=*l;
    *l=new;
    return 0;
}

static void del_head(LL **list){
/**
* delete the first item in list
* list: list
**/
    LL *l = (*list);
    (*list)=(*list)->next;
    free(l);
}


/*K_MEANS*/
static int* K_MEANS(T** observations,T** centroids,int K, int N, int d, int MAX_ITER) {
/**
* kmeans
* observations: list of vectors
* centroids: list of starting centroids
* K: number of centers for the data
* N: number of observations
* d: dimension of the points
* MAX_ITER: maximum number of iterations before stopping if convergence hasn't occurred
* returns: list of indexes such that indexes[i] is the cluster of observations[i]
**/
    int i, iters = 0, insert_index, status;
    double changed=1, tmp;
    LL **clusters;
    int *cluster_indexes;
    T *v;
    clusters = (LL**) calloc(K,sizeof(LL*));
    //assert(clusters!=NULL);
    if (clusters==NULL){
        return NULL;
    }
    cluster_indexes = (int*) calloc(N, sizeof(int));
    //assert(cluster_indexes);
    if (cluster_indexes==NULL){
        return NULL;
    }

    for (i = 0; i < K; ++i) {
        clusters[i] = init_d(d);
        if (clusters[i]==NULL){
            return NULL;
        }
    }
    /*2*/
    while (changed && iters++ < MAX_ITER) {

        changed=0;
        /*a*/
        for (i = 0; i < N; ++i) {
            v = observations[i];
            insert_index = findClosestCentroid(v, centroids, K, d);
            cluster_indexes[i] = insert_index;
            status = add(&(clusters[insert_index]), v);
            if (status){
                return NULL;
            }
        }
        /*b*/
        for (i=0;i<K;++i){
            tmp = updateCentroid(&(centroids[i]),&(clusters[i]),d);
            if (tmp==-1){
                return NULL;
            }
            changed+=tmp;
        }
    }
    return cluster_indexes;
}
static T* listToPointer(PyObject* list){
/**
* converting a python list into an array
* list: python list
* returns: pointer to the array fo the list
**/
    Py_ssize_t n = PyList_Size(list);
    Py_ssize_t i;
    T* vect = calloc(n,sizeof(T));
    PyObject * item;
    for (i=0;i<n;i++){
        item = PyList_GetItem(list, i);
        if (!PyNumber_Check(item))
        {
            free(vect);
            return NULL;
        }
        vect[i] = PyFloat_AsDouble(item);
    }
    return vect;
}


static T** listOfListsToPointer(PyObject* list){
/**
* converting a python nested list into a multi-dimensional array
* list: list of lists in python
* returns: pointer to pointers to arrays
**/

    Py_ssize_t i,j, n = PyList_Size(list);
    T** observations = calloc(n,sizeof(T*));
    T* vect;
    PyObject* sub_list;
    for(i=0;i<n;i++){
        sub_list = PyList_GetItem(list,i);
        if (!PyList_Check(sub_list)){
            for (j=0;j<=i;j++){
                free(observations[i]);
            }
            free(observations);
            return NULL;
        }
        vect = listToPointer(sub_list);
        if (!vect){
            for (j=0;j<=i;j++){
                free(observations[i]);
            }
            free(observations);
            return NULL;
        }
        observations[i]=vect;
    }
    return observations;
}

static PyObject * CreateList(int* list, int n){
/**
* create a python list from array of integers of size n
* list: array of integers
* n: size of list
* returns: python list of the array given
**/
    int i;
    PyObject* ret = PyList_New(n);
    PyObject *cur;
    for (i=0;i<n;i++){
        cur = Py_BuildValue("i",list[i]);
        PyList_SetItem(ret, i, cur);
    }

    return ret;
}

/* CAPI */
static PyObject * CAPI_KMEANS(PyObject* self, PyObject* args) {
/**
the kmeans algorithm as was presented in assignment 1
args contains:
observations: the points to classify
centroids: list of starting centroids
K: number of centers
N: number of points
d: dimension of points
max_iter: maximum iterations for the kmeans algorithm
returns: the indexes for each observation as obtained from the kmeans algorithm, aka, indexes_empirical[i] is the cluster of obs[i] in the algorithm
**/
    PyObject *obser_list, *cent_list;
    T **observations, **centroids;
    int *cluster_indexes;
    int K, N, d, MAX_ITER;
    int i;
    PyObject *ret = NULL;
    if (!PyArg_ParseTuple(args, "OOiiii", &obser_list, &cent_list, &K, &N, &d, &MAX_ITER)) {
        Py_RETURN_NONE;
    }
    if (!PyList_Check(obser_list) || (!PyList_Check(cent_list))) {
        Py_RETURN_NONE;
    }
    observations = listOfListsToPointer(obser_list);

    if (!observations) {
        Py_RETURN_NONE;
    }
    centroids = listOfListsToPointer(cent_list);
    if (!centroids) {
        for (i = 0; i < N; ++i) {
            free(observations[i]);
        }
        free(observations);
        Py_RETURN_NONE;
    }
    cluster_indexes = K_MEANS(observations, centroids, K, N, d, MAX_ITER);
    if (cluster_indexes) {
        ret = CreateList(cluster_indexes, N);
    }
    for (i=0;i<K;i++){
        free(centroids[i]);
    }
    free(centroids);

    for (i=0;i<N;++i){
        free(observations[i]);
    }
    free(observations);
    //free(cluster_indexes);
    if (cluster_indexes) {
        return ret;
    }
    Py_RETURN_NONE;

}
/* from here on out everything is defined as seen in class */
static PyMethodDef methodDef [] = {
        {"kmeans",
                CAPI_KMEANS,
                   METH_VARARGS,
                     NULL},
        {NULL,NULL,0,NULL}
};

static struct PyModuleDef moduleDef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp",
        NULL,
        -1,
        methodDef
};


PyMODINIT_FUNC PyInit_mykmeanssp(void){
    PyObject *m;
    m = PyModule_Create(&moduleDef);
    if (!m){
        return NULL;
    }
    return m;
}