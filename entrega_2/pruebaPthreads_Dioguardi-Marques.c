#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define PI 3.14159265358979323846
#define DOUBLE_PI PI * 2

/*****************************************************************/

/* Compartidas */
int NUM_THREADS, strip_size_by_threads, block_size_by_threads, N, bs, size, threads_in_barrier;
double *A, *B, *C, *T, *R, *M, *R1, *R2, *RA, *RB, average1, average2;
pthread_mutex_t average_lock1, average_lock2, average_lock;
pthread_barrier_t barrier;

/*****************************************************************/


void* calculate(void* identifier) {
	int start_block, i, j, k, f, c, h, offset_i, offset_j, offset_c, offset_f, mini_row_index, pos;
	int id = *((int *)identifier);
	start_block = block_size_by_threads * id;
	double num, aSin, aCos, *ablk, *bblk, *cblk;
	double localAverage1 = 0;
	double localAverage2 = 0;

	double end_block = block_size_by_threads * (id + 1);

	double start_strip = strip_size_by_threads * id;
	double end_strip = strip_size_by_threads * (id + 1);

	// Calcula R1 y R2 y sus promedios
	for (i = start_block; i < end_block; i++) {
		offset_i = i * N;
		for (i = 0; i < N; i++) {
			pos = offset_i + j;

			num = (1 - T[pos]);
			aSin = sin(M[pos]);
			aCos = cos(M[pos]);
			R1[i] = num * (1 - aCos) + (T[pos] * aSin);
			R2[i] = num * (1 - aSin) + (T[pos] * aCos);

			localAverage1 += R1[i];
			localAverage2 += R2[i];
		}
	}

	pthread_mutex_lock(&average_lock1);
	average1 += localAverage1;
	pthread_mutex_unlock(&average_lock1);

	pthread_mutex_lock(&average_lock2);
	average2 += localAverage2;
	pthread_mutex_unlock(&average_lock2);

	for (i = start_block; i < end_block; i += bs)
	{
		offset_i = i * N;
		for (j = 0; j < N; j += bs)
		{
			offset_j = j * N;
			cblk = &RA[offset_i + j];

			for  (k = 0; k < N; k += bs)
			{
				ablk = &R1[offset_i + k];
				bblk = &A[offset_j + k];

				for (f = 0; f < bs; f++)
				{
					offset_f = f * N;
					for (c = 0; c < bs; c++)
					{
						offset_c = c * N;
						mini_row_index = offset_f + c;

						for  (h = 0; h < bs; h++)
						{
							cblk[mini_row_index] += ablk[offset_f + h] * bblk[offset_c + h];
	} } } } } }

	for (i = start_block; i < end_block; i += bs)
	{
		offset_i = i * N;
		for (j = 0; j < N; j += bs)
		{
			offset_j = j * N;
			cblk = &RB[offset_i + j];

			for  (k = 0; k < N; k += bs)
			{
				ablk = &R2[offset_i + k];
				bblk = &B[offset_j + k];

				for (f = 0; f < bs; f++)
				{
					offset_f = f * N;
					for (c = 0; c < bs; c++)
					{
						offset_c = c * N;
						mini_row_index = offset_f + c;

						for  (h = 0; h < bs; h++)
						{
							cblk[mini_row_index] += ablk[offset_f + h] * bblk[offset_c + h];
	} } } } } }

	if (threads_in_barrier + 1 < NUM_THREADS) {
		pthread_mutex_lock(&average_lock);
		threads_in_barrier++;
		pthread_mutex_unlock(&average_lock);
	}
	else {
		average1 = (average1 / size) * (average2 / size);
	}

	pthread_barrier_wait(&barrier);

	// Calcula C
	for (i = start_strip; i < end_strip; i++) {
		C[i] = T[i] + average1 * (RA[i] + RB[i]);
	}

	pthread_exit(0);
}

/*****************************************************************/

// Para calcular tiempo - Función de la cátedra
double dwalltime(){
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec / 1000000.0;
	return sec;
}

/*****************************************************************/

// Main del programa
int main(int argc, char* argv[]){
	double average, timetick_start, timetick_end;
	int i;

	// Controla los argumentos al programa
	if ((argc != 4)
		|| ((N = atoi(argv[1])) <= 0)
		|| ((NUM_THREADS = atoi(argv[2])) <= 0)
		|| ((bs = atoi(argv[3])) <= 0)
		|| ((N % bs) != 0))
	{
		printf("\nError en los parámetros. Usage: ./%s N T B\n", argv[0]);
		exit(1);
	}

	// Aloca memoria para las matrices
	size = N*N;
	A = (double*)malloc(sizeof(double)*size); // ordenada por columnas
	B = (double*)malloc(sizeof(double)*size); // ordenada por columnas
	C = (double*)malloc(sizeof(double)*size); // ordenada por filas
	T = (double*)malloc(sizeof(double)*size); // ordenada por filas
	R = (double*)malloc(sizeof(double)*size); // ordenada por filas
	M = (double*)malloc(sizeof(double)*size); // ordenada por filas
	R1 = (double*)malloc(sizeof(double)*size); // ordenada por filas
	R2 = (double*)malloc(sizeof(double)*size); // ordenada por filas
	RA = (double*)malloc(sizeof(double)*size); // ordenada por filas
	RB = (double*)malloc(sizeof(double)*size); // ordenada por filas

	// Inicializa el randomizador
	time_t t;
	srand((unsigned) time(&t));

	// Inicializa las matrices A, B, T, M, R1, R2, RA, y RB
	for(i = 0; i < size ; i++) {
		A[i] = 1.0;
		B[i] = 1.0;
		T[i] = 1.0;
		M[i] = PI/2;
		R1[i] = 0;
		R2[i] = 0;
		RA[i] = 0;
		RB[i] = 0;
	}

	// Setup mutex
	pthread_mutex_init(&average_lock1, NULL);
	pthread_mutex_init(&average_lock2, NULL);
	pthread_mutex_init(&average_lock, NULL);

	// Setup barrera
	pthread_barrier_init(&barrier, NULL, NUM_THREADS);

	// Setup hilos
	pthread_t threads[NUM_THREADS];
	int ids[NUM_THREADS];

	block_size_by_threads = N / NUM_THREADS;
	strip_size_by_threads = size / NUM_THREADS;
	threads_in_barrier = 0;

	// Inicia el timer (hacerlo antes o despues de setup de hilos)
	timetick_start = dwalltime();

	// Crea hilos y calcula R
	for (i = 0; i < NUM_THREADS; i++) {
		ids[i] = i;
		pthread_create(&threads[i], NULL, calculate, &ids[i]);
	}

	// Une hilos
	for (i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	// Tear down mutex
	pthread_mutex_destroy(&average_lock1);
	pthread_mutex_destroy(&average_lock2);
	pthread_mutex_destroy(&average_lock);

	// Tear down barrera
	pthread_barrier_destroy(&barrier);

	// Detiene el timer
	timetick_end = dwalltime();

	printf("Tiempo en segundos %f\n", timetick_end - timetick_start);

	// Verifica el resultado
	int check = 1;
	double correct_result = 1;
	for (i = 0; i < size; i++) {
		if (fabs(C[i] - correct_result) > 0.000001) {
			check = 0;
			break;
		}
	}

	if (check) {
		printf("Multiplicacion de matrices resultado correcto\n");
	}
	else {
		printf("Multiplicacion de matrices resultado erroneo\n");
	}

	// Libera memoria
	free(A);
	free(B);
	free(C);
	free(T);
	free(R);
	free(M);
	free(RA);
	free(RB);
	free(R1);
	free(R2);

	return(0);
}
