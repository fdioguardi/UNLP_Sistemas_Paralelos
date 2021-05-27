#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define PI 3.14159265358979323846
#define DOUBLE_PI PI * 2

double tiempopara, tiemposecu;

/*****************************************************************/

/* Compartidas */
int NUM_THREADS, strip_size_by_threads, block_size_by_threads, N, bs, size, threads_in_barrier;
double *A, *B, *C, *T, *R, *M, *R1, *R2, *RA, *RB, average1, average2;
pthread_mutex_t average_lock1, average_lock2, average_lock;
pthread_barrier_t barrier;

/*****************************************************************/

// Para calcular tiempo - Función de la cátedra
double dwalltime(){
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec / 1000000.0;
	return sec;
}

// Para generar un número aleatorio - Función de la cátedra
double randFP(double min, double max) {
	double range = (max - min);
	double div = RAND_MAX / range;
	return min + (rand() / div);
}

/*****************************************************************/

void* calculate(void* identifier) {
	int start_block, i, j, k, f, c, h, offset_i, offset_j, offset_c, offset_f, mini_row_index, pos;
	int id = *((int *)identifier);
	start_block = block_size_by_threads * id;
	double num, aSin, aCos, *ablk, *bblk, *cblk;
	double localAverage1 = 0;
	double localAverage2 = 0;

	int end_block = block_size_by_threads * (id + 1);

	int start_strip = strip_size_by_threads * id;
	int end_strip = strip_size_by_threads * (id + 1);

	for (i = start_block; i < end_block; i++) {
		offset_i = i * N;
		for (j = 0; j < N; j++) {
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

// Main del programa
int main(int argc, char* argv[]){

	double *C_secuencial, timetick_start, timetick_end, timetick_end_secuencial, timetick_start_secuencial;
	int i;

	// Controla los argumentos al programa
	if ((argc != 4)
			|| ((N = atoi(argv[1])) <= 0)
			|| ((NUM_THREADS = atoi(argv[2])) <= 0)
			|| ((bs = atoi(argv[3])) <= 0)
			|| ((N % bs) != 0))
	{
		printf("\nError en los parámetros. Usage: ./%s N T BS (N debe ser multiplo de BS)\n", argv[0]);
		exit(1);
	}

	// Aloca memoria para las matrices
	size = N*N;
	A = (double*)malloc(sizeof(double)*size); // ordenada por columnas
	B = (double*)malloc(sizeof(double)*size); // ordenada por columnas
	C = (double*)malloc(sizeof(double)*size); // ordenada por filas
	C_secuencial = (double*)malloc(sizeof(double)*size); // ordenada por filas
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
		A[i] = randFP(0, 5);
		B[i] = randFP(0, 5);
		T[i] = randFP(0, 5);
		M[i] = randFP(0, DOUBLE_PI);
		R1[i] = 0;
		R2[i] = 0;
		RA[i] = 0;
		RB[i] = 0;
	}

	average1 = 0;
	average2 = 0;


	/*********************************** Pthread ******************************/


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

	// Detiene el timer
	timetick_end = dwalltime();

	// Tear down mutex
	pthread_mutex_destroy(&average_lock1);
	pthread_mutex_destroy(&average_lock2);
	pthread_mutex_destroy(&average_lock);

	// Tear down barrera
	pthread_barrier_destroy(&barrier);

	printf("Tiempo en segundos con Pthread %f\n", timetick_end - timetick_start);
	tiempopara = timetick_end - timetick_start;


	/*********************************** Secuencial ******************************/

	double num, aSin, aCos, *ablk, *bblk, *cblk;
	int j, k, offset_i, offset_j, row_index, f, c, h, offset_f, offset_c, mini_row_index;

	average1 = 0;
	average2 = 0;

	// Resetea las matrices R1, R2, RA, y RB
	for(i = 0; i < size ; i++) {
		R1[i] = 0;
		R2[i] = 0;
		RA[i] = 0;
		RB[i] = 0;
	}

	// Inicia el timer
	timetick_start_secuencial = dwalltime();

	// Calcula Rs y su promedio
	for (i = 0; i < size; i++) {
		num = (1 - T[i]);
		aSin = sin(M[i]);
		aCos = cos(M[i]);
		R1[i] = num * (1 - aCos) + (T[i] * aSin);
		R2[i] = num * (1 - aSin) + (T[i] * aCos);

		average1 += R1[i];
		average2 += R2[i];
	}

	// RA = R1 * A
	for (i = 0; i < N; i += bs)
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
						}
					}
				}
			}
		}
	}

	// RB = R2 * B
	for (i = 0; i < N; i += bs)
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
						}
					}
				}
			}
		}
	}


	average1 = (average1 / size) * (average2 / size);

	// Calcula C_secuencial
	for (i = 0; i < size; i++) {
		C_secuencial[i] = T[i] + average1 * (RA[i] + RB[i]);
	}

	// Detiene el timer
	timetick_end_secuencial = dwalltime();

	printf("Tiempo en segundos del secuencial %f\n", timetick_end_secuencial - timetick_start_secuencial);
	tiemposecu = timetick_end_secuencial - timetick_start_secuencial;

	printf("Speedup: %f\n", tiemposecu/tiempopara);
	printf("Eficiencia: %f\n", (tiemposecu/tiempopara)/NUM_THREADS);

	// Comprueba que C y C_secuencial sean iguales
	int check = 1;
	for (i = 0; i < size; i++) {
		if (fabs(C[i] - C_secuencial[i]) > 0.000001) {
			printf("C: paralelo: %f, secuencial: %f, indice: %d\n", C[i], C_secuencial[i], i);
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
	free(C_secuencial);
	free(T);
	free(R);
	free(M);
	free(R1);
	free(R2);
	free(RA);
	free(RB);

	return(0);
}
